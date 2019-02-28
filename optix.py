from collections import OrderedDict
import numpy as np
import os
import shutil
import time
import multiprocessing as mp
import classes as c
import csv
import matplotlib.pyplot as plt
import itertools

np.set_printoptions(precision = 14)

zero = 1.0e-20

def minimize(fun,x0,**kwargs):
    """Minimize a scalar function in one or more variables"""

    #Initialize settings
    settings = c.Settings(**kwargs)

    #Initialize design variables
    n_vars = len(x0)
    x_start = np.reshape(x0,(n_vars,1))

    #Initialize multiprocessing
    with mp.Pool(settings.max_processes) as pool:
        manager = mp.Manager()
        queue = manager.Queue()

        #Initialize objective function
        grad = kwargs.get("grad")
        hess = kwargs.get("hess")
        f = c.Objective(fun,pool,queue,settings,grad=grad,hess=hess)

        #Initialize constraints
        constraints = kwargs.get("constraints")
        if constraints != None:
            n_cstr = len(constraints)
            n_ineq_cstr = 0
            g = []
            # Inequality constraints are stored first
            for constraint in constraints:
                if constraint["type"] == "ineq":
                    n_ineq_cstr += 1
                    grad = constraint.get("grad")
                    constr = c.Constraint(constraint["type"],constraint["fun"],pool,queue,settings,grad=grad)
                    g.append(constr)
            for constraint in constraints:
                if constraint["type"] == "eq":
                    grad = constraint.get("grad")
                    constr = c.Constraint(constraint["type"],constraint["fun"],pool,queue,settings,grad=grad)
                    g.append(constr)
            g = np.array(g)
        else:
            g = None
            n_cstr = 0
            n_ineq_cstr = 0

        if n_cstr-n_ineq_cstr > n_vars:
            raise ValueError("The problem is overconstrained")

        settings.n_cstr = n_cstr
        settings.n_ineq_cstr = n_ineq_cstr
        bounds = kwargs.get("bounds")

        #Begin formatting of output files
        opt_header = '{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}, {5:>20}'.format('iter', 'outer', 'inner', 'fitness', 'alpha', 'mag(dx)')
        for i in range(n_vars):
            opt_header += ', {0:>20}'.format('x'+str(i))
        for i in range(n_cstr):
            opt_header += ', {0:>20}'.format('g'+str(i))

        opt_filename = "optimize"+settings.file_tag+".txt"
        settings.opt_file = opt_filename
        with open(opt_filename, 'w') as opt_file:
            opt_file.write(opt_header + '\n')

        grad_header = '{0:>84}  {1:>20}'.format(' ','df')
        for i in range(n_cstr):
            grad_header += (', {0:>'+str(21*n_vars)+'}').format('dg'+str(i))
        grad_header += '\n{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}, {5:>20}'.format('iter', 'outer', 'inner', 'fitness', 'alpha', 'mag(dx)')
        for j in range(n_cstr+1):
            for i in range(n_vars):
                grad_header += ', {0:>20}'.format('dx'+str(i))
        
        grad_filename = "gradient"+settings.file_tag+".txt"
        settings.grad_file = grad_filename
        with open(grad_filename, 'w') as grad_file:
            grad_file.write(grad_header + '\n')

        eval_header = '{0:>20}'.format('f')
        for i in range(n_vars):
            eval_header += ', {0:>20}'.format('x'+str(i))
        eval_filename = "evaluations"+settings.file_tag+".txt"

        # Kick off evaluation storage process
        writer = pool.apply_async(eval_write,(eval_filename,eval_header,queue))

        # Print setup information to command line
        printSetup(n_vars,x_start,bounds,n_cstr,n_ineq_cstr,settings)
        print(opt_header)

        # Drive to the minimum
        opt = find_minimum(f,g,x_start,settings)

        # Finalize multiprocessing
        queue.put('kill')

    # Run the final case
    return opt


def find_minimum(f,g,x_start,settings):
    """Calls specific optimization algorithm as needed"""
    if settings.method == "bfgs":
        return bfgs(f,x_start,settings)
    elif settings.method == "sqp":
        return sqp(f,g,x_start,settings)
    elif settings.method == "grg":
        return grg(f,g,x_start,settings)
    else:
        raise ValueError("Method improperly specified.")


def bfgs(f,x_start,settings):
    """Performs quasi-Newton, unconstrained optimization"""

    if settings.verbose: print("Beginning simple unconstrained BFGS optimization.")
    iter = -1
    n = len(x_start)
    o_iter = -1
    mag_dx = 1
    x0 = np.copy(x_start)
    while iter < settings.max_iterations and mag_dx > settings.termination_tol:
        if settings.verbose: print("Setting Hessian to the identity matrix.")
        o_iter += 1
        i_iter = 0
        iter += 1

        f0_eval = f.pool.apply_async(f.f,(x0,))
        del_f0 = f.del_f(x0)
        f0 = f0_eval.get()
        append_file(iter,o_iter,i_iter,f0,0.0,0.0,x0,del_f0,settings)
        N0 = np.eye(n)

        # Determine search direction and perform line search
        s = -np.dot(N0,del_f0)
        alpha_guess = alpha_from_s(s,settings.n_search)
        mag_s = np.linalg.norm(s)
        s = s/mag_s
        if settings.verbose: print("Predicted optimum alpha: {0}".format(mag_s))
        x1,f1,alpha = line_search(x0,f0,s,del_f0,f,alpha_guess,settings)
        delta_x0 = x1-x0
        mag_dx = alpha

        while iter < settings.max_iterations and mag_dx > settings.termination_tol:
            i_iter += 1
            iter += 1

            # Update gradient and output file
            del_f1 = f.del_f(x1)
            append_file(iter,o_iter,i_iter,f1,alpha,mag_dx,x1,del_f1,settings)

            # Check for gradient termination
            if np.linalg.norm(del_f1)<settings.grad_tol:
                return c.OptimizerResult(f1,x1,True,"Gradient tolerance reached.",iter,f.eval_calls.value)

            # Check second Wolfe condition
            if np.inner(delta_x0.T,del_f1.T) < settings.wolfe_curv*np.inner(delta_x0.T,del_f0.T):
                print("Wolfe condition ii not satisfied.")
            #    break

            # Update Hessian
            N1 = get_N(N0,delta_x0,del_f0,del_f1)

            # Determine new search direction and perform line search
            s = -np.dot(N1,del_f1)
            alpha_guess = alpha_from_s(s,settings.n_search)
            mag_s = np.linalg.norm(s)
            s = s/mag_s
            if settings.verbose: print("Predicted optimum alpha: {0}".format(mag_s))
            x2,f2,alpha = line_search(x1,f1,s,del_f1,f,alpha_guess,settings)
            delta_x1 = x2-x1
            mag_dx = alpha

            # Update variables for next iteration
            x0 = x1
            f0 = f1
            del_f0 = del_f1
            delta_x0 = delta_x1
            
            x1 = x2
            f1 = f2

    return c.OptimizerResult(f2,x2,True,"Step tolerance reached.",iter,f.eval_calls.value)

def alpha_from_s(s,n_search):
    return np.linalg.norm(s)*2/n_search

def get_N(N0,delta_x0,del_f0,del_f1):
    gamma0 = del_f1-del_f0
    denom = np.matrix(delta_x0).T*np.matrix(gamma0)
    NG = np.matrix(N0)*np.matrix(gamma0)
    A = np.asscalar(1+np.matrix(gamma0).T*NG/denom)
    B = (np.matrix(delta_x0)*np.matrix(delta_x0).T/denom)
    C = (np.matrix(delta_x0)*np.matrix(gamma0).T*np.matrix(N0)+NG*np.matrix(delta_x0).T)/denom
    N1 = N0+A*B-C
    return N1
    
    
def line_search(x0,f0,s,del_f0,f,alpha,settings):
    """Perform line search to find a minimum in the objective function."""

    if settings.verbose:
        print('Line Search ----------------------------------------------------------------------------')
        print('Search Direction: {0}'.format(s))

    if settings.alpha_d is not None:
        alpha = settings.alpha_d

    if settings.verbose: print('Initial step size: {0}'.format(alpha))

    while True:
        x_search = [x0+s*alpha*i for i in range(1,settings.n_search+1)]
        with mp.Pool(processes=settings.max_processes) as pool:
            f_search = pool.map(f.f,x_search)
        x_search = [x0]+x_search
        f_search = [f0]+f_search

        if settings.verbose:
            for i in range(settings.n_search + 1):
                out = '{0:5d}, {1:15.7E}'.format(i,f_search[i])
                for j in range(len(x0)):
                    out += ', {0:15.7E}'.format(np.asscalar(x_search[i][j]))
                print(out)

        # Check for invalid results
        if np.isnan(f_search).any():
            print('Found NaN in line search at the following design point:')
            print(x_search[np.where(np.isnan(f_search))[0]])
            break

        # Check for stopping criteria
        if f_search[1] > f_search[0] and alpha < settings.termination_tol:
            if settings.verbose: print('Alpha within stopping tolerance: alpha = {0}'.format(alpha))
            return x0,f0,alpha
        
        # Check for plateau
        if min(f_search) == max(f_search):
            if settings.verbose: print('Objective function has plateaued')
            break
            
        # See if alpha needs to be adjusted
        min_ind = f_search.index(min(f_search))
        if min_ind == 0:
            if settings.verbose: print('Too big of a step. Reducing alpha by {0}'.format(settings.alpha_mult))
            alpha /= settings.alpha_mult
        elif min_ind == settings.n_search:
            if settings.verbose: print('Too small of a step. Increasing alpha by {0}'.format(settings.alpha_mult))
            alpha *= settings.alpha_mult
        else:
            break
    
    # Find optimum value of alpha
    a = [alpha*i for i in range(settings.n_search+1)]
    alpha_opt = find_opt_alpha(a,f_search,min_ind,settings)
    if settings.verbose: print('Final alpha = {0}'.format(alpha_opt))
    x1 = x0+s*alpha_opt
    f1 = f.f(x1)

    # Check first Wolfe condition (NOTE: does not affect execution)
    armijo = f0+settings.wolfe_armijo*alpha_opt*np.inner(s.T,del_f0.T)
    if f1 > armijo:
        print("Wolve condition i not satisfied.")
    return x1,f1,alpha_opt


def find_opt_alpha(a,f_search,min_ind,settings):
    # Quadratic method
    if settings.method == 'quadratic':
        q = quadratic(np.asarray(a),np.asarray(f_search))
        (alpha_opt,f_opt) = q.vertex()
        
        # If the quadratic fit is good, return its vertex
        if not (alpha_opt is None or alpha_opt < 0 or not q.convex() or q.rsq < settings.rsq_tol):
            return alpha_opt
    
    # If bracketting method is selected, or is quadratic method fails, find the vertex defined by 3 minimum points
    a1 = a[min_ind - 1]
    a2 = a[min_ind]
    a3 = a[min_ind + 1]
    f1 = f_search[min_ind - 1]
    f2 = f_search[min_ind]
    f3 = f_search[min_ind + 1]
    
    alpha_opt = (f1*(a2**2-a3*2)+f2*(a3**2-a1**2)+f3*(a1**2-a2**2))/(2*(f1*(a2-a3)+f2*(a3-a1)+f3*(a1-a3)))
    if alpha_opt > a3 or alpha_opt < a1:
        alpha_opt = a2
    return alpha_opt
    
    
def sqp(f,g,x_start,settings):
    """Performs Sequntial Quadratic Programming on a constrained optimization function."""
    
    # Initialization
    iter = 0
    o_iter = 0
    n_vars = len(x_start)
    n_cstr = settings.n_cstr
    n_ineq_cstr = settings.n_ineq_cstr
    
    x0 = np.copy(x_start)
    mag_dx = 1 # Ensures the loop executes at least once
    
    # Start outer iteration
    while iter < settings.max_iterations and mag_dx > settings.termination_tol:
        if settings.verbose: print("Setting Lagrangian Hessian to the identity matrix.")
        o_iter += 1
        i_iter = 1
        iter += 1

        # Create quadratic approximation
        f0_eval = f.pool.apply_async(f.f,(x0,))
        g0 = eval_constr(g,x0)
        del_f0,del_g0 = eval_grad(x0,f,g,n_vars,n_cstr)
        del_2_L0 = np.eye(n_vars)
        f0 = f0_eval.get()
        append_file(iter,o_iter,i_iter,f0,0.0,0.0,x0,del_f0,settings,g=g0,del_g=del_g0)
            
        # Estimate initial penalty function. We allow this to be artificially high.
        P0 = np.copy(f0)
        for constr in g0:
            if constr < 0:
                P0 -= constr

        # Get step
        delta_x,l,x1,f1,g1,P1 = get_delta_x(x0,f0,f,g,P0,n_vars,n_cstr,n_ineq_cstr,del_2_L0,del_f0,del_g0,g0,settings)
        
        mag_dx = np.linalg.norm(delta_x)

        # Start inner iteration
        while mag_dx > settings.termination_tol:
            first = False
            iter += 1
            i_iter += 1
        
            # Create quadratic approximation
            del_f1,del_g1 = eval_grad(x1,f,g,n_vars,n_cstr)
        
            # Update the Lagrangian Hessain
            del_2_L1 = get_del_2_L(del_2_L0,del_f0,del_f1,l,del_g0,del_g1,n_vars,n_cstr,delta_x)

            append_file(iter,o_iter,i_iter,f1,mag_dx,mag_dx,x1,del_f1,settings,g=g1,del_g=del_g1)
        
            # Get step
            delta_x,l,x2,f2,g2,P2 = get_delta_x(x1,f1,f,g,P1,n_vars,n_cstr,n_ineq_cstr,del_2_L1,del_f1,del_g1,g1,settings)

            # Setup variables for next iterations
            x0 = x1
            x1 = x2
            f0 = f1
            f1 = f2
            del_f0 = del_f1
            g0 = g1
            g1 = g2
            del_g0 = del_g1
            del_2_L0 = del_2_L1
            P1 = P2
            mag_dx = np.linalg.norm(delta_x)

            # The algorithm may be stuck at a level point outside of feasible space.
            if mag_dx<settings.termination_tol and P2>f2:
                if settings.verbose: print("Stuck at optimum outside of feasible space. Resetting BFGS update.")
                mag_dx = 1
                break # Reset BFGS
            
            # End of inner loop
        # End of outer loop
    
    # Evaluate final case
    iter += 1
    i_iter += 1
    del_f1 = f.del_f(x1)
    del_g1 = np.zeros((n_vars,n_cstr))
    for i in range(n_cstr):
        del_g1[:,i] = g[i].del_g(x1).flatten()
    append_file(iter,o_iter,i_iter,f1,mag_dx,mag_dx,x1,del_f1,settings,g=g1,del_g=del_g1)
    cstr_calls = []
    for i in range(n_cstr):
        cstr_calls.append(g[i].eval_calls.value)
    return c.OptimizerResult(f1,x1,True,"Optimizer exitted normally.",iter,f.eval_calls.value,cstr_calls)


def eval_grad(x0,f,g,n_vars,n_cstr):
    # Evaluate gradients at specified point
    del_f0 = f.del_f(x0)
    del_g0 = np.zeros((n_vars,n_cstr))
    for i in range(n_cstr):
        del_g0[:,i] = g[i].del_g(x0).flatten()
    return del_f0,del_g0


def get_del_2_L(del_2_L0,del_f0,del_f1,l,del_g0,del_g1,n_vars,n_cstr,delta_x):
    # BFGS update for Lagrangian Hessian
    del_L0 = np.copy(del_f0)
    del_L1 = np.copy(del_f1)
    for i in range(n_cstr):
        del_L0 -= np.asscalar(l[i])*np.reshape(del_g0[:,i],(n_vars,1))
        del_L1 -= np.asscalar(l[i])*np.reshape(del_g1[:,i],(n_vars,1))
    gamma_0 = np.matrix(del_L1-del_L0)
    first = gamma_0*gamma_0.T/(gamma_0.T*np.matrix(delta_x))
    second = del_2_L0*(np.matrix(delta_x)*np.matrix(delta_x).T)*del_2_L0/(np.matrix(delta_x).T*del_2_L0*np.matrix(delta_x))
    return np.asarray(del_2_L0+first-second)


def get_delta_x(x0,f0,f,g,P0,n_vars,n_cstr,n_ineq_cstr,del_2_L0,del_f0,del_g0,g0,settings):
    # Solve for delta_x and lambda given each possible combination of binding/non-binding constraints
    if settings.verbose: print("Penalty to beat: {0}".format(P0))

    # If a given combination has no negative Lagrangian multipliers corresponding to inequality constraints, the loop exits.
    # An equality constraint is always binding and its Lagrange multiplier my be any value.
    cstr_opts = [[True,False] for i in range(n_ineq_cstr)] + [[True] for i in range(n_ineq_cstr,n_cstr)]
    poss_combos = np.array(list(itertools.product(*cstr_opts)))
    for cstr_b in poss_combos:

        if sum(cstr_b)>n_vars: # At most, n constraints may be binding in n-dimensional space.
            continue

        if sum(cstr_b)>1:
            _,s,_ = np.linalg.svd(del_g0[:,cstr_b].T) # Check linear independence of constraint gradients.
            if (abs(s)<1e-14).any():
                continue

        delta_x, l = get_x_lambda(n_vars,n_cstr,del_2_L0,del_g0,del_f0,g0,cstr_b)

        x1 = x0+delta_x
        g1 = eval_constr(g,x1)
        if (g1[cstr_b==False]<0).any(): # Do not allow non-binding constraints to be violated.
            continue

        # Check if constraints assumed to be binding are actually non-binding.
        if not (l[:n_ineq_cstr].flatten()<0).any():
            if settings.verbose: print("Optimal combination found.")
            break
    else:
        # If an optimal combination is not found, relax the conditions by allowing non-binding constraints to be violated.
        if settings.verbose: print("Optimal combination not found. Allowing non-binding constraints to be violated.")
        for cstr_b in poss_combos:

            if sum(cstr_b)>n_vars: # At most, n constraints may be binding in n-dimensional space.
                continue

            if sum(cstr_b)>1:
                _,s,_ = np.linalg.svd(del_g0[:,cstr_b].T) # Check linear independence of constraint gradients.
                if (abs(s)<1e-14).any():
                    continue

            delta_x, l = get_x_lambda(n_vars,n_cstr,del_2_L0,del_g0,del_f0,g0,cstr_b)

            x1 = x0+delta_x
            g1 = eval_constr(g,x1)

            # Check if constraints assumed to be binding are actually non-binding.
            if not (l[:n_ineq_cstr].flatten()<0).any():
                if settings.verbose: print("Optimal combination found.")
                break

    if settings.verbose: print("Optimal combination of binding constraints: {0}".format(cstr_b))

    # Check penalty function at proposed optimum
    f1 = f.f(x1)
    P1 = np.copy(f1)
    for i in range(n_cstr):
        P1 += np.asscalar(abs(l[i])*abs(g1[i]))
    if settings.verbose: print("Point: {0}, Objective: {1}, Penalty: {2}".format(x1.flatten(),f1,P1))

    # Cut back step if the penalty function has increased
    while settings.strict_penalty and P1 > P0 and np.linalg.norm(delta_x) > settings.termination_tol:
        if settings.verbose: print("Stepped too far! Cutting step in half.")
        delta_x /= 2
        x1 = x0+delta_x
        f1 = f.f(x1)
        P1 = np.copy(f1)
        g1 = eval_constr(g,x1)
        for i in range(n_cstr):
            if i<n_ineq_cstr:
                if g1[i]>0: # We may have stepped back across a constraint, meaning it should no longer affect the penalty function
                    continue
                elif l[i]==0 and g1[i]<0: # We may have started violating a new constraint
                    P1 += abs(g1[i])
                    continue
            P1 += np.asscalar(abs(g1[i]))
        if settings.verbose: print("Point: {0}, Objective: {1}, Penalty: {2}".format(x1.flatten(),f1,P1))

    return delta_x,l,x1,f1,g1,P1


def get_x_lambda(n_vars,n_cstr,del_2_L0,del_g0,del_f0,g0,cstr_b):
    n_bind = np.asscalar(sum(cstr_b)) # Number of binding constraints

    # Create linear system to solve for delta_x and lambda
    A = np.zeros((n_vars+n_bind,n_vars+n_bind))
    b = np.zeros((n_vars+n_bind,1))
    A[:n_vars,:n_vars] = del_2_L0
    A[:n_vars,n_vars:] = -del_g0[:,cstr_b]
    A[n_vars:,:n_vars] = del_g0[:,cstr_b].T
    b[:n_vars] = -del_f0
    b[n_vars:] = np.reshape(-g0[cstr_b],(n_bind,1))
    
    # Solve system and parse solution
    x_lambda = np.linalg.solve(A,b)
    delta_x = x_lambda[0:n_vars]
    l_sol = x_lambda[n_vars:]
    l = np.zeros((n_cstr,1))
    l[cstr_b] = l_sol
    return delta_x,l


def eval_constr(g,x1):
    n_cstr = len(g)
    g1 = np.zeros(n_cstr)
    for i in range(n_cstr):
        g1[i] = g[i].g(x1)
    return g1


def append_file(iter,o_iter,i_iter,obj_fcn_value,alpha,mag_dx,design_point,gradient,settings,**kwargs):
    g = kwargs.get("g")
    del_g = kwargs.get("del_g")

    msg = '{0:4d}, {1:5d}, {2:5d}, {3: 20.13E}, {4: 20.13E}, {5: 20.13E}'.format(iter, o_iter, i_iter, obj_fcn_value, alpha, mag_dx)
    values_msg = msg
    for value in design_point:
        values_msg = ('{0}, {1: 20.13E}'.format(values_msg, np.asscalar(value)))
    if not g is None:
        for cstr in g:
            values_msg = ('{0}, {1: 20.13E}'.format(values_msg, np.asscalar(cstr)))
    print(values_msg)
    with open(settings.opt_file, 'a') as opt_file:
        print(values_msg, file = opt_file)

    grad_msg = msg
    for grad in gradient:
        grad_msg = ('{0}, {1: 20.13E}'.format(grad_msg, np.asscalar(grad)))
    if not del_g is None:
        for i in range(settings.n_cstr):
            for j in range(len(design_point)):
                grad_msg = ('{0}, {1: 20.13E}'.format(grad_msg, np.asscalar(del_g[j,i])))
    with open(settings.grad_file, 'a') as grad_file:
        print(grad_msg, file = grad_file)

def printSetup(n_vars,x_start,bounds,n_cstr,n_ineq_cstr,settings):
    print("\nOptix.py from USU AeroLab\n")
    
    print('---------- Variables ----------')
    print("Optimizing in {0} variables.".format(n_vars))
    print("Initial guess:\n{0}".format(x_start))
    if bounds != None:
        print("Variable bounds:\n{0}".format(bounds))
    print("")
    
    print('---------- Constraints ----------')
    print('{0} total constraints'.format(n_cstr))
    print('{0} inequality constraints'.format(n_ineq_cstr))
    print('{0} equality constraints'.format(n_cstr-n_ineq_cstr))
    print("")

    print('---------- Settings ----------')
    print('            method: {0}'.format(settings.method))
    print('     obj func args: {0}'.format(settings.args))
    print('     default alpha: {0}'.format(settings.alpha_d))
    print('    stopping delta: {0}'.format(settings.termination_tol))
    print('     max processes: {0}'.format(settings.max_processes))
    print(' dx (finite diffs): {0}'.format(settings.dx))
    print('          file tag: {0}'.format(settings.file_tag))
    print('           verbose: {0}'.format(settings.verbose))
    if settings.use_finite_diff:
        if settings.central_diff:
            print('using central difference approximation')
        else:
            print('using forward difference approximation')
    print('')

def eval_write(args):
    filename,header,q = args
    with open(filename,'w') as f:
        f.write(header+"\n")
        while True:
            msg = q.get()
            if msg=='kill':
                break
            f.write(msg+"\n")
