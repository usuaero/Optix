from collections import OrderedDict
import numpy as np
import os
import shutil
import time
import multiprocessing
import classes as c

np.set_printoptions(precision = 14)

zero = 1.0e-20

def minimize(fun,x0,**kwargs):
    """Minimize a scalar function in one or more variables

    Inputs
    ------

        fun(callable)
        - Objective to be minimized. Must be a scalar function:
        def fun(x,*args):
            return float
        where x is a vector of the design variables and *args is all
        other parameters necessary for calling the function.

        x0(array-like,shape(n,))
        - A starting guess for the independent variables. May be
        a list or numpy array. All variables must be represented as
        minimize determines the number of variables from the length
        of this vector.

        args(tuple,optional)
        - Arguments to be passed to the objective function.

        method(str,optional)
        - Method to be used by minimize to find the minimum of the
        objective function. May be one of the following:
            Unconstrained problem:
                "bgfs" - quasi-Nexton with BGFS Hessian update
            Constrained problem:
                "sqp" - sequential quadratic programming
                "grg" - generalized reduced gradient
        If no method is specified, minimize will choose either "bgfs"
        or "sqp", based on whether constraints were given.

        grad(callable,optional)
        - Returns the gradient of the objective function at a specified
        point. Definition is the same as fun() but must return array-like,
        shape(n,). If not specified, will be estimated using a finite-
        difference approximation.

        hess(callable,optional)
        - Returns the Hessian of of the objective function at a specified
        point. Definition is the same as fun() but must return array-like,
        shape(n,n). If not specified, will be estimated using a finite-
        difference approximation.
        NOTE: bgfs, sqp, and grg do not require direct Hessian evaluations,
        so this functionality is not defined at this time.

        bounds(sequence of tuple,optional)
        - Bounds on independent variables. Can only be used with constrained
        methods. Should be a sequence of (min,max) pairs for each element in
        x. Use -numpy.inf or numpy.inf to specify mo bound.

        constraints(list of {Constraint,dict}, optional)
        - Constraints on the design space. Can only be used with constrained
        methods. Given as a list of dictionaries, each having the following
        keys:
            type (str)
                Constraint type; either 'eq' for equality or 'ineq' for
                inequality; equality means the constraint function must
                equate to 0 and inequality means the constraint function
                must be positive.
            fun (callable)
                Value of the constraint function. Must return a scalar. May 
                only have one argument, being an array of the design variables.
            grad (callable,optional)
                Returns the gradient of the constraint function at a
                specified point. Must return array-like, shape(n,). May
                only have one argument, being an array of the design variables.

        termination_tol(float,optional)
        - Point at which the optimization will quit. Execution terminates
        if the change in x for any step becomes less than the termination
        tolerance. Defaults to 1e-12.

        verbose(bool,optional)
        - If set to true, extra information about each step of the
        optimization will be printed to the command line.

        cent_diff(bool,optional)
        - Flag for setting finite-difference approximation method. If set
        to false, a forward-difference approximation will be used. Otherwise
        defaults to a central-difference.

        file_tag(str,optional)
        - Tag to be appended to the output filenames. If not specified,
        output files will be overwritten each time minimize() is called.
        Output files may still be overwritten if file_tag does not change
        with each call.

        max_processes(int,optional)
        - Maximum number of processes to be used in multiprocessing. Defaults
        to 1.

        dx(float,optional)
        - Step size to be used in finite difference methods. Defaults to 0.001

        default_alpha(float,optional)
        - Initial step size to be used in the line search. Defaults to 1.

        line_search(string,optional)
        - Specifies which type of line search should be conducted in the search
        direction. The following types are possible:
            "bracket" - backets minimum and finds vertex of parabola formed by
            3 minimum points
            "quadratic" - fits a quadratic to the search points and finds vertex
        Defaults to bracket. Not defined for SQP algorithm.

        n_search(int,optional)
        -Number of points to be considered in the search direction. Defaults to
        8. Not defined for SQP algorithm.

        max_iterations(int,optional)
        -Maximum number of iterations for the optimization algorithm. Defaults to
        inf.

        wolfe_armijo(float,optional)
        -Value of c1 in the Wolfe conditions. Defaults to 1e-4.

        wolfe_curv(float,optional)
        -Value of c2 in the Wolfe conditions. Defaults to 0.9 for BGFS.

    Output
    ------

        Optimum(OptimizerResult)
        - Object containing information about the result of the optimization.
        Attributes include:
            x(array-like,shape(n,))
                Point in the design space where the optimization ended.
            f(scalar)
                Value of the objective function at optimum.
            success(bool)
                Indicates whether the optimizer exitted normally.
            message(str)
                Message about how the optimizer exitted.
            obj_calls(int)
                How many calls were made to the objective function during optimization.
            cstr_calls(array-like(n_cstr),int)
                How many calls were made to each constraint function during optimization.

    """

    #Initialize settings
    settings = c.Settings(**kwargs)

    #Initialize objective function
    grad = kwargs.get("grad")
    hess = kwargs.get("hess")
    f = c.Objective(fun,settings,grad=grad,hess=hess)

    #Initialize design variables
    n_vars = len(x0)
    x_start = np.reshape(x0,(n_vars,1))

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
                constr = c.Constraint(constraint["type"],constraint["fun"],settings,grad=grad)
                g.append(constr)
        for constraint in constraints:
            if constraint["type"] == "eq":
                grad = constraint.get("grad")
                constr = c.Constraint(constraint["type"],constraint["fun"],settings,grad=grad)
                g.append(constr)
        g = np.array(g)
    else:
        g = None
        n_cstr = 0
        n_ineq_cstr = 0

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

    #Print setup information to command line
    printSetup(n_vars,x_start,bounds,n_cstr,n_ineq_cstr,settings)
    print(opt_header)

    # Drive to the minimum
    opt = find_minimum(f,g,x_start,settings)
    opt.obj_calls = f.eval_calls.value
    for i in range(n_cstr):
        opt.cstr_calls.append(g[i].eval_calls.value)
    
    # Run the final case
    return opt


def find_minimum(f,g,x_start,settings):
    """Calls specific optimization algorithm as needed"""
    if settings.method == "bgfs":
        return bgfs(f,x_start,settings)
    elif settings.method == "sqp":
        return sqp(f,g,x_start,settings)
    elif settings.method == "grg":
        return grg(f,g,x_start,settings)
    else:
        raise ValueError("Method improperly specified.")


def bgfs(f,x_start,settings):
    """Performs quasi-Newton, unconstrained optimization"""

    if settings.verbose: print("Beginning simple unconstrained BGFS optimization.")
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

        f0 = f.f(x0)
        del_f0 = f.del_f(x0)
        append_file(iter,o_iter,i_iter,f0,settings.alpha_d,mag_dx, x0, del_f0, settings)
        N0 = np.eye(n)
        s = -np.dot(N0,del_f0)
        s = s/np.linalg.norm(s)
        x1,f1,alpha = line_search(x0,f0,s,del_f0,f,settings)
        delta_x0 = x1-x0
        mag_dx = np.linalg.norm(delta_x0)

        while iter < settings.max_iterations and mag_dx > settings.termination_tol:
            i_iter += 1
            iter += 1

            # Update gradient and output file
            del_f1 = f.del_f(x1)
            append_file(iter,o_iter,i_iter,f1,alpha,mag_dx,x1,del_f1,settings)

            # Check second Wolfe condition
            if np.inner(s.T,del_f1.T) < settings.wolfe_curv*np.inner(s.T,del_f0.T):
                print("Wolfe condition ii not satisfied.")
                break

            # Update Hessian
            gamma0 = del_f1-del_f0
            denom = np.matrix(delta_x0).T*np.matrix(gamma0)
            NG = np.matrix(N0)*np.matrix(gamma0)
            A = np.asscalar(1+np.matrix(gamma0).T*NG/denom)
            B = (np.matrix(delta_x0)*np.matrix(delta_x0).T/denom)
            C = (np.matrix(delta_x0)*np.matrix(gamma0).T*np.matrix(N0)+NG*np.matrix(delta_x0).T)/denom
            N1 = N0+A*B-C

            # Determine new search direction and perform line search
            s = -np.dot(N1,del_f1)
            s = s/np.linalg.norm(s)
            x2,f2,alpha = line_search(x1,f1,s,del_f1,f,settings)
            delta_x1 = x2-x1
            mag_dx = np.linalg.norm(delta_x1)

            # Update variables for next iteration
            x0 = x1
            f0 = f1
            del_f0 = del_f1
            delta_x0 = delta_x1
            
            x1 = x2
            f1 = f2

    del_f2 = f.del_f(x2)
    append_file(iter,o_iter,i_iter,f2,alpha,mag_dx,x2,del_f2,settings)
    return c.OptimizerResult(f2,x2,True,"Optimizer exitted normally.",iter)
    
    
def line_search(x0,f0,s,del_f0,f,settings):
    """Perform line search to find a minimum in the objective function.

    This subroutine evaluates the objective function multiple times in the
    direction of s. It either selects the minimum and two bracketting points
    or all points and fits a parabola to these to find the vertex. The step 
    length is adjusted if no bracketted minimum can be found.

    Inputs
    ------

        x0(ndarray(n,))
        -Point at which to start the line search.

        f0(float)
        -Objective function value at initial point.

        s(ndarray(n,))
        -Search direction.

        del_f0(ndarray(n,))
        -Gradient at x0

        f(Objective)
        -Objective function object.

        settigns(Settings)
        -Settings object.

    Outputs
    -------

        x1(ndarray(n,))
        -Optimum point in the search direction.

        f1(float)
        -Value of objective function at optimum point.

        alpha(float)
        -Step size used to find optimum.

    """
    if settings.verbose:
        print('line search ----------------------------------------------------------------------------')

    alpha = np.float(np.copy(settings.alpha_d))
    alpha_mult = settings.n_search/2.0

    while True:
        x_search = [x0+s*alpha*i for i in range(1,settings.n_search+1)]
        with multiprocessing.Pool(processes=settings.max_processes) as pool:
            f_search = pool.map(f.f,x_search)
        x_search = [x0]+x_search
        f_search = [f0]+f_search

        if settings.verbose:
            for i in range(settings.n_search + 1):
                out = '{0:5d}'.format(i)
                for j in range(len(x0)):
                    out += ', {0:15.7E}'.format(np.asscalar(x_search[i][j]))
                out += ', {0:15.7E}'.format(f_search[i])
                print(out)

        # Check for invalid results
        if np.isnan(f_search).any():
            print('Found NaN in line search at the following design point:')
            print(x_search[np.where(np.isnan(f_search))])
            break

        # Check for stopping criteria
        if f_search[1] > f_search[0] and alpha < settings.termination_tol:
            print('Alpha within stopping tolerance: alpha = {0}'.format(alpha))
            return x0,f0,alpha
        
        # Check for plateau
        if min(f_search) == max(f_search):
            print('Objective function has plateaued')
            break
            
        # See if alpha needs to be adjusted
        min_ind = f_search.index(min(f_search))
        if min_ind == 0:
            if settings.verbose: print('Too big of a step. Reducing alpha')
            alpha /= alpha_mult
        elif min_ind == settings.n_search:
            if settings.verbose: print('Too small of a step. Increasing alpha')
            alpha *= alpha_mult
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
    mag_dx = 1
    while iter < settings.max_iterations and mag_dx > settings.termination_tol:
        if settings.verbose: print("Setting Lagrangian Hessian to the identity matrix.")
        o_iter += 1
        i_iter = 1
        iter += 1

        # Create quadratic approximation
        f0 = f.f(x0)
        del_f0 = f.del_f(x0)
        g0 = np.zeros((n_cstr,1))
        del_g0 = np.zeros((n_vars,n_cstr))
        for i in range(n_cstr):
            g0[i] = g[i].g(x0)
            del_g0[:,i] = g[i].del_g(x0).flatten()
        del_2_L0 = np.identity(n_vars)
        append_file(iter,o_iter,i_iter,f0,mag_dx,mag_dx,x0,del_f0,settings,g=g0,del_g=del_g0)
            
        # Create the system of equations to solve for delta_x and lambda
        n_eqns = n_vars+n_cstr # At first assume all constraints are binding
        A = np.zeros((n_eqns,n_eqns))
        b = np.zeros((n_eqns,1))
        for i in range(n_eqns):
            for j in range(n_eqns):
                if i<n_vars and j<n_vars:
                    A[i][j] = del_2_L0[i][j]
                elif i<n_vars:
                    A[i][j] = -del_g0[i]
                elif j<n_vars:
                    A[i][j] = del_g0[j]
                else:
                    A[i][j] = 0
            if i<n_vars:
                b[i] = -del_f0[i]
            else:
                b[i] = -g0
        
        x_lambda = np.linalg.solve(A,b)
        delta_x = x_lambda[0:n_vars]
        l = x_lambda[n_vars:n_eqns]
        
        if l < 0: # Constraint is not binding
            print("Constraint is not binding!")
            A = np.zeros((n_vars,n_vars))
            b = np.zeros((n_vars,1))
            for i in range(n_vars):
                for j in range(n_vars):
                    A[i][j] = del_2_L0[i][j]
                b[i] = -del_f0[i]
            
            delta_x = np.linalg.solve(A,b)
            l = np.zeros((n_cstr,1))
        
        x1 = x0+delta_x
        P1 = f.f(x1)
        for i in range(n_cstr):
            P1 += l[i]*abs(g[i].g(x1))
        mag_dx = np.linalg.norm(delta_x)

        while P1 > f0:
            if settings.verbose: print("Stepped too far! Cutting step in half.")
            delta_x /= 2
            x2 = x1+delta_x
            P2 = f.f(x2)
            for i in range(n_cstr):
                P2 += l[i]*abs(g[0][i](x2))
        
        while mag_dx > settings.termination_tol:
            iter += 1
            i_iter += 1
        
            # Create quadratic approximation
            f1 = f.f(x1)
            del_f1 = f.del_f(x1)
            g1 = np.zeros((n_cstr,1))
            del_g1 = np.zeros((n_vars,n_cstr))
            for i in range(n_cstr):
                g1[i] = g[i].g(x1)
                del_g1[:,i] = g[i].del_g(x1).flatten()
        
            # Update the Lagrangian Hessain
            del_L0 = del_f0-np.asscalar(l)*del_g0
            del_L1 = del_f1-np.asscalar(l)*del_g1
            gamma_0 = np.matrix(del_L1-del_L0)
            first = gamma_0*gamma_0.T/(gamma_0.T*np.matrix(delta_x))
            second = del_2_L0*(np.matrix(delta_x)*np.matrix(delta_x).T)*del_2_L0/(np.matrix(delta_x).T*del_2_L0*np.matrix(delta_x))
            del_2_L1 = np.asarray(del_2_L0+first-second)

            append_file(iter,o_iter,i_iter,f1,mag_dx,mag_dx,x1,del_f1,settings,g=g1,del_g=del_g1)
        
            # Create the system of equations to solve for delta_x and lambda
            n_eqns = n_vars+n_cstr # At first assume all constraints are binding
            A = np.zeros((n_eqns,n_eqns))
            b = np.zeros((n_eqns,1))
            for i in range(n_eqns):
                for j in range(n_eqns):
                    if i<n_vars and j<n_vars:
                        A[i][j] = del_2_L1[i][j]
                    elif i<n_vars:
                        A[i][j] = -del_g1[i]
                    elif j<n_vars:
                        A[i][j] = del_g1[j]
                    else:
                        A[i][j] = 0
                if i<n_vars:
                    b[i] = -del_f1[i]
                else:
                    b[i] = -g1
            
            x_lambda = np.linalg.solve(A,b)
            delta_x = x_lambda[0:n_vars]
            l = x_lambda[n_vars:n_eqns]
        
            if l < 0: # Constraint is not binding
                A = np.zeros((n_vars,n_vars))
                b = np.zeros((n_vars,1))
                for i in range(n_vars):
                    for j in range(n_vars):
                        A[i][j] = del_2_L1[i][j]
                    b[i] = -del_f1[i]
                
                delta_x = np.linalg.solve(A,b)
                l = np.zeros((n_cstr,1))
            
            x2 = x1+delta_x
            P2 = f.f(x2)
            for i in range(n_cstr):
                P2 += l[i]*abs(g[i].g(x2))
        
            while P2 > P1:
                if settings.verbose: print("Stepped too far! Cutting step in half.")
                delta_x /= 2
                x2 = x1+delta_x
                P2 = f.f(x2)
                for i in range(n_cstr):
                    P2 += l[i]*abs(g[i].g(x2))
        
            x0 = x1
            x1 = x2
            f0 = f1
            del_f0 = del_f1
            g0 = g1
            del_g0 = del_g1
            del_2_L0 = del_2_L1
            P1 = P2
            mag_dx = np.linalg.norm(delta_x)
    
    f2 = f.f(x2)
    del_f2 = f.del_f(x2)
    g2 = np.zeros((n_cstr,1))
    del_g2 = np.zeros((n_vars,n_cstr))
    for i in range(n_cstr):
        g2[i] = g[i].g(x2)
        del_g2[:,i] = g[i].del_g(x2).flatten()
    append_file(iter,o_iter,i_iter,f2,mag_dx,mag_dx,x2,del_f2,settings,g=g2,del_g=del_g2)
    return c.OptimizerResult(f2,x2,True,"Optimizer exitted normally.",iter)


def append_file(iter,o_iter,i_iter,obj_fcn_value,alpha,mag_dx,design_point,gradient,settings,**kwargs):
    g = kwargs.get("g")
    del_g = kwargs.get("del_g")

    msg = '{0:4d}, {1:5d}, {2:5d}, {3: 20.13E}, {4: 20.13E}, {5: 20.13E}'.format(iter, o_iter, i_iter, obj_fcn_value, alpha, mag_dx)
    values_msg = msg
    for value in design_point:
        values_msg = ('{0}, {1: 20.13E}'.format(values_msg, np.asscalar(value)))
    if (g != None).any():
        for cstr in g[0]:
            values_msg = ('{0}, {1: 20.13E}'.format(values_msg, np.asscalar(cstr)))
    print(values_msg)
    with open(settings.opt_file, 'a') as opt_file:
        print(values_msg, file = opt_file)

    grad_msg = msg
    for grad in gradient:
        grad_msg = ('{0}, {1: 20.13E}'.format(grad_msg, np.asscalar(grad)))
    if (del_g != None).any():
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

