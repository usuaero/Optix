import numpy as np
import os
import shutil
import time
import multiprocessing as mp
import classes as c
import csv
import matplotlib.pyplot as plt
import itertools
from time import time
import warnings
import copy

np.set_printoptions(precision=14)
np.seterr(all='warn')

zero = 1.0e-20


def minimize(fun, x0, **kwargs):
    """Minimize a scalar function in one or more variables

        Parameters
        ------

            fun(callable)
            - Objective to be minimized. Must be a scalar function:
            def fun(x,*args):
                return float
            where x is a vector of the design variables and *args is all
            other parameters necessary for calling the function. Note that
            Optix will pass x as a numpy column matrix (i.e. shape(n_vars,1)).
            This should be taken into consideration when writing fun().

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

            grad_tol(float,optional)
            - Point at which the optimization will quit. Execution terminates
            if the norm of the gradient at any step becomes less than this
            tolerance. Defaults to 1e-12.

            verbose(bool,optional)
            - If set to true, extra information about each step of the
            optimization will be printed to the command line.

            cent_diff(bool,optional)
            - Flag for setting finite-difference approximation method. If set
            to false, a forward-difference approximation will be used. Otherwise,
            defaults to a central-difference.

            file_tag(str,optional)
            - Tag to be appended to the output filenames. If not specified,
            output files will be overwritten each time minimize() is called.
            Output files may still be overwritten if file_tag does not change
            with each call.

            max_processes(int,optional)
            - Maximum number of processes to be used in multiprocessing. Defaults
            fo 1.

            dx(float,optional)
            - Step size to be used in finite difference methods. Defaults to 0.001

            max_iterations(int,optional)
            - Maximum number of iterations for the optimization algorithm. Defaults to
            inf.

            num_avg(int,optional)
            - Number of times to run the objective function at each point. The objective
            value returned will be the average of all calls. This can be useful when
            dealing with noisy models. Defaults to 1.

        Returns
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

        Method Specific Arguments
        -------------------------

        BFGS

            n_search(int,optional)
            -Number of points to be considered in the search direction. Defaults to
            8.

            alpha_d(float,optional)
            - Step size to be used in line searches. Defaults to 1/n_search for the first
            iteration and the optimum step size from the previous iteration for all
            subsequent iterations.

            alpha_mult(float,optional)
            - Factor by which alpha is adjusted during each iteration of the line
            search. Defaults to n_search-1.

            line_search(string,optional)
            - Specifies which type of line search should be conducted in the search
            direction. The following types are possible:
                "bracket" - backets minimum and finds vertex of parabola formed by
                3 minimum points
                "quadratic" - fits a quadratic to the search points and finds vertex
            Defaults to bracket.

            rsq_tol(float,optional):
            - Specifies the necessary quality of the quadratic fit to the line search
            (only used if line_search is "quadratic"). The quadratic fit will only be
            accepted if the R^2 value of the fit is above rsq_tol. Defaults to 0.8.

            wolfe_armijo(float,optional)
            - Value of c1 in the Wolfe conditions. Defaults to 1e-4.

            wolfe_curv(float,optional)
            - Value of c2 in the Wolfe conditions. Defaults to 0.9.

            hess_init(float,optional)
            - Sets the value of the Hessian to hess_init*[I] for the first iteration of
            the BFGS update. Increasing this value may help speed convergence of some
            problems. Defaults to 1.

        SQP

            strict_penalty(bool,optional)
            - Specifies whether a given step in the optimization must result in a decrease in
            the penatly function. Setting this to false may help convergence of some problems
            and speed computation. Defaults to true.

            hess_init(float,optional)
            - Sets the value of the Hessian to hess_init*[I] for the first iteration of
            the BFGS update. Increasing this value may help speed convergence of some
            problems, but this is not recommended in most cases. Behavior is not stable if
            this value is less than 1. Defaults to 1.

        GRG

            n_search(int,optional)
            -Number of points to be considered in the search direction. Defaults to
            8.

            alpha_d(float,optional)
            - Step size to be used in line searches. If not specified, the step size
            is the optimum step size from the previous iteration.

            alpha_mult(float,optional)
            - Factor by which alpha is adjusted during each iteration of the line
            search. Defaults to n_search - 1

            cstr_tol(float,optional)
            - A constraint is considered to be binding if it evaluates to less than this
            number. Defaults to 1e-4.
    """

    # Initialize settings
    settings = c.Settings(**kwargs)

    # Initialize design variables
    n_vars = len(x0)
    x_start = np.reshape(x0, (n_vars, 1))

    # Initialize multiprocessing
    with mp.Pool(settings.max_processes) as pool:
        manager = mp.Manager()
        queue = manager.Queue()

        # Initialize objective function
        grad = kwargs.get("grad")
        hess = kwargs.get("hess")
        f = c.Objective(fun, pool, queue, settings, grad=grad, hess=hess)

        # Initialize constraints
        g, n_cstr, n_ineq_cstr = get_constraints(
            kwargs.get("constraints"), pool, queue, settings)
        settings.n_cstr = n_cstr
        settings.n_ineq_cstr = n_ineq_cstr
        bounds = kwargs.get("bounds")

        # Check constraints
        if n_cstr-n_ineq_cstr > n_vars:
            raise ValueError("The problem is overconstrained.")

        # Print setup information to command line
        print_setup(n_vars, x_start, bounds, n_cstr, n_ineq_cstr, settings)

        # Initialize formatting of output files
        format_output_files(n_vars, n_cstr, settings, pool, queue)

        # Kick off evaluation storage process (for more than one process)
        if settings.max_processes > 1:
            eval_header = '{0:>20}'.format('f')
            for i in range(n_vars):
                eval_header += ', {0:>20}'.format('x'+str(i))
            eval_filename = "evaluations"+settings.file_tag+".txt"

            writer = pool.apply_async(
                eval_write, (eval_filename, eval_header, queue))

        # Drive to the minimum
        opt = find_minimum(f, g, x_start, settings)

        # Kick off evaluation storage process (for only one process)
        if settings.max_processes == 1:
            eval_header = '{0:>20}'.format('f')
            for i in range(n_vars):
                eval_header += ', {0:>20}'.format('x'+str(i))
            eval_filename = "evaluations"+settings.file_tag+".txt"

            writer = pool.apply_async(
                eval_write, (eval_filename, eval_header, queue))

        # Kill evaluation printer process
        queue.put('kill')
        writer_success = writer.get()
        if not writer_success:
            print("Evaluation writer did not terminate successfully.")
        pool.close()
        pool.join()

    return opt


def find_minimum(f, g, x_start, settings):
    """Calls specific optimization algorithm as needed"""
    if settings.method == "bfgs":
        return bfgs(f, x_start, settings)
    elif settings.method == "sqp":
        return sqp(f, g, x_start, settings)
    elif settings.method == "grg":
        return grg(f, g, x_start, settings)
    else:
        raise ValueError("Method improperly specified.")


def bfgs(f, x_start, settings):
    """Performs quasi-Newton, unconstrained optimization"""

    # Initialize
    iter = -1
    n = len(x_start)
    o_iter = -1
    mag_dx = 1
    x0 = np.asmatrix(np.copy(x_start))

    # Outer loop. Sets the N matrix to [I].
    while iter < settings.max_iterations and mag_dx > settings.termination_tol:
        if settings.verbose:
            print("Setting Hessian to the identity matrix.")
        o_iter += 1
        i_iter = 0
        iter += 1

        f0_eval = f.pool.apply_async(f.f, (x0,))
        del_f0 = f.del_f(x0)
        f0 = f0_eval.get()
        append_file(iter, o_iter, i_iter, f0, 0.0, x0, del_f0, settings)
        N0 = np.eye(n)*settings.hess_init

        # Determine search direction and perform line search
        s = -np.dot(N0, del_f0)
        alpha_guess = 1/settings.n_search
        mag_s = np.linalg.norm(s)
        s = s/mag_s
        x1, f1, alpha, wolfe_satis = line_search(
            x0, f0, s, del_f0, f, alpha_guess, settings)
        delta_x0 = x1-x0
        mag_dx = alpha

        # Inner loop. Uses BFGS update for N.
        while iter < settings.max_iterations and mag_dx > settings.termination_tol:
            i_iter += 1
            iter += 1

            # Update gradient and output file
            del_f1 = f.del_f(x1)
            append_file(iter, o_iter, i_iter, f1, mag_dx, x1, del_f1, settings)

            # Check for gradient termination
            if np.linalg.norm(del_f1) < settings.grad_tol:
                return c.OptimizerResult(f1, x1, True, "Gradient tolerance reached.", iter, f.eval_calls.value)

            # Check second Wolfe condition. If not satisfied, reset BFGS update.
            if np.inner(delta_x0.T, del_f1.T) < settings.wolfe_curv*np.inner(delta_x0.T, del_f0.T):
                print("Wolfe condition ii not satisfied (step did not result in a sufficient decrease in objective function gradient).")
                x0 = x1
                break

            # Update Hessian
            N1 = get_N(N0, delta_x0, del_f0, del_f1)

            # Determine new search direction and perform line search
            s = -np.dot(N1, del_f1)
            mag_s = np.linalg.norm(s)
            s = s/mag_s
            x2, f2, alpha, wolfe_satis = line_search(
                x1, f1, s, del_f1, f, mag_dx, settings)
            if not wolfe_satis:  # Check first Wolfe condition. If not satisfied, reset BFGS update.
                x0 = x2
                print(
                    "Wolfe condition i not satisfied (step did not result in a sufficient decrease in the objective function).")
                break
            delta_x1 = x2-x1
            mag_dx = alpha

            # Update variables for next iteration
            x0 = x1
            f0 = f1
            del_f0 = del_f1
            delta_x0 = delta_x1
            x1 = x2
            f1 = f2

    return c.OptimizerResult(f2, x2, True, "Step tolerance reached.", iter, f.eval_calls.value)


def get_N(N0, delta_x0, del_f0, del_f1):
    """Perform BFGS update on N matrix"""
    gamma0 = del_f1-del_f0
    denom = np.matrix(delta_x0).T*np.matrix(gamma0)
    NG = np.matrix(N0)*np.matrix(gamma0)
    A = np.asscalar(1+np.matrix(gamma0).T*NG/denom)
    B = (np.matrix(delta_x0)*np.matrix(delta_x0).T/denom)
    C = (np.matrix(delta_x0)*np.matrix(gamma0).T *
         np.matrix(N0)+NG*np.matrix(delta_x0).T)/denom
    N1 = N0+A*B-C
    return N1


def line_search(x0, f0, s, del_f0, f, alpha, settings):
    """Perform line search to find a minimum in the objective function."""

    if settings.verbose:
        print('Line Search ----------------------------------------------------------------------------')
        print('Search Direction: {0}'.format(s))

    if settings.alpha_d is not None:
        alpha = settings.alpha_d

    prev_reduced = False
    prev_increased = False

    while True:
        if settings.verbose:
            print("Step size: {0}".format(alpha))

        # Get objective function values in the line search
        x_search = [x0+s*alpha*i for i in range(1, settings.n_search+1)]
        with mp.Pool(processes=settings.max_processes) as pool:
            f_search = pool.map(f.f, x_search)
        x_search = [x0]+x_search
        f_search = [f0]+f_search

        if settings.verbose:
            for i in range(settings.n_search + 1):
                out = '{0:5d}, {1:15.7E}'.format(i, f_search[i])
                for j in range(len(x0)):
                    out += ', {0:15.7E}'.format(np.asscalar(x_search[i][j]))
                print(out)

        # Check for invalid results
        if np.isnan(f_search).any():
            print('Found NaN in line search at the following design point:')
            print(x_search[np.where(np.isnan(f_search))[0]])
            raise ValueError("Objective function returned a NaN")

        # Check for plateau
        if min(f_search) == max(f_search):
            if settings.verbose:
                print('Objective function has plateaued')
            return x0, f0, alpha  # A plateaued objective will break find_opt_alpha()

        # Check for alpha getting too small
        if f_search[1] > f_search[0] and alpha < settings.termination_tol:
            if settings.verbose:
                print(
                    'Alpha within stopping tolerance: alpha = {0}'.format(alpha))
            return x0, f0, alpha, True

        # See if alpha needs to be adjusted
        min_ind = f_search.index(min(f_search))
        if min_ind == 0:
            if prev_increased:
                multiplier = settings.alpha_mult-1
                prev_increased = False
            else:
                multiplier = settings.alpha_mult
                prev_reduced = True
            if settings.verbose:
                print(
                    'Too big of a step. Reducing alpha by {0}'.format(multiplier))
            alpha /= multiplier
        elif min_ind == settings.n_search:
            if prev_reduced:
                multiplier = settings.alpha_mult-1
                prev_reduced = False
            else:
                multiplier = settings.alpha_mult
                prev_increased = True
            if settings.verbose:
                print(
                    'Too small of a step. Increasing alpha by {0}'.format(multiplier))
            alpha *= multiplier
        else:
            break

    # Find value of alpha at the optimum point in the search direction
    a = [alpha*i for i in range(settings.n_search+1)]
    alpha_opt = find_opt_alpha(a, f_search, min_ind, settings)
    if settings.verbose:
        print('Final alpha = {0}'.format(alpha_opt))
    x1 = x0+s*alpha_opt
    f1 = f.f(x1)

    # Check first Wolfe condition. Will break out of inner BFGS loop if not satisfied.
    armijo = f0+settings.wolfe_armijo*alpha_opt*np.inner(s.T, del_f0.T)
    if f1 > armijo:
        wolfe_satis = False
    else:
        wolfe_satis = True
    return x1, f1, alpha_opt, wolfe_satis


def find_opt_alpha(a, f_search, min_ind, settings):
    # Quadratic method
    if settings.search_type == 'quadratic':

        # Fit quadratic
        q = c.quadratic(np.asarray(a), np.asarray(f_search))
        (alpha_opt, f_opt) = q.vertex()

        # If the quadratic fit is good, return its vertex
        if not (alpha_opt is None or alpha_opt < 0 or not q.convex() or q.rsq < settings.rsq_tol):
            return alpha_opt

    # If bracketting method is selected, or if quadratic method fails, find the vertex defined by 3 minimum points
    a1 = a[min_ind - 1]
    a2 = a[min_ind]
    a3 = a[min_ind + 1]
    f1 = f_search[min_ind - 1]
    f2 = f_search[min_ind]
    f3 = f_search[min_ind + 1]

    alpha_opt = (f1*(a2**2-a3*2)+f2*(a3**2-a1**2)+f3*(a1**2-a2**2)
                 )/(2*(f1*(a2-a3)+f2*(a3-a1)+f3*(a1-a3)))
    if alpha_opt > a3 or alpha_opt < a1:
        alpha_opt = a2
    return alpha_opt


def sqp(f, g, x_start, settings):
    """Performs Sequntial Quadratic Programming on a constrained optimization function."""

    # Initialization
    iter = 0
    o_iter = 0
    n_vars = len(x_start)
    n_cstr = settings.n_cstr
    n_ineq_cstr = settings.n_ineq_cstr

    x0 = np.copy(x_start)
    mag_dx = 1  # Ensures the loop executes at least once

    # Start outer iteration
    while iter < settings.max_iterations and mag_dx > settings.termination_tol:
        if settings.verbose:
            print("Setting Lagrangian Hessian to the identity matrix.")
        o_iter += 1
        i_iter = 1
        iter += 1

        # Create quadratic approximation
        f0_eval = f.pool.apply_async(f.f, (x0,))
        g0 = eval_constr(g, x0)
        del_f0, del_g0 = eval_grad(x0, f, g, n_vars, n_cstr)
        del_2_L0 = np.eye(n_vars)*settings.hess_init
        f0 = f0_eval.get()
        append_file(iter, o_iter, i_iter, f0, 0.0, x0,
                    del_f0, settings, g=g0, del_g=del_g0)

        # Estimate initial penalty function. We allow this to be artificially high.
        P0 = np.copy(f0)
        for constr in g0:
            if constr < 0:
                P0 -= constr

        # Get step
        delta_x, l, x1, f1, g1, P1 = get_delta_x(
            x0, f0, f, g, P0, n_vars, n_cstr, n_ineq_cstr, del_2_L0, del_f0, del_g0, g0, settings)

        mag_dx = np.linalg.norm(delta_x)

        # Start inner iteration
        while mag_dx > settings.termination_tol:
            first = False
            iter += 1
            i_iter += 1

            # Create quadratic approximation
            del_f1, del_g1 = eval_grad(x1, f, g, n_vars, n_cstr)

            # Check gradient termination
            if np.linalg.norm(del_f1) < settings.grad_tol:
                cstr_calls = []
                for i in range(n_cstr):
                    cstr_calls.append(g[i].eval_calls.value)
                return OptimizerResult(f1, x1, True, "Gradient termination tolerance reached.", iter, f.eval_calls.value, cstr_calls)

            # Update the Lagrangian Hessain
            del_2_L1 = get_del_2_L(
                del_2_L0, del_f0, del_f1, l, del_g0, del_g1, n_vars, n_cstr, delta_x)

            append_file(iter, o_iter, i_iter, f1, mag_dx, x1,
                        del_f1, settings, g=g1, del_g=del_g1)

            # Get step
            delta_x, l, x2, f2, g2, P2 = get_delta_x(
                x1, f1, f, g, P1, n_vars, n_cstr, n_ineq_cstr, del_2_L1, del_f1, del_g1, g1, settings)

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
            if mag_dx < settings.termination_tol and P2 > f2:
                if settings.verbose:
                    print(
                        "Stuck at optimum outside of feasible space. Resetting BFGS update.")
                mag_dx = 1
                break  # Reset BFGS

            # End of inner loop
        # End of outer loop

    # Evaluate final case
    iter += 1
    i_iter += 1
    del_f1 = f.del_f(x1)
    del_g1 = np.zeros((n_vars, n_cstr))
    for i in range(n_cstr):
        del_g1[:, i] = g[i].del_g(x1).flatten()
    append_file(iter, o_iter, i_iter, f1, mag_dx, x1,
                del_f1, settings, g=g1, del_g=del_g1)
    cstr_calls = []
    for i in range(n_cstr):
        cstr_calls.append(g[i].eval_calls.value)
    return c.OptimizerResult(f1, x1, True, "Step termination tolerance reached.", iter, f.eval_calls.value, cstr_calls)


def eval_grad(x0, f, g, n_vars, n_cstr):
    # Evaluate gradients at specified point
    del_f0 = f.del_f(x0)
    del_g0 = np.zeros((n_vars, n_cstr))
    for i in range(n_cstr):
        del_g0[:, i] = g[i].del_g(x0).flatten()
    return del_f0, del_g0


def get_del_2_L(del_2_L0, del_f0, del_f1, l, del_g0, del_g1, n_vars, n_cstr, delta_x):
    # BFGS update for Lagrangian Hessian
    del_L0 = np.copy(del_f0)
    del_L1 = np.copy(del_f1)
    for i in range(n_cstr):
        del_L0 -= np.asscalar(l[i])*np.reshape(del_g0[:, i], (n_vars, 1))
        del_L1 -= np.asscalar(l[i])*np.reshape(del_g1[:, i], (n_vars, 1))
    gamma_0 = np.matrix(del_L1-del_L0)
    first = gamma_0*gamma_0.T/(gamma_0.T*np.matrix(delta_x))
    second = del_2_L0*(np.matrix(delta_x)*np.matrix(delta_x).T) * \
        del_2_L0/(np.matrix(delta_x).T*del_2_L0*np.matrix(delta_x))
    return np.asarray(del_2_L0+first-second)


def get_delta_x(x0, f0, f, g, P0, n_vars, n_cstr, n_ineq_cstr, del_2_L0, del_f0, del_g0, g0, settings):
    # Solve for delta_x and lambda given each possible combination of binding/non-binding constraints
    if settings.verbose:
        print("Penalty to beat: {0}".format(P0))

    # If a given combination has no negative Lagrangian multipliers corresponding to inequality constraints, the loop exits.
    # An equality constraint is always binding and its Lagrange multiplier my be any value.
    cstr_opts = [[True, False] for i in range(
        n_ineq_cstr)] + [[True] for i in range(n_ineq_cstr, n_cstr)]
    poss_combos = np.array(list(itertools.product(*cstr_opts)))
    for cstr_b in poss_combos:

        # At most, n constraints may be binding in n-dimensional space.
        if sum(cstr_b) > n_vars:
            continue

        if sum(cstr_b) > 1:
            # Check linear independence of constraint gradients.
            _, s, _ = np.linalg.svd(del_g0[:, cstr_b].T)
            if (abs(s) < 1e-14).any():
                continue

        delta_x, l = get_x_lambda(
            n_vars, n_cstr, del_2_L0, del_g0, del_f0, g0, cstr_b)

        x1 = x0+delta_x
        g1 = eval_constr(g, x1)
        # Do not allow non-binding constraints to be violated.
        if (g1[cstr_b == False] < 0).any():
            continue

        # Check if constraints assumed to be binding are actually non-binding.
        if not (l[:n_ineq_cstr].flatten() < 0).any():
            if settings.verbose:
                print("Optimal combination found.")
            break
    else:
        # If an optimal combination is not found, relax the conditions by allowing non-binding constraints to be violated.
        if settings.verbose:
            print(
                "Optimal combination not found. Allowing non-binding constraints to be violated.")
        for cstr_b in poss_combos:

            # At most, n constraints may be binding in n-dimensional space.
            if sum(cstr_b) > n_vars:
                continue

            if sum(cstr_b) > 1:
                # Check linear independence of constraint gradients.
                _, s, _ = np.linalg.svd(del_g0[:, cstr_b].T)
                if (abs(s) < 1e-14).any():
                    continue

            delta_x, l = get_x_lambda(
                n_vars, n_cstr, del_2_L0, del_g0, del_f0, g0, cstr_b)

            x1 = x0+delta_x
            g1 = eval_constr(g, x1)

            # Check if constraints assumed to be binding are actually non-binding.
            if not (l[:n_ineq_cstr].flatten() < 0).any():
                if settings.verbose:
                    print("Optimal combination found.")
                break

    if settings.verbose:
        print("Optimal combination of binding constraints: {0}".format(cstr_b))

    # Check penalty function at proposed optimum
    f1 = f.f(x1)
    P1 = np.copy(f1)
    for i in range(n_cstr):
        P1 += np.asscalar(abs(l[i])*abs(g1[i]))
    if settings.verbose:
        print("Point: {0}, Objective: {1}, Penalty: {2}".format(
            x1.flatten(), f1, P1))

    # Cut back step if the penalty function has increased
    while settings.strict_penalty and P1 > P0 and np.linalg.norm(delta_x) > settings.termination_tol:
        if settings.verbose:
            print("Stepped too far! Cutting step in half.")
        delta_x /= 2
        x1 = x0+delta_x
        f1 = f.f(x1)
        P1 = np.copy(f1)
        g1 = eval_constr(g, x1)
        for i in range(n_cstr):
            if i < n_ineq_cstr:
                if g1[i] > 0:  # We may have stepped back across a constraint, meaning it should no longer affect the penalty function
                    continue
                elif l[i] == 0 and g1[i] < 0:  # We may have started violating a new constraint
                    P1 += abs(g1[i])
                    continue
            P1 += np.asscalar(abs(g1[i]))
        if settings.verbose:
            print("Point: {0}, Objective: {1}, Penalty: {2}".format(
                x1.flatten(), f1, P1))

    return delta_x, l, x1, f1, g1, P1


def get_x_lambda(n_vars, n_cstr, del_2_L0, del_g0, del_f0, g0, cstr_b):
    n_bind = np.asscalar(sum(cstr_b))  # Number of binding constraints

    # Create linear system to solve for delta_x and lambda
    A = np.zeros((n_vars+n_bind, n_vars+n_bind))
    b = np.zeros((n_vars+n_bind, 1))
    A[:n_vars, :n_vars] = del_2_L0
    A[:n_vars, n_vars:] = -del_g0[:, cstr_b]
    A[n_vars:, :n_vars] = del_g0[:, cstr_b].T
    b[:n_vars] = -del_f0
    b[n_vars:] = np.reshape(-g0[cstr_b], (n_bind, 1))

    # Solve system and parse solution
    x_lambda = np.linalg.solve(A, b)
    delta_x = x_lambda[0:n_vars]
    l_sol = x_lambda[n_vars:]
    l = np.zeros((n_cstr, 1))
    l[cstr_b] = l_sol
    return delta_x, l


def grg(f, g, x_start, settings):
    """Performs Generalized Reduced Gradient optimization on a constrained optimization function."""

    # Initialization
    iter = 0
    n_vars = len(x_start)
    n_cstr = settings.n_cstr
    n_ineq_cstr = settings.n_ineq_cstr

    x0 = np.copy(x_start)
    mag_dx = 1  # Ensures the loop executes at least once
    f0 = f.f(x0)
    g0 = eval_constr(g, x0)

    while mag_dx > settings.termination_tol and iter < settings.max_iterations:
        iter += 1

        # Evaluate current point
        del_f0, del_g0 = eval_grad(x0, f, g, n_vars, n_cstr)

        append_file(iter, iter, iter, f0, mag_dx, x0,
                    del_f0, settings, g=g0, del_g=del_g0)

        # Determine binding constraints
        # Equality constraints are always binding.
        cstr_b = np.reshape([list(g0[:n_ineq_cstr].flatten(
        ) <= settings.cstr_tol)+[True for i in range(n_cstr-n_ineq_cstr)]], (n_cstr, 1))
        n_binding = np.asscalar(sum(cstr_b))

        # If there are more binding constraints than design variables, we must ignore some binding constraints to ensure linear independence.
        # Equality constraints will never be ignored.
        if n_binding > n_vars:
            if settings.verbose:
                print("Ignoring {0} binding constraints.".format(
                    n_binding-n_vars))
            unbound = 0
            for i in range(n_cstr):
                if cstr_b[i] and unbound < n_binding-n_vars:
                    cstr_b[i] = False
                    unbound += 1
        n_binding = np.asscalar(sum(cstr_b))

        if settings.verbose:
            print("{0} binding constraints".format(n_binding))

        d_psi_d_x0 = - \
            del_g0.T[np.repeat(cstr_b, n_vars, axis=1)
                     ].reshape((n_binding, n_vars))
        cstr_b = cstr_b.flatten()

        # Add slack variables
        s0 = g0[cstr_b].reshape((n_binding, 1))
        # We place the slack variables first since we would prefer those be the independent variables
        variables0 = np.concatenate((s0, x0), axis=0)

        # Partition variables
        z0, del_f_z0, d_psi_d_z0, z_ind0, y0, del_f_y0, d_psi_d_y0, y_ind0 = partition_vars(
            n_vars, n_binding, variables0, del_f0, d_psi_d_x0, settings)

        # Compute reduced gradient
        if n_binding != 0:
            del_f_r0 = (np.matrix(del_f_z0).T-np.matrix(del_f_y0).T *
                        np.linalg.inv(d_psi_d_y0)*np.matrix(d_psi_d_z0)).T
        else:
            del_f_r0 = np.matrix(del_f_z0)

        # Check gradient termination
        if np.linalg.norm(del_f_r0) < settings.grad_tol:
            cstr_calls = []
            for i in range(n_cstr):
                cstr_calls.append(g[i].eval_calls.value)
            return_message = "Gradient termination tolerance reached (magnitude = {0}).".format(
                np.linalg.norm(del_f_r0))
            return c.OptimizerResult(f0, x0, True, return_message, iter, f.eval_calls.value, cstr_calls)

        # The search direction is opposite the direction of the reduced gradient
        s = -1*del_f_r0/np.linalg.norm(del_f_r0)
        if settings.verbose:
            print("Search Direction: {0}".format(s.T))

        # Conduct line search
        x1, f1, g1 = grg_line_search(s, z0, z_ind0, y0, y_ind0, f, f0, g, g0, cstr_b,
                                     mag_dx, d_psi_d_z0, d_psi_d_y0, n_vars, n_cstr, n_binding, settings)

        delta_x = x1-x0
        mag_dx = np.linalg.norm(delta_x)
        x0 = x1
        f0 = f1
        g0 = g1

    del_f0, del_g0 = eval_grad(x0, f, g, n_vars, n_cstr)
    append_file(iter+1, iter+1, iter+1, f0, mag_dx, x0,
                del_f0, settings, g=g0, del_g=del_g0)
    cstr_calls = []
    for i in range(n_cstr):
        cstr_calls.append(g[i].eval_calls.value)
    return c.OptimizerResult(f1, x1, True, "Step termination tolerance reached (magnitude = {0}).".format(mag_dx), iter, f.eval_calls.value, cstr_calls)


def eval_constr(g, x1):
    n_cstr = len(g)
    g1 = np.zeros(n_cstr)
    for i in range(n_cstr):
        g1[i] = g[i].g(x1)
    return g1


def partition_vars(n_vars, n_binding, variables0, del_f0, d_psi_d_x0, settings):
    """Partitions independent and dependent variables."""
    # Search for independent variables and determine gradients
    z0 = np.zeros((n_vars, 1))
    del_f_z0 = np.zeros((n_vars, 1))
    d_psi_d_z0 = np.zeros((n_binding, n_vars))
    z_ind0 = []
    var_ind = -1
    for i in range(n_vars):
        while True:
            var_ind += 1
            # and (abs(variables0[var_ind])<1e-4 or variables0[var_ind]<0): # Slack variable at limit
            if var_ind < n_binding:
                z0[i] = variables0[var_ind]
                del_f_z0[i] = 0  # df/ds is always 0
                d_psi_d_z0[i, i] = 1  # dg/ds is always 1
                z_ind0.append(var_ind)
                break
            else:  # Design variable
                z0[i] = variables0[var_ind]
                del_f_z0[i] = del_f0[var_ind-n_binding]
                d_psi_d_z0[:, i] = d_psi_d_x0[:, var_ind-n_binding]
                z_ind0.append(var_ind)
                break

    # Search for dependent variables and determine gradients
    # Note the number of dependent variables is equal to the number of binding constraints
    y0 = np.zeros((n_binding, 1))
    del_f_y0 = np.zeros((n_binding, 1))
    d_psi_d_y0 = np.zeros((n_binding, n_binding))
    y_ind0 = []
    var_ind = -1
    for i in range(n_binding):
        while True:
            var_ind += 1
            if var_ind not in z_ind0:  # The variable is not independent
                y0[i] = variables0[var_ind]
                del_f_y0[i] = del_f0[var_ind-n_binding]
                d_psi_d_y0[:, i] = d_psi_d_x0[:, var_ind-n_binding]
                y_ind0.append(var_ind)
                break

    # Check that this matrix is not singular
    _, s, _ = np.linalg.svd(d_psi_d_y0)
    swap_var = 0
    while (abs(s) < 1e-14).any():
        if settings.verbose:
            print("Swapping independent and dependent variables.")
        tempind = copy.copy(z_ind0[n_binding+swap_var])
        z_ind0[n_binding+swap_var] = y_ind0[swap_var]
        y_ind0[swap_var] = tempind

        tempz = np.copy(z0[n_binding+swap_var])
        z0[n_binding+swap_var] = y0[swap_var]
        y0[swap_var] = tempz

        tempgrad = np.copy(del_f_z0[n_binding+swap_var])
        del_f_z0[n_binding+swap_var] = del_f_y0[swap_var]
        del_f_y0[swap_var] = tempgrad

        temppsi = np.copy(d_psi_d_z0[:, n_binding+swap_var])
        d_psi_d_z0[:, n_binding+swap_var] = d_psi_d_y0[:, swap_var]
        d_psi_d_y0[:, swap_var] = temppsi

        # Check that this matrix is not singular
        _, s, _ = np.linalg.svd(d_psi_d_y0)
        swap_var += 1

    return z0, del_f_z0, d_psi_d_z0, z_ind0, y0, del_f_y0, d_psi_d_y0, y_ind0


def grg_line_search(s, z0, z_ind0, y0, y_ind0, f, f0, g, g0, cstr_b, alpha, d_psi_d_z0, d_psi_d_y0, n_vars, n_cstr, n_binding, settings):
    """Performs line search in independent variables to find a minimum."""
    if settings.verbose:
        print("Line Search------------------------------")
        msg = ["{0:>20}".format("f")]
        for i in range(n_vars):
            msg.append(", {0:>20}".format("x"+str(i)))
        for i in range(n_cstr):
            msg.append(", {0:>20}".format("g"+str(i)))
        print("".join(msg))

    if settings.alpha_d is not None:
        alpha = settings.alpha_d

    while alpha > settings.termination_tol:
        if settings.verbose:
            print("Step size: {0}".format(alpha))

        # Initialize search arrays
        x_search = []
        f_search = []
        g_search = []

        # Add initial point
        if n_binding != 0:
            var_i = np.zeros(n_vars+n_binding)
            var_i[z_ind0] = z0.flatten()
            var_i[y_ind0] = y0.flatten()
        else:
            var_i = z0
        x_search.append(var_i[n_binding:].flatten())
        f_search.append(f0)
        g_search.append(g0.flatten())
        count = 1

        point_evals = []
        for i in range(1, settings.n_search+1):
            point_evals.append(f.pool.apply_async(eval_search_point, (f, g, z0, y0, alpha*i,
                                                                      s, d_psi_d_y0, d_psi_d_z0, z_ind0, y_ind0, n_vars, n_binding, cstr_b, settings)))
        for i in range(1, settings.n_search+1):
            point_vals = point_evals[i-1].get()
            if point_vals == None:
                continue
            x_search.append(point_vals[0])
            f_search.append(point_vals[1])
            g_search.append(point_vals[2])
            count += 1

        if count == 1:  # Line search returned no valid results
            alpha /= settings.alpha_mult
            if settings.verbose:
                print("Minimum not found. Decreasing step size.")
            continue

        x_search = np.asarray(x_search).T
        f_search = np.asarray(f_search)
        g_search = np.asarray(g_search).T

        if settings.verbose:
            for i in range(count):
                msg = ["{0:>20E}".format(f_search[i])]
                for x in x_search[:, i]:
                    msg.append(", {0:>20E}".format(x))
                for gi in g_search[:, i]:
                    msg.append(", {0:>20E}".format(gi))
                print("".join(msg))

        min_ind = np.argmin(f_search)
        # If the starting point is within feasible space, keep the algorithm from stepping outside of feasible space. If the starting point is outside, let the minimum point
        # exist as the first point outside of feasible space.
        while min_ind > 0 and ((g_search[:settings.n_ineq_cstr, min_ind] < -settings.cstr_tol).any() or (abs(g_search[settings.n_ineq_cstr:, min_ind]) > settings.cstr_tol).any()):
            min_ind -= 1  # Step back to feasible space
        if min_ind == 0 and ((g_search[:settings.n_ineq_cstr, min_ind] < -settings.cstr_tol).any() or (abs(g_search[settings.n_ineq_cstr:, min_ind]) > settings.cstr_tol).any()):
            min_ind += 1

        if min_ind == settings.n_search:  # Minimum at end of line search, step size must be increased
            alpha *= settings.alpha_mult
            if settings.verbose:
                print("Minimum not found. Increasing step size.")
            continue
        if min_ind == 0:  # Minimum at beginning of line search, step size must be reduced
            alpha /= settings.alpha_mult
            if settings.verbose:
                print("Minimum not found. Decreasing step size.")
            continue
        else:  # Minimum is found in the middle of the line search
            x1 = x_search[:, min_ind].reshape((n_vars, 1))
            f1 = f_search[min_ind]
            g1 = g_search[:, min_ind].reshape((settings.n_cstr, 1))
            break
    else:
        if len(x_search) == 1:
            raise RuntimeWarning(
                "Failed to converge to one or more constraint boundaries.")
            return x_search[0].reshape((n_vars, 1)), f_search[0], g_search[0].reshape((settings.n_cstr, 1))
        return x_search[:, 0].reshape((n_vars, 1)), f_search[0], g_search[:, 0].reshape((settings.n_cstr, 1))

    return x1, f1, g1


def eval_search_point(f, g, z0, y0, alpha, s, d_psi_d_y0, d_psi_d_z0, z_ind0, y_ind0, n_vars, n_binding, cstr_b, settings):

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            # Determine new point
            z_search = (z0+alpha*s)
            if n_binding != 0:
                y_search = np.asarray(
                    y0-np.linalg.inv(d_psi_d_y0)*np.matrix(d_psi_d_z0)*np.matrix(alpha*s))
                var_i = np.zeros(n_vars+n_binding)
                var_i[z_ind0] = z_search.flatten()
                var_i[y_ind0] = y_search.flatten()
            else:
                var_i = z_search
            x_search = var_i[n_binding:]

            # Evaluate constraints
            g_search = eval_constr(g, x_search)

            # Drive dependent variables back to the boundary of binding constraints which were violated (equality constraints are always binding).
            cstr_v = (cstr_b & ((g_search < settings.cstr_tol) | (
                np.array([i >= settings.n_ineq_cstr-1 for i in range(settings.n_cstr)]))))
            iterations = 0
            while n_binding != 0 and (abs(g_search[cstr_v]) > settings.cstr_tol).any() and iterations < 1000:
                iterations += 1  # To avoid divergence of the N-R method

                # Binding, non-violated constraints should just be left alone
                g_search[~cstr_v] = 0
                y_search = np.asarray(
                    y_search+np.linalg.inv(d_psi_d_y0)*np.matrix(g_search[cstr_b]).T)
                if n_binding != 0:
                    var_i = np.zeros(n_vars+n_binding)
                    var_i[z_ind0] = z_search.flatten()
                    var_i[y_ind0] = y_search.flatten()
                else:
                    var_i = z0
                x_search = np.asarray(var_i[n_binding:]).flatten()

                g_search = eval_constr(g, x_search)

            f_search = f.f(x_search)
            return np.asarray(x_search).flatten(), f_search, np.asarray(g_search).flatten()
        except Warning:
            return None


def get_constraints(constraints, pool, queue, settings):
    if constraints != None:
        n_cstr = len(constraints)
        n_ineq_cstr = 0
        g = []
        # Inequality constraints are stored first
        for constraint in constraints:
            if constraint["type"] == "ineq":
                n_ineq_cstr += 1
                grad = constraint.get("grad")
                constr = c.Constraint(
                    constraint["type"], constraint["fun"], pool, queue, settings, grad=grad)
                g.append(constr)
        for constraint in constraints:
            if constraint["type"] == "eq":
                grad = constraint.get("grad")
                constr = c.Constraint(
                    constraint["type"], constraint["fun"], pool, queue, settings, grad=grad)
                g.append(constr)
        g = np.array(g)
    else:
        g = None
        n_cstr = 0
        n_ineq_cstr = 0
    return g, n_cstr, n_ineq_cstr


def append_file(iter, o_iter, i_iter, obj_fcn_value, mag_dx, design_point, gradient, settings, **kwargs):
    g = kwargs.get("g")
    del_g = kwargs.get("del_g")

    msg = '{0:4d}, {1:5d}, {2:5d}, {3: 20.13E}, {4: 20.13E}'.format(
        iter, o_iter, i_iter, obj_fcn_value, mag_dx)
    values_msg = msg
    for value in design_point:
        values_msg = ('{0}, {1: 20.13E}'.format(
            values_msg, np.asscalar(value)))
    if not g is None:
        for cstr in g:
            values_msg = ('{0}, {1: 20.13E}'.format(
                values_msg, np.asscalar(cstr)))
    print(values_msg)
    with open(settings.opt_file, 'a') as opt_file:
        print(values_msg, file=opt_file)

    grad_msg = msg
    for grad in gradient:
        grad_msg = ('{0}, {1: 20.13E}'.format(grad_msg, np.asscalar(grad)))
    if not del_g is None:
        for i in range(settings.n_cstr):
            for j in range(len(design_point)):
                grad_msg = ('{0}, {1: 20.13E}'.format(
                    grad_msg, np.asscalar(del_g[j, i])))
    with open(settings.grad_file, 'a') as grad_file:
        print(grad_msg, file=grad_file)


def print_setup(n_vars, x_start, bounds, n_cstr, n_ineq_cstr, settings):
    print("\nOptix.py from USU AeroLab")

    print('\n---------- Variables ----------')
    print("Optimizing in {0} variables.".format(n_vars))
    print("Initial guess:\n{0}".format(x_start))
    if bounds != None:
        print("Variable bounds:\n{0}".format(bounds))

    print('\n---------- Constraints ----------')
    print('{0} total constraints'.format(n_cstr))
    print('{0} inequality constraints'.format(n_ineq_cstr))
    print('{0} equality constraints'.format(n_cstr-n_ineq_cstr))
    print('***Please note, Optix will rearrange constraints so the inequality constraints are listed first.')
    print('***Otherwise, the order is maintained.')

    print('\n---------- Settings ----------')
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


def eval_write(filename, header, q):
    with open(filename, 'w') as f:
        f.write(header+"\n")
        f.flush()
        while True:
            try:
                msg = q.get()
            except:
                continue
            if msg == 'kill':
                break
            f.write(msg+"\n")
            f.flush()
    return True


def format_output_files(n_vars, n_cstr, settings, pool, queue):
    opt_header = '{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}'.format(
        'iter', 'outer', 'inner', 'fitness', 'mag(dx)')
    for i in range(n_vars):
        opt_header += ', {0:>20}'.format('x'+str(i))
    for i in range(n_cstr):
        opt_header += ', {0:>20}'.format('g'+str(i))

    opt_filename = "optimize"+settings.file_tag+".txt"
    settings.opt_file = opt_filename
    with open(opt_filename, 'w') as opt_file:
        opt_file.write(opt_header + '\n')

    grad_header = '{0:>84}  {1:>20}'.format(' ', 'df')
    for i in range(n_cstr):
        grad_header += (', {0:>'+str(21*n_vars)+'}').format('dg'+str(i))
    grad_header += '\n{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}'.format(
        'iter', 'outer', 'inner', 'fitness', 'mag(dx)')
    for j in range(n_cstr+1):
        for i in range(n_vars):
            grad_header += ', {0:>20}'.format('dx'+str(i))

    grad_filename = "gradient"+settings.file_tag+".txt"
    settings.grad_file = grad_filename
    with open(grad_filename, 'w') as grad_file:
        grad_file.write(grad_header + '\n')

    # Print header to command line
    print(opt_header)
