import itertools

import numpy as np
import multiprocessing_on_dill as mp

from optix.classes import Settings, Objective, Quadratic, OptimizerResult
from optix.helpers import print_setup, get_constraints, eval_write, format_output_files, append_file


def bfgs(f, x_start, settings):
    """Performs quasi-Newton, unconstrained optimization using the BFGS Hessian update."""

    # Initialize
    iteration = -1
    n = len(x_start)
    o_iter = -1
    mag_dx = 1
    x0 = np.copy(x_start)
    alpha_guess = None

    # Outer loop. Sets the N matrix to [I].
    while iteration < settings.max_iterations and mag_dx > settings.termination_tol:

        # Print Hessian message
        if settings.verbose:
            print("Setting Hessian to the identity matrix.")

        # Initialize iterations
        o_iter += 1
        i_iter = 0
        iteration += 1

        # Get starting point
        f0_eval = f.pool.apply_async(f.f, (x0,))
        del_f0 = f.del_f(x0)
        f0 = f0_eval.get()
        append_file(iteration, o_iter, i_iter, f0, 0.0, x0, del_f0, settings)
        N0 = np.eye(n)*settings.hess_init

        # Determine search direction and perform line search
        s = -np.matmul(N0, del_f0)
        if alpha_guess is None or settings.alpha_reset:
            alpha_guess = settings.alpha_init
        else:
            alpha_guess = mag_dx
        mag_s = np.linalg.norm(s)
        s = s/mag_s
        x1, f1, alpha, wolfe_satis = _line_search(x0, f0, s, del_f0, f, alpha_guess, settings)
        delta_x0 = x1-x0
        mag_dx = alpha

        # Inner loop. Uses BFGS update for N.
        while iteration < settings.max_iterations and mag_dx > settings.termination_tol:
            i_iter += 1
            iteration += 1

            # Update gradient and output file
            del_f1 = f.del_f(x1)
            append_file(iteration, o_iter, i_iter, f1, mag_dx, x1, del_f1, settings)

            # Check for gradient termination
            if np.linalg.norm(del_f1) < settings.grad_tol:
                return OptimizerResult(f1, x1, True, "Gradient tolerance reached.", iteration, f.eval_calls.value)

            # Check second Wolfe condition. If not satisfied, reset BFGS update.
            if np.inner(delta_x0.T, del_f1.T) < settings.wolfe_curv*np.inner(delta_x0.T, del_f0.T):
                print("Wolfe condition ii not satisfied (step did not result in a sufficient decrease in objective function gradient).")
                x0 = x1
                break

            # Update Hessian inverse
            N1 = _get_N(N0, delta_x0, del_f0, del_f1)

            # Determine new search direction and perform line search
            s = -np.matmul(N1, del_f1)
            mag_s = np.linalg.norm(s)
            s = s/mag_s
            if settings.alpha_reset:
                alpha_guess = settings.alpha_init
            else:
                alpha_guess = mag_dx
            x2, f2, alpha, wolfe_satis = _line_search(x1, f1, s, del_f1, f, alpha_guess, settings)
            if not wolfe_satis:  # Check first Wolfe condition. If not satisfied, reset BFGS update.
                x0 = x2
                print("Wolfe condition i not satisfied (step did not result in a sufficient decrease in the objective function).")
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

    return OptimizerResult(f2, x2, True, "Step tolerance reached.", iteration, f.eval_calls.value)


def _get_N(N0, delta_x0, del_f0, del_f1):
    """Perform BFGS update on inverse Hessian matrix"""

    # Initial calcs
    y_k = del_f1 - del_f0
    sigma_k = 1.0/np.inner(delta_x0, y_k)

    ## Intermediate matrics
    #NG = np.matmul(N0, y_k)
    #A = 1.0 + np.matrix(y_k).T*NG*sigma_k
    #B = np.outer(delta_x0, delta_x0)*sigma_k
    #C = ( np.matmul(np.outer(delta_x0, y_k), N0) + np.matmul(NG, delta_x0) ) * sigma_k

    ## Calculate new Hessian
    #N1 = N0 + A*B - C
    A = np.eye(len(delta_x0)) - sigma_k*np.outer(delta_x0, y_k)
    B = sigma_k*np.outer(delta_x0, delta_x0)

    return np.matmul(A, np.matmul(N0, A)) + B


def _line_search(x0, f0, s, del_f0, f, alpha, settings):
    """Perform line search to find a minimum in the objective function."""

    if settings.verbose:
        print('Line Search ----------------------------------------------------------------------------')
        print('Search Direction: {0}'.format(s))

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
                    out += ', {0:15.7E}'.format(x_search[i][j])
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
    alpha_opt = _find_opt_alpha(a, f_search, min_ind, settings)
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


def _find_opt_alpha(a, f_search, min_ind, settings):
    # Quadratic method
    if settings.search_type == 'quadratic':

        # Fit quadratic
        q = Quadratic(np.asarray(a), np.asarray(f_search))
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