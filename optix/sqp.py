import itertools

import numpy as np
import multiprocessing_on_dill as mp

from optix.classes import OptimizerResult
from optix.helpers import append_file, eval_grad, eval_constr


def sqp(f, g, x_start, settings):
    """Performs Sequntial Quadratic Programming on a constrained optimization function."""

    # Initialization
    iteration = 0
    o_iter = 0
    n_vars = len(x_start)
    n_cstr = settings.n_cstr
    n_ineq_cstr = settings.n_ineq_cstr

    x0 = np.copy(x_start)
    mag_dx = 1  # Ensures the loop executes at least once

    # Start outer iteration
    while iteration < settings.max_iterations and mag_dx > settings.termination_tol:
        if settings.verbose:
            print("Setting Lagrangian Hessian to the identity matrix.")
        o_iter += 1
        i_iter = 1
        iteration += 1

        # Create quadratic approximation
        f0_eval = f.pool.apply_async(f.f, (x0,))
        g0 = eval_constr(g, x0)
        del_f0, del_g0 = eval_grad(x0, f, g, n_vars, n_cstr)
        del_2_L0 = np.eye(n_vars)*settings.hess_init
        f0 = f0_eval.get()
        append_file(iteration, o_iter, i_iter, f0, 0.0, x0, settings, gradient=del_f0, g=g0, del_g=del_g0)

        # Estimate initial penalty function. We allow this to be artificially high.
        P0 = np.copy(f0)
        for constr in g0:
            if constr < 0:
                P0 -= constr

        # Get step
        delta_x, l, x1, f1, g1, P1 = _get_delta_x(
            x0, f0, f, g, P0, n_vars, n_cstr, n_ineq_cstr, del_2_L0, del_f0, del_g0, g0, settings)

        mag_dx = np.linalg.norm(delta_x)

        # Start inner iteration
        while mag_dx > settings.termination_tol:
            first = False
            iteration += 1
            i_iter += 1

            # Create quadratic approximation
            del_f1, del_g1 = eval_grad(x1, f, g, n_vars, n_cstr)

            # Check gradient termination
            if np.linalg.norm(del_f1) < settings.grad_tol:
                cstr_calls = []
                for i in range(n_cstr):
                    cstr_calls.append(g[i].eval_calls.value)
                return OptimizerResult(f1, x1, True, "Gradient termination tolerance reached.", iteration, f.eval_calls.value, cstr_calls)

            # Update the Lagrangian Hessain
            del_2_L1 = _get_del_2_L(del_2_L0, del_f0, del_f1, l, del_g0, del_g1, n_vars, n_cstr, delta_x)

            append_file(iteration, o_iter, i_iter, f1, mag_dx, x1, settings, gradient=del_f1, g=g1, del_g=del_g1)

            # Get step
            delta_x, l, x2, f2, g2, P2 = _get_delta_x(x1, f1, f, g, P1, n_vars, n_cstr, n_ineq_cstr, del_2_L1, del_f1, del_g1, g1, settings)

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
    iteration += 1
    i_iter += 1
    del_f1 = f.del_f(x1)
    del_g1 = np.zeros((n_vars, n_cstr))
    for i in range(n_cstr):
        del_g1[:, i] = g[i].del_g(x1).flatten()
    append_file(iteration, o_iter, i_iter, f1, mag_dx, x1, settings, gradient=del_f1, g=g1, del_g=del_g1)
    cstr_calls = []
    for i in range(n_cstr):
        cstr_calls.append(g[i].eval_calls.value)
    return OptimizerResult(f1, x1, True, "Step termination tolerance reached.", iteration, f.eval_calls.value, cstr_calls)


def _get_del_2_L(del_2_L0, del_f0, del_f1, l, del_g0, del_g1, n_vars, n_cstr, delta_x):
    # BFGS update for Lagrangian Hessian

    del_L0 = np.copy(del_f0)
    del_L1 = np.copy(del_f1)

    # Add in constraint graidents
    for i in range(n_cstr):
        del_L0 -= l[i]*del_g0[:, i]
        del_L1 -= l[i]*del_g1[:, i]

    # Intermediate calcuations
    gamma_0 = del_L1 - del_L0
    first = np.outer(gamma_0, gamma_0) / np.inner(gamma_0, delta_x)
    second = np.matmul(np.matmul(del_2_L0, np.outer(delta_x, delta_x)), del_2_L0) / np.inner(delta_x, np.matmul(del_2_L0, delta_x))

    # Calculate Hessian
    return del_2_L0 + first - second


def _get_delta_x(x0, f0, f, g, P0, n_vars, n_cstr, n_ineq_cstr, del_2_L0, del_f0, del_g0, g0, settings):
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

        delta_x, l = _get_x_lambda(
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
            print("Optimal combination not found. Allowing non-binding constraints to be violated.")

        for cstr_b in poss_combos:

            # At most, n constraints may be binding in n-dimensional space.
            if sum(cstr_b) > n_vars:
                continue

            # Check linear independence of constraint gradients.
            if sum(cstr_b) > 1:
                _, s, _ = np.linalg.svd(del_g0[:, cstr_b].T)
                if (abs(s) < 1e-14).any():
                    continue

            delta_x, l = _get_x_lambda(n_vars, n_cstr, del_2_L0, del_g0, del_f0, g0, cstr_b)

            x1 = x0 + delta_x
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
        P1 += abs(l[i])*abs(g1[i])
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
            P1 += abs(g1[i])
        if settings.verbose:
            print("Point: {0}, Objective: {1}, Penalty: {2}".format(
                x1.flatten(), f1, P1))

    return delta_x, l, x1, f1, g1, P1


def _get_x_lambda(n_vars, n_cstr, del_2_L0, del_g0, del_f0, g0, cstr_b):
    # Does something

    # Get number of binding constraints
    n_bind = np.count_nonzero(cstr_b)

    # Create linear system to solve for delta_x and lambda
    A = np.zeros((n_vars+n_bind, n_vars+n_bind))
    b = np.zeros(n_vars+n_bind)
    A[:n_vars, :n_vars] = del_2_L0
    A[:n_vars, n_vars:] = -del_g0[:, cstr_b]
    A[n_vars:, :n_vars] = del_g0[:, cstr_b].T
    b[:n_vars] = -del_f0
    b[n_vars:] = -g0[cstr_b]

    # Solve system and parse solution
    x_lambda = np.linalg.solve(A, b)
    delta_x = x_lambda[0:n_vars]
    l_sol = x_lambda[n_vars:]
    l = np.zeros(n_cstr)
    l[cstr_b] = l_sol
    return delta_x, l