import itertools
import warnings
import copy

import numpy as np
import multiprocessing_on_dill as mp

from optix.classes import Settings, OptimizerResult, Objective, Quadratic
from optix.helpers import append_file, print_setup, get_constraints, eval_write, format_output_files, eval_grad, eval_constr


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

        append_file(iter, iter, iter, f0, mag_dx, x0, del_f0, settings, g=g0, del_g=del_g0)

        # Determine binding constraints
        # Equality constraints are always binding.
        cstr_b = np.array(list(g0[:n_ineq_cstr] <= settings.cstr_tol) + [True for i in range(n_cstr-n_ineq_cstr)])
        n_binding = np.count_nonzero(cstr_b)

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
        n_binding = np.count_nonzero(cstr_b)

        if settings.verbose:
            print("{0} binding constraints".format(n_binding))

        # Assemble derivatives of binding constraints
        d_psi_d_x0 = -del_g0[:,cstr_b].T

        # Add slack variables
        s0 = g0[cstr_b]

        # We place the slack variables first since we would prefer those be the independent variables
        variables0 = np.concatenate((s0, x0), axis=0)

        # Partition variables
        z0, del_f_z0, d_psi_d_z0, z_ind0, y0, del_f_y0, d_psi_d_y0, y_ind0 = _partition_vars(n_vars, n_binding, variables0, del_f0, d_psi_d_x0, settings)

        # Compute reduced gradient
        if n_binding != 0:
            x = np.linalg.solve(d_psi_d_y0, del_f_y0)
            del_f_r0 = del_f_z0 - np.matmul(d_psi_d_z0.T, x)
        else:
            del_f_r0 = del_f_z0

        # Check gradient termination
        if np.linalg.norm(del_f_r0) < settings.grad_tol:
            cstr_calls = []
            for i in range(n_cstr):
                cstr_calls.append(g[i].eval_calls.value)
            return_message = "Gradient termination tolerance reached (magnitude = {0}).".format(np.linalg.norm(del_f_r0))
            return OptimizerResult(f0, x0, True, return_message, iter, f.eval_calls.value, cstr_calls)

        # The search direction is opposite the direction of the reduced gradient
        s = -del_f_r0 / np.linalg.norm(del_f_r0)
        if settings.verbose:
            print("Search Direction: {0}".format(s.T))

        # Conduct line search
        x1, f1, g1, err = _grg_line_search(s, z0, z_ind0, y0, y_ind0, f, f0, g, g0, cstr_b, mag_dx, d_psi_d_z0, d_psi_d_y0, n_vars, n_cstr, n_binding, settings)
        if err == -1:
            cstr_calls = []
            for i in range(n_cstr):
                cstr_calls.append(g[i].eval_calls.value)
            return_message = "Failed to converge to one or more constraint boundaries."
            return OptimizerResult(f1, x1, False, return_message, iter, f.eval_calls.value, cstr_calls)

        delta_x = x1-x0
        mag_dx = np.linalg.norm(delta_x)
        x0 = x1
        f0 = f1
        g0 = g1

    del_f0, del_g0 = eval_grad(x0, f, g, n_vars, n_cstr)
    append_file(iter+1, iter+1, iter+1, f0, mag_dx, x0, del_f0, settings, g=g0, del_g=del_g0)
    cstr_calls = []
    for i in range(n_cstr):
        cstr_calls.append(g[i].eval_calls.value)
    return_message = "Step termination tolerance reached (magnitude = {0}).".format(mag_dx)
    return OptimizerResult(f1, x1, True, return_message, iter, f.eval_calls.value, cstr_calls)


def _partition_vars(n_vars, n_binding, variables0, del_f0, d_psi_d_x0, settings):
    """Partitions independent and dependent variables."""

    # Initialize some things
    z0 = np.zeros(n_vars)
    del_f_z0 = np.zeros(n_vars)
    d_psi_d_z0 = np.zeros((n_binding, n_vars))
    z_ind0 = []
    var_ind = -1

    # Search for independent variables and determine gradients
    for i in range(n_vars):
        while True:

            var_ind += 1
            # and (abs(variables0[var_ind])<1e-4 or variables0[var_ind]<0): # Slack variable at limit
            if var_ind < n_binding:
                z0[i] = variables0[var_ind]
                del_f_z0[i] = 0.0  # df/ds is always 0
                d_psi_d_z0[i, i] = 1.0  # dg/ds is always 1
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
    y0 = np.zeros(n_binding)
    del_f_y0 = np.zeros(n_binding)
    d_psi_d_y0 = np.zeros((n_binding, n_binding))
    y_ind0 = []
    var_ind = -1
    for i in range(n_binding):
        while True:

            var_ind += 1

            # Check if this variable is not independent
            if var_ind not in z_ind0:
                y0[i] = variables0[var_ind]
                del_f_y0[i] = del_f0[var_ind-n_binding]
                d_psi_d_y0[:, i] = d_psi_d_x0[:, var_ind-n_binding]
                y_ind0.append(var_ind)
                break

    # Check that this matrix is not singular
    _, s, _ = np.linalg.svd(d_psi_d_y0)
    swap_var = 0

    # Swap things around until the matrix is not singular
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

        # Check that the matrix is not singular
        _, s, _ = np.linalg.svd(d_psi_d_y0)
        swap_var += 1

    return z0, del_f_z0, d_psi_d_z0, z_ind0, y0, del_f_y0, d_psi_d_y0, y_ind0


def _grg_line_search(s, z0, z_ind0, y0, y_ind0, f, f0, g, g0, cstr_b, alpha, d_psi_d_z0, d_psi_d_y0, n_vars, n_cstr, n_binding, settings):
    """Performs line search in independent variables to find a minimum."""

    # Print line search header
    if settings.verbose:
        print("Line Search------------------------------")
        msg = ["{0:>20}".format("f")]
        for i in range(n_vars):
            msg.append(", {0:>20}".format("x"+str(i)))
        for i in range(n_cstr):
            msg.append(", {0:>20}".format("g"+str(i)))
        print("".join(msg))

    # Reset alpha if needed
    if settings.alpha_reset:
        alpha = settings.alpha_init

    # Alpha loop
    while alpha > settings.termination_tol:

        # Print step size
        if settings.verbose:
            print("Step size: {0}".format(alpha))

        # Initialize search arrays
        x_search = []
        f_search = []
        g_search = []

        # Set up initial point
        if n_binding != 0:
            var_i = np.zeros(n_vars+n_binding)
            var_i[z_ind0] = z0
            var_i[y_ind0] = y0
        else:
            var_i = z0

        # Store in lists
        x_search.append(var_i[n_binding:])
        f_search.append(f0)
        g_search.append(g0)
        count = 1

        # Kick off multiprocessing evaluations
        point_evals = []
        for i in range(1, settings.n_search+1):
            point_evals.append(f.pool.apply_async(_eval_search_point, (f, g, z0, y0, alpha*i, s, d_psi_d_y0, d_psi_d_z0, z_ind0, y_ind0, n_vars, n_binding, cstr_b, settings)))

        # Get results for each point
        for i in range(1, settings.n_search+1):

            # Get result
            point_vals = point_evals[i-1].get()
            if point_vals == None:
                continue

            # Store
            x_search.append(point_vals[0])
            f_search.append(point_vals[1])
            g_search.append(point_vals[2])
            count += 1

        # Convert to arrays
        x_search = np.array(x_search).T
        f_search = np.array(f_search)
        g_search = np.array(g_search).T

        # Line search returned no valid results
        if count == 1:
            alpha /= settings.alpha_mult
            if settings.verbose:
                print("Minimum not found. Decreasing step size.")
            continue

        # Print out line search results
        if settings.verbose:
            for i in range(count):
                msg = ["{0:>20E}".format(f_search[i])]
                for x in x_search[:, i]:
                    msg.append(", {0:>20E}".format(x))
                for gi in g_search[:, i]:
                    msg.append(", {0:>20E}".format(gi))
                print("".join(msg))

        # Find minimum index
        min_ind = np.argmin(f_search)

        # If the starting point is within feasible space, keep the algorithm from stepping outside of feasible space. If the starting point is outside, let the minimum point
        # exist as the first point outside of feasible space.
        while min_ind > 0 and ((g_search[:settings.n_ineq_cstr, min_ind] < -settings.cstr_tol).any() or (abs(g_search[settings.n_ineq_cstr:, min_ind]) > settings.cstr_tol).any()):
            min_ind -= 1  # Step back to feasible space
        if min_ind == 0 and ((g_search[:settings.n_ineq_cstr, min_ind] < -settings.cstr_tol).any() or (abs(g_search[settings.n_ineq_cstr:, min_ind]) > settings.cstr_tol).any()):
            min_ind += 1

        # Minimum at end of line search, step size must be increased
        if min_ind == settings.n_search:
            alpha *= settings.alpha_mult
            if settings.verbose:
                print("Minimum not found. Increasing step size.")
            continue

        # Minimum at beginning of line search, step size must be reduced
        if min_ind == 0:
            alpha /= settings.alpha_mult
            if settings.verbose:
                print("Minimum not found. Decreasing step size.")
            continue

        # Minimum is found in the middle of the line search, which is what we want, so we can step out of the step size loop
        else:
            x1 = x_search[:,min_ind]
            f1 = f_search[min_ind]
            g1 = g_search[:,min_ind]
            code = 1
            break

    else:

        # We've reached the minimum step size, so just return the starting point
        x1 = x_search[:,0]
        f1 = f_search[0]
        g1 = g_search[:,0]

        # We had problems...
        if len(x_search) == 1:
            code = -1

        else:
            code = 1

    return x1, f1, g1, code


def _eval_search_point(f, g, z0, y0, alpha, s, d_psi_d_y0, d_psi_d_z0, z_ind0, y_ind0, n_vars, n_binding, cstr_b, settings):

    with warnings.catch_warnings():

        warnings.filterwarnings('error', category=RuntimeWarning)

        try:

            # Determine new point
            z_search = z0 + alpha*s

            if n_binding != 0:
                y_step = np.linalg.solve(d_psi_d_y0, np.matmul(d_psi_d_z0, alpha*s))
                y_search = y0 - y_step
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

                # Calculate search point
                y_search = y_search + np.linalg.solve(d_psi_d_y0, g_search[cstr_b])

                if n_binding != 0:
                    var_i = np.zeros(n_vars+n_binding)
                    var_i[z_ind0] = z_search
                    var_i[y_ind0] = y_search
                else:
                    var_i = z0

                x_search = var_i[n_binding:]

                g_search = eval_constr(g, x_search)

            f_search = f.f(x_search)

            return x_search, f_search, g_search

        except Warning:
            return None