import numpy as np
import multiprocessing_on_dill as mp

from optix.classes import OptimizerResult


def nelder_mead(f, x_start, settings):
    """Optimizes the objective function using the Nelder-Mead algorithm."""

    # Initialize
    N = len(x_start)
    dists = np.zeros(N)

    # Set up initial simplex
    x_simp = np.zeros((N+1,N))
    x_simp[0,:] = x_start
    for i in range(1, N+1):
        x_simp[i,:] = x_start
        x_simp[i,i-1] += 1.0

    # Get function values for initial simplex
    with mp.Pool(processes=settings.max_processes) as pool:
        f_simp = np.array(pool.map(f.f, list(x_simp)))

    # Loop
    iteration = 0
    while True:

        # Increment iteration
        iteration += 1
        print()
        print(x_simp)
        print(f_simp)

        # Sort simplex vertices based on objective function value
        i_sorted = np.argsort(f_simp)

        # Check distances from minimum
        for i in range(1,N+1):
            dists[i-1] = np.sum((x_simp[i_sorted[i]] - x_simp[i_sorted[0]])**2)
        if np.max(dists) <= settings.termination_tol:
            break

        # Calculate centroid
        x_cent = np.average(x_simp[i_sorted[:N]], axis=0)

        # Reflect
        x_r = x_cent + settings.alpha*(x_cent - x_simp[i_sorted[N],:])
        f_r = f.f(x_r)

        # If the new point is the best, then expand the simplex
        if f_r < f_simp[i_sorted[0]]:

            # Calculate expanded point
            x_e = x_cent + settings.gamma*(x_r - x_cent)
            f_e = f.f(x_e)

            # Choose the best between the reflected and expanded points
            if f_e < f_r:
                x_simp[i_sorted[N],:] = x_e
                f_simp[i_sorted[N]] = f_e
            else:
                x_simp[i_sorted[N],:] = x_r
                f_simp[i_sorted[N]] = f_r

            continue

        # If the reflected point is not the best but good, then keep it
        elif f_r < f_simp[i_sorted[N-1]]:
            x_simp[i_sorted[N],:] = x_r
            f_simp[i_sorted[N]] = f_r
            continue

        # Contraction
        else:

            # Check if we're better than the worst
            if f_r < f_simp[i_sorted[N]]:

                # Calculate contracted point
                x_c = x_cent + settings.rho*(x_r - x_cent)
                f_c = f.f(x_c)

                # See if this is better than the reflected point
                if f_c < f_r:
                    x_simp[i_sorted[N],:] = x_c
                    f_simp[i_sorted[N]] = f_c
                    continue

            # We're worse than the worst
            else:

                # Calculate contracted point
                x_c = x_cent + settings.rho*(x_simp[i_sorted[N],:] - x_cent)
                f_c = f.f(x_c)

                # See if this is better than the worst point
                if f_c < f_simp[i_sorted[N]]:
                    x_simp[i_sorted[N],:] = x_c
                    f_simp[i_sorted[N]] = f_c
                    continue

        # If we've made it here, then the simplex needs to shrink
        for i in range(1,N):

            x_simp[i_sorted[i],:] = x_simp[i_sorted[0],:] + settings.sigma*(x_simp[i_sorted[i],:] - x_simp[i_sorted[0],:])
        
        # Calculate new objective function values
        with mp.Pool(processes=settings.max_processes) as pool:
            f_new = pool.map(f.f, list(x_simp[i_sorted[1:]]))
            f_simp[i_sorted[1:]] = f_new


    return OptimizerResult(f_simp[i_sorted[0]], x_simp[i_sorted[0],:], True, "", iteration, f.eval_calls.value)