import numpy as np
import multiprocessing_on_dill as mp

from optix.classes import Settings, Objective
from optix.helpers import print_setup, get_constraints, eval_write, format_output_files
from optix.grg import grg
from optix.sqp import sqp
from optix.bfgs import bfgs
from optix.nelder_mead import nelder_mead

np.set_printoptions(precision=14)
np.seterr(all='warn')

zero = 1.0e-20

def minimize(fun, x0, **kwargs):
    """Minimize a scalar function in one or more variables

        Parameters
        ----------
        fun : callable
            Objective to be minimized. Must be a scalar function of the form
            ```
                def fun(x,*args):
                    return float
            ```
            where x is a vector of the design variables and *args is all other parameters necessary for calling the function.

        x0 : array-like
            A starting guess for the independent variables. May be a list or numpy array.

        args : tuple, optional
            Arguments to be passed to the objective function.

        method : str, optional
            Method to be used by minimize to find the minimum of the objective function. May be one of the following:

                Unconstrained problem:
                    "bfgs" - quasi-Newton with bfgs Hessian update
                    "nelder-mead" - Nelder-Mead simplex optimization (gradient-free)

                Constrained problem:
                    "sqp" - sequential quadratic programming
                    "grg" - generalized reduced gradient

            If no method is specified, either "bfgs" or "sqp" will be chosen, based on whether constraints were given.

        grad : callable, optional
            Returns the gradient of the objective function at a specified point. Definition is the same as fun() but must return array-like,
            shape(n). If not specified, will be estimated using a finite-difference approximation.

        constraints : list of dict, optional
            Constraints on the design space. Can only be used with constrained
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

                grad (callable, optional
                    Returns the gradient of the constraint function at a
                    specified point. Must return array-like, shape(n,). May
                    only have one argument, being an array of the design variables.

        termination_tol : float, optional
            Execution terminates if the change in x for any step becomes less than the termination tolerance. Defaults to 1e-12.

        grad_tol : float, optional
            Execution terminates if the norm of the gradient at any step becomes less than this tolerance. Defaults to 1e-12.

        verbose : bool, optional
            If set to true, extra information about each step of the optimization will be printed to the command line. Defaults to False.

        cent_diff : bool, optional
            Flag for setting finite-difference approximation method. If set to false, a forward-difference approximation will be used. Otherwise,
            defaults to a central-difference.

        file_tag : str, optional
            Tag to be appended to the output filenames. If not specified, output files will be overwritten each time minimize() is called.
            Output files may still be overwritten if file_tag does not change with each call.

        max_processes : int, optional
            Maximum number of processes to be used in multiprocessing. Defaults fo 1.

        dx : float, optional
            Step size to be used in finite difference methods. Defaults to 0.001

        max_iterations : int, optional
            Maximum number of iterations for the optimization algorithm. Defaults to inf.

        num_avg : int, optional
            Number of times to run the objective function at each point. The objective value returned will be the average of all calls. This can be useful when
            dealing with noisy models. Defaults to 1.

        Returns
        ------

        Optimum : OptimizerResult
            Object containing information about the result of the optimization.

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

        n_search : int, optional
            Number of points to be considered in the search direction. Defaults to 8.

        alpha_init : float, optional
            Step size to be used for the first iteration of the first line search. Defaults to 1/n_search.

        alpha_reset : bool, optional
            If this is set to True, the value of alpha will be reset to the initial value at the beginning of each line search. If set to False, the value of alpha will
            be set to the optimum step size from the previous line search. Defaults to False.

        alpha_mult : float, optional
            Factor by which alpha is adjusted during each iteration of the line search. Defaults to n_search-1.
            Optix will occasionally adjust this value to keep the line search from oscillating between two values of alpha.

        line_search : str, optional
            Specifies which type of line search should be conducted in the search direction. The following types are possible:

                "bracket" - backets minimum and finds the vertex of the parabola formed by 3 minimum points
                "quadratic" - fits a quadratic to the search points and finds the vertex

            Defaults to bracket.

        rsq_tol : float, optional:
            Specifies the necessary quality of the quadratic fit to the line search (only used if line_search is "quadratic"). The quadratic fit will only be
            accepted if the R^2 value of the fit is above rsq_tol. Otherwise, the method will fall back to bracketing. Defaults to 0.8.

        wolfe_armijo : float, optional
            Value of c1 in the Wolfe conditions. Defaults to 1e-4.

        wolfe_curv : float, optional
            Value of c2 in the Wolfe conditions. Defaults to 0.9.

        hess_init : float, optional
            Sets the value of the Hessian to hess_init*[I] for the first iteration of the BFGS update. Increasing this value may help speed convergence of some
            problems. Defaults to 1.

        SQP

        strict_penalty : bool, optional
            Specifies whether a given step in the optimization must result in a decrease in the penatly function. Setting this to false may help convergence of some problems
            and speed computation. Defaults to true.

        hess_init : float, optional
            Sets the value of the Hessian to hess_init*[I] for the first iteration of the BFGS update. Increasing this value may help speed convergence of some
            problems, but this is not recommended in most cases. Behavior is not stable if this value is less than 1. Defaults to 1.

        GRG

        n_search : int, optional
            Number of points to be considered in the search direction. Defaults to 8.

        alpha_d : float, optional
            Step size to be used in line searches. If not specified, the step size is the optimum step size from the previous iteration.

        alpha_mult : float, optional
            Factor by which alpha is adjusted during each iteration of the line search. Defaults to n_search - 1

        cstr_tol : float, optional
            A constraint is considered to be binding if it evaluates to less than this number. Defaults to 1e-4.

        Nelder-Mead

        reflection_coef : float, optional
            Reflection parameter. Defaults to 1.0.

        expansion_coef : float, optional
            Expansion parameter. Defaults to 2.0.

        contraction_coef : float, optional
            Contraction parameter. Defaults to 0.5.

        shrink_coef : float, optional
            Shrink parameter. Defaults to 0.5.

    """

    # Initialize settings
    settings = Settings(**kwargs)

    # Initialize design variables
    n_vars = len(x0)
    x_start = np.array(x0)

    # Initialize multiprocessing
    with mp.Pool(settings.max_processes) as pool:
        manager = mp.Manager()
        queue = manager.Queue()

        # Initialize objective function
        grad = kwargs.get("grad")
        hess = kwargs.get("hess")
        f = Objective(fun, pool, queue, settings, grad=grad, hess=hess)

        # Initialize constraints
        constraints = kwargs.get('constraints', None)

        g, n_cstr, n_ineq_cstr = get_constraints(
            kwargs.get("constraints"), pool, queue, settings)
        settings.n_cstr = n_cstr
        settings.n_ineq_cstr = n_ineq_cstr

        # Check constraints
        if n_cstr-n_ineq_cstr > n_vars:
            raise IOError("There are too many equality constraints; the problem is overconstrained.")

        # Print setup information to command line
        print_setup(n_vars, x_start, n_cstr, n_ineq_cstr, settings)

        # Initialize formatting of output files
        format_output_files(n_vars, n_cstr, settings, pool, queue)

        # Kick off evaluation storage process (for more than one process)
        if settings.max_processes > 1:
            eval_header = '{0:>20}'.format('f')
            for i in range(n_vars):
                eval_header += ', {0:>20}'.format('x'+str(i))
            eval_filename = "evaluations"+settings.file_tag+".txt"

            writer = pool.apply_async(eval_write, (eval_filename, eval_header, queue))

        # Drive to the minimum
        opt = _find_minimum(f, g, x_start, settings)

        # Kick off evaluation storage process (for only one process)
        if settings.max_processes == 1:
            eval_header = '{0:>20}'.format('f')
            for i in range(n_vars):
                eval_header += ', {0:>20}'.format('x'+str(i))
            eval_filename = "evaluations"+settings.file_tag+".txt"

            writer = pool.apply_async(eval_write, (eval_filename, eval_header, queue))

        # Kill evaluation printer process
        queue.put('kill')
        writer_success = writer.get()
        if not writer_success:
            print("Evaluation writer did not terminate successfully.")
        pool.close()
        pool.join()

    return opt


def _find_minimum(f, g, x_start, settings):
    """Calls specific optimization algorithm as needed"""

    # BFGS
    if settings.method == "bfgs":
        return bfgs(f, x_start, settings)

    # SQP
    elif settings.method == "sqp":
        return sqp(f, g, x_start, settings)

    # GRG
    elif settings.method == "grg":
        return grg(f, g, x_start, settings)

    # Nelder-Mead
    elif settings.method == "nelder-mead":
        return nelder_mead(f, x_start, settings)

    else:
        raise ValueError("Method improperly specified.")