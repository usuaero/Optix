--------------------------------------------------------------------
\                  #######                       
 \         X       #     # #####  ##### # #    # 
  \       / \      #     # #    #   #   #  #  #  
   \     /   X     #     # #    #   #   #   ##   
    X---X    | X-X #     # #####    #   #   ##   
             |/    #     # #        #   #  #  #  
             X     ####### #        #   # #    # 
--------------------------------------------------------------------
AeroLab Optimization Software

Created by:
Dr Doug Hunsaker (professor, Utah State University; director, USU AeroLab)
Josh Hodson (graduate student, Utah State University)
Cory Goates (undergraduate student, Utah State University)

README (last revision: 2/9/2019)

NOTE FOR LEGACY OPTIX USERS:
Since January of 2019, API of Optix has changed significantly to somewhat
mirror that of scipy.optimize. We continue working to improve the API and
functionality of Optix. This README will always contain update information
for using Optix.

AS OF 2/22/2019 GRG FUNCTIONALITY IN OPTIX ARE NOT AVAILABLE.

AS OF 2/22/2019, BOUNDS FUNCTIONALITY IN OPTIX IS NOT AVAILABLE.

--------------------------------------------------------------------
INTRODUCTION
--------------------------------------------------------------------

Optix is a gradient-based optimization tool designed with aerodynamics
in mind. Recognizing that aerodynamic functions are often computationally
intensive, Optix has been designed to be as light-weight and parallel
as possible, while offering an intuitive API.

--------------------------------------------------------------------
API
--------------------------------------------------------------------

All functionality of Optix is wrapped within the minimize() function in
optix.py.

OpimizerResult object = minimize(fun,x0,**kwargs)

Inputs
------

    fun(callable)
    - Objective to be minimized. Must be a scalar function:
    def fun(x,*args):
        return float
    where x is a vector of the design variables and *args is all
    other parameters necessary for calling the function. Note that
    Optix will handle x as a column vector (i.e. shape(n_vars,1)).
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
    to 1.

    dx(float,optional)
    - Step size to be used in finite difference methods. Defaults to 0.001

    alpha_d(float,optional)
    - Step size to be used in line searches. If not specified, the step size
    will be calculated from the predicted optimum of the approximation. Not
    defined for SQP.

    alpha_mult(float,optional)
    - Factor by which alpha is adjusted during each iteration of the line
    search. Defaults to n_search. Not defined for SQP.

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

--------------------------------------------------------------------
FILE OUTPUTS
--------------------------------------------------------------------

optix.py will also output results to 3 .txt files. These 3 files are named optimize, gradient,
and evaluations, each appended with the user-specified file tag. The first two are written
to during runtime. The last is written to only after successful completion of the optimization.
Optimize mimics what is printed to the command line, giving information about the fitness,
point in the design space, magnitude of steps, etc. Gradient gives information about the 
objective and constraint gradients at each point in the optimization. Evaluations simply
outputs the value of the objective function at each point considered during optimization,
including points used to calculate finite differences.

--------------------------------------------------------------------
CONTACT
--------------------------------------------------------------------

For bugs, please create an issue on the github repository.
