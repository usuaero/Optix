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

        Inputs:

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
                must be negative.
            fun (callable)
                Value of the constraint function. Must return a scalar. May 
                only have one argument, being an array of the design variables.
            grad (callable)
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
        Defaults to bracket.

        n_search(int,optional)
        -Number of points to be considered in the search direction. Defaults to
        8.

    Output

        Optimum(OptimizerResult)
        - Object containing information about the result of the optimization.
        Attributes include:
            x(array-like,shape(n,))
                Point in the design space where the optimization ended.
            success(bool)
                Indicates whether the optimizer exitted normally.
            message(str)
                Message about how the optimizer exitted.

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
        for constraint in constraints:
            if constraint["type"] == "ineq":
                n_ineq_cstr += 1
                grad = constraint.get("grad")
                g.append(c.Constraint(constraint["type"],constraint["fun"],settings,grad=grad))
        for constraint in constraints:
            if constraint["type"] == "eq":
                grad = constraint.get("grad")
                g.append(c.Constraint(constraint["type"],constraint["fun"],settings,grad=grad))
        g = np.reshape((n_cstr,1))
    else:
        g = None
        n_cstr = 0
        n_ineq_cstr = 0

    bounds = kwargs.get("bounds")

    #Begin formatting of output files
    header = ('{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}, {5:>20}'
        .format('iter', 'outer', 'inner', 'fitness', 'alpha', 'mag(dx)'))
    for i in range(n_vars):
        header += ', {0:>20}'.format('x'+str(i))

    opt_filename = "optimize"+settings.file_tag+".txt"
    settings.opt_file = opt_filename
    with open(opt_filename, 'w') as opt_file:
        opt_file.write(header + '\n')
    
    grad_filename = "gradient"+settings.file_tag+".txt"
    settings.grad_file = grad_filename
    with open(grad_filename, 'w') as grad_file:
        grad_file.write(header + '\n')

    #Print setup information to command line
    printSetup(n_vars,x_start,bounds,n_cstr,n_ineq_cstr,settings)
    
    iter = 0
    o_iter = 0
    mag_dx = 1.0
    x0 = np.copy(x_start)
    while mag_dx > settings.termination_tol:
        # Outer iteration; resets Hessian to identity matrix for BGFS update
        print('Beginning new update matrix')
        print(header)
        i_iter = 0
        alpha = settings.alpha_d

        f0 = f.f(x0)
        print(f0)
        del_f0 = f.del_f(x0)
        N = np.eye(n_vars)

        append_file(iter, o_iter, i_iter, f0, alpha, mag_dx, x0, del_f0, settings)
        while mag_dx > settings.termination_tol:
            # Inner iteration
            i_iter += 1
            append_file(iter, o_iter, i_iter, obj_value, alpha, mag_dx, design_point, gradient, settings)

            # Update the direction matrix
            dx = np.matrix(design_point - design_point_prev)  # 1 x n
            gamma = np.matrix(gradient - gradient_prev)  # 1 x n
            NG = N * np.transpose(gamma)  # n x 1
            denom = dx * np.transpose(gamma)  # 1 x 1
            N += ((1.0 + np.dot(gamma, NG) / denom)[0,0] * (np.transpose(dx) * dx) / denom
                  - ((np.transpose(dx) * (gamma * N)) + (NG * dx)) / denom
                 )

            # Calculate the second Wolfe condition for the previous
            # iteration. The curvature condition ensures that the slope is
            # sufficiently large to contribute to a reduction in the
            # objective function. If this condition is not met, the inner
            # loop is stopped and the direction matrix is reset to the
            # direction of steepest descent.
            if np.dot(s, gradient) < settings.wolfe_curv * np.dot(s, gradient_prev):
                print("Wolfe condition (ii): curvature condition not satisified!")
                break
                
            s = -np.dot(N, gradient)
            design_point_prev = np.copy(design_point)
            gradient_prev = np.copy(gradient)
            
            alpha, design_point = line_search(design_point[:], obj_value, gradient, s, obj_model, settings)
            
            dx = design_point - design_point_prev
            mag_dx = np.linalg.norm(dx)
            i_iter += 1
            iter += 1
            
#TODO:        save_file_name = 'optix_save.json'
#TODO:        self.write_optix_file(save_file_name)
        
        dx = design_point - design_point_init
        for i in range(settings.nconstraints):
            settings.penalty[i] = settings.penalty[i] * settings.penalty_factor[i]
    
    # Run the final case
    obj_value = obj_model.obj_fcn((design_point, -1))
    append_file(iter, o_iter, i_iter, obj_value, 0.0, mag_dx, design_point, gradient, settings)
    return (obj_value, design_point)
    
    
def line_search(design_point, obj_value, gradient, s, obj_model, settings):
    """Single method which will call the line search or quadratic line search as needed"""
    if settings.line_search_type == 'quadratic':
        return line_search_quad(design_point, obj_value, gradient, s, obj_model, settings)
    else:
        return line_search_lin(design_point, obj_value, s, obj_model, settings)
    
    
def line_search_lin(design_point, obj_value, s, obj_model, settings):
    """Perform line search to find the value of alpha corresponding to a minimum in the objective function.

    This subroutine evaluates the objective function multiple times in the
    direction of s. It then selects the minimum point and the two bracketing
    points and fits a parabola to these to find the vertex. The step length
    is adjusted if no bracketted minimum can be found.

    Inputs
    ------
    design_point = A list of design variables defining the design point at
                   which to begin the line search
 
    obj_value = The value of the objective function at the specified design
                point.

    s = The direction matrix defining the direction in which to conduct the
        line search.
    
    obj_model = The objective model object
    
    settings = The optimization settings object

    Outputs
    -------
    alpha_min = The alpha corresponding to the minimum value of the objective
                function in the current direction
                
    design_point = The design point corresponding to the minimum value of the
                   objective function in the current direction
    """
    if settings.verbose:
        print('line search ----------------------------------------------------------------------------')

    s_norm = np.linalg.norm(s)
    alpha = max(settings.default_alpha, 1.1 * settings.stop_delta / s_norm)
    alpha_mult = settings.nsearch / 2.0
    
    found_min = False
    while not found_min:
        xval, yval = run_mult_cases(settings.nsearch, alpha, s, design_point, obj_value, obj_model)
        if settings.verbose:
            for i in range(settings.nsearch + 1):
                print('{0:5d}, {1:15.7E}, {2:15.7E}'.format(i, xval[i], yval[i]))
        
        mincoord = yval.index(min(yval))
        if yval[1] > yval[0]:
            if (alpha * s_norm) < settings.stop_delta:
                print('Line search within stopping tolerance: alpha = {0}'.format(alpha))
                return alpha, design_point
            elif mincoord == 0:
                if settings.verbose: print('Too big of a step. Reducing alpha')
                alpha /= alpha_mult
            else:
                if mincoord < settings.nsearch: found_min = True
                else: alpha *= alpha_mult
        else:
            if settings.verbose: print('mincoord = {0}'.format(mincoord))
            if mincoord == 0: return alpha, design_point
            elif mincoord < settings.nsearch: found_min = True
            else:
                if settings.verbose: print('Too small of a step. Increasing alpha')
                alpha *= alpha_mult
    
    a1 = xval[mincoord - 1]
    a2 = xval[mincoord]
    a3 = xval[mincoord + 1]
    f1 = yval[mincoord - 1]
    f2 = yval[mincoord]
    f3 = yval[mincoord + 1]
    
    da = a2 - a1
    alpha = a1 + da * (4.0 * f2 - f3 - 3.0 * f1) / (2.0 * (2.0 * f2 - f3 - f1))
    if alpha > a3 or alpha < a1:
        if f2 > f1: alpha = a1
        else: alpha = a2
        
    for i in range(len(design_point)):
        design_point[i] += alpha * s[i]

#TODO:    self.constraints()
    if settings.verbose: print('Final alpha = {0}'.format(alpha))
    return alpha, design_point


def line_search_quad(design_point, obj_value, gradient, s, obj_model, settings):
    """Perform a quadratic line search to minimize the objective function
    
    This subroutine evaluates the objective function multiple times in the
    direction of s and fits a parabola to the results using a least-squares
    algorithm to identify the minimum value for the objective function in
    the current direction.
    
    Inputs
    ------
    design_point = A list of design variables defining the design point at
                   which to begin the line search
 
    obj_value = The value of the objective function at the specified design
                point.

    gradient = The gradient of the objective function at the specified design
               point.
                
    s = The direction matrix defining the direction in which to conduct the
        line search.
    
    obj_model = The objective model object
    
    settings = The optimization settings object

    Outputs
    -------
    alpha_min = The alpha corresponding to the minimum value of the objective
                function in the current direction
                
    design_point = The design point corresponding to the minimum value of the
                   objective function in the current direction
    """
    if settings.verbose:
        print('Performing quadratic line search...')

    # Determine the initial step size to use in the direction of s
    stop_delta = settings.stop_delta / np.linalg.norm(s)
    alpha = max(settings.default_alpha, 1.1 * stop_delta)

    found_min = False
    line_search_min = (0.0, obj_value)
    alpha_history = []

    # Determine the maximum number of adjustments in alpha to attempt.
    nadjust = int(np.ceil(-np.log10(stop_delta) /
        np.log10(np.ceil(settings.nsearch / 2))))

    for i in range(nadjust):
        # Compute the objective function multiple times in the direction of s
        alphas, obj_vals = run_mult_cases(settings.nsearch, alpha, s,
                design_point, obj_value, obj_model)
        alpha_history.append(alpha)

        # Save the minimum data point for later comparisons
        ind = obj_vals.index(min(obj_vals))
        if obj_vals[ind] < line_search_min[1]:
            line_search_min = (alphas[ind], obj_vals[ind])
        alpha_min_est = line_search_min[0]

        if settings.verbose:
            for j in range(settings.nsearch + 1):
                print('{:5d}, {:23.15E}, {:23.15E}'.format(
                    j, alphas[j], obj_vals[j]))
            
        # Check for invalid results
        if np.isnan(obj_vals).any():
            print('Found NaN')
            break

        # Check for plateau
        if min(obj_vals) == max(obj_vals):
            print('Objective function has plateaued')
            break
            
        # Check stopping criteria
        if alpha <= stop_delta and ind < settings.nsearch - 1:
            print('stopping criteria met')
            break
            
        # Fit a quadratic through the data and find the resulting minimum
        q = quadratic(np.asarray(alphas), np.asarray(obj_vals))
        (alpha_min_est, obj_value_est) = q.vertex()
        
        if (alpha_min_est is None or alpha_min_est < 0 or not q.convex() or
                q.rsq < settings.rsq_tol):
            # Can't find a better minimum by curve fitting all data points.
            # Try a quadratic through minimum and two closest neighbors.
            left = min(max(ind - 1, 0), len(alphas) - 3)
            right = left + 3
            q = quadratic(np.asarray(alphas[left:right]), np.asarray(obj_vals[left:right]))
            (alpha_min_est, obj_value_est) = q.vertex()
            
            if (alpha_min_est is None or alpha_min_est < 0 or not q.convex()):
                if ind == settings.nsearch:
                    # If minimum is at the end, try increasing alpha
                    alpha_min_est = alpha * 4
                elif ind == 0:
                    # If minimum is at beginning, try reducing alpha
                    alpha_min_est = alpha / 2
                else:
                    # Can't find a better minimum by curve fitting,
                    # so just use the current minimum.
                    break
    
        # Set alpha for next iteration
        alpha = max(alpha / settings.max_alpha_factor, min(alpha * settings.max_alpha_factor,
                alpha_min_est / np.ceil(settings.nsearch / 2.0)))
        print('alpha for next iteration = {0}'.format(alpha))
        
        # Check to see if we've already tried close to this alpha
        alpha_close = min(alpha_history, key=lambda a: abs(a - alpha) / alpha)
        delta = abs(alpha_close - alpha) / alpha
        if delta <= settings.alpha_tol:
            break
    
    # Update design point based on alpha that minimized objective function
    alpha_min = line_search_min[0]
    design_point[:] += alpha_min * s[:]

    # Calculate the first Wolfe condition. This is a measure of how much the
    # step length (alpha) decreases the objective function, but has no effect
    # on the behavior of the quadratic line search.
    armijo = obj_value + settings.wolfe_armijo * alpha_min * np.dot(s, gradient)
    if line_search_min[1] > armijo:
        print("Wolfe condition (i): Armijo rule not satisfied.")

#TODO:    self.constraints()
    if settings.verbose: print('Line search minimized at alpha = {0}'.format(alpha_min))
    return alpha_min, design_point

def run_mult_cases(nevals, alpha, s, dp0, obj_fcn0, obj_model):
    # Calculate linearly distributed alphas for the line search
    alphas = [(i * alpha) for i in range(nevals + 1)]
    
    # Set up the design points in the direction of the line search
    design_points = []
    for i in range(nevals):
        design_points.append([(dp0[j] + alphas[i + 1] * s[j]) for j in range(len(dp0))])

    # Evaluate the function at each design point
    obj_fcn_values = [obj_fcn0] + obj_model.evaluate(design_points)
    
    return alphas, obj_fcn_values
    
    
def append_file(iter, o_iter, i_iter, obj_fcn_value, alpha, mag_dx, design_point, gradient, settings):
    msg = ('{0:4d}, {1:5d}, {2:5d}, {3: 20.13E}, {4: 20.13E}, {5: 20.13E}'
        .format(iter, o_iter, i_iter, obj_fcn_value, alpha, mag_dx))
    values_msg = msg
    for value in design_point:
        values_msg = ('{0}, {1: 20.13E}'.format(values_msg, value))
    print(values_msg)
    with open(settings.opt_file, 'a') as opt_file:
        print(values_msg, file = opt_file)

    grad_msg = msg
    for grad in gradient:
        grad_msg = ('{0}, {1: 20.13E}'.format(grad_msg, grad))
    with open(settings.grad_file, 'a') as grad_file:
        print(grad_msg, file = grad_file)


class quadratic(object):
    """Class for fitting, evaluating, and interrogating quadratic functions

    This class is used for fitting a quadratic function to a data set
    evaluating the function at specific points, and determining the
    characteristics of the function.
    """
    def __init__(self, x, y):
        """
        Construct a quadratic object from tabulated data.
        Quadratic is of the form f(x) = ax^2 + bx + c

        Inputs
        ------
        x = List of independent values
        y = List of dependent values
        """
        super().__init__()

        # Calculate the quadratic coefficients
        x_sq = [xx**2 for xx in x]
        A = np.vstack([x_sq, x, np.ones(len(x))]).T
        self.a, self.b, self.c = np.linalg.lstsq(A, y)[0]
        
        # Calculate the coefficient of determination
        f = [self.f(xx) for xx in x]
        ssres = ((f - y)**2).sum()
        sstot = ((y - y.mean())**2).sum()

        if abs(sstot) < zero:
            # Data points actually formed a horizontal line
            self.rsq = 0.0
        else:
            self.rsq = 1 - ssres / sstot


    def convex(self):
        """
        Test to see if the quadratic is convex (opens up).
        """
        # Convex has positive curvature (2nd derivative)
        # f"(x) = 2a, so a > 0 corresponds to convex
        return (self.a > 0)


    def vertex(self):
        """
        Find the coordinates of the vertex
        """
        if self.a != 0.0:
            # Find x where f'(x) = 2ax + b = 0
            x = -0.5 * self.b / self.a
            return (x, self.f(x))
        else:
            # Quadratic is actually a line, no minimum!
            return (None, None)


    def f(self, x):
        """
        Evaluate the quadratic function at x
        """
        if x is not None: return self.a * x**2 + self.b * x + self.c
        else: return None

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

