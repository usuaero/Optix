from myjson import myjson
from collections import OrderedDict
import numpy
import os
import shutil
import time

import multiprocessing

numpy.set_printoptions(precision = 14)

zero = 1.0e-20


class objective_model(object):
    """Defines the evaluation model of an objective function
    
    This class defines a model consisting of an objective function that can be
    evaluated individually and with gradients. The class allows this function
    to be executed at multiple design points, either synchronously or
    asynchronously (using the Python multiprocessing module).
    
    Two methods for evaluating the function are used. The first method is
    required and evaluates only the function itself, while the second method
    is optional and evaluates the function and its gradient with respect to the
    design variables. If the second function is not provided, a second-order
    central differencing scheme is used to approximate gradients when needed.
    """
    def __init__(self,
                 objective_fcn,
                 objective_fcn_with_gradient = None,
                 max_processes = 1,
                 dx = 0.01
                ):
        """Constructor
        
        Constructor for the objective_model class
        
        Inputs
        ------
        objective_fcn:
            The function to evaluate through this model. The function should
            accept two arguments. The first argument is a list of design
            variable values specifying a fixed design point. The second
            argument is the case number assigned to the evaluation. The
            function should return a single result that is the value of the
            objective function at the specified design point.

        objective_fcn_with_gradient:
            The function to evaluate through this model when gradients are
            requested. The function should accept the same two arguments as
            objective_fcn. The function should return two results: the value
            of the objective function at the specified design point and the
            gradient of the objective function at the specified design point.
            Note that the first return value should be equal to the
            objective_fcn return value for a given design point.
            
            If objective_fcn_with_gradient is not specified (default), a
            second-order central difference approximation will be used with
            objective_fcn when gradients are needed.
            
        max_processes:
            The maximum number of simultaneous processes to use. If set to 1
            (default), all function evaluations will be executed sequentially.
            Otherwise, the Python multiprocessing module will be used to
            execute multiple function evaluations simultaneously.
            
        dx:
            The perturbation size to use if the second-order central difference
            approximation is used to estimate the gradient of the function. If
            objective_fcn_with_gradient is specified, dx is not used.
        """
        # Set the objective function
        self.obj_fcn = objective_fcn
        
        # Set the gradient function
        if objective_fcn_with_gradient is not None:
            # Use the user-specified function to calculate gradients
            self.obj_fcn_with_gradient = objective_fcn_with_gradient
        else:
            # Use central differencing scheme to approximate the gradient
            self.obj_fcn_with_gradient = self.central_difference
            self.dx = dx
            
        # Set the maximum number of simultaneous processes
        self.max_processes = max_processes
        
        # Initialize the number of function/gradient evaluations
        self.n_fcn_evals = 0
        self.n_grad_evals = 0

        
    def evaluate(self, design_points):
        """Evaluate the function at multiple design points
        
        This routine evaluates the objective function at multiple design
        points, each specified by a list of design variables.
        
        Inputs
        ------
        design_points = A list of design points. A design point is defined by
                        a list of design variables that are passed into the
                        objective function for a single evaluation. Therefore,
                        design_points is a list of lists.
               
        Outputs
        -------
        objective = A list of results from the objective function,
                    corresponding to the value of the objective function at
                    each design point specified.
        
        """
        objective = []
        if self.max_processes > 1:
            # Execute function at multiple design points in parallel
            with multiprocessing.Pool(processes = self.max_processes) as pool:
                args = [(design_points[i], i + 1) for i in range(len(design_points))]
                objective = pool.map(self.obj_fcn, args)
        else:
            # Execute function at each design point sequentially
            for i in range(len(design_points)):
                objective.append(self.obj_fcn((design_points[i], i + 1)))

        # Increment the number of function evaluations
        self.n_fcn_evals += len(design_points)
        
        return objective

        
    def evaluate_gradient(self, design_point):
        """Evaluate the function and its gradient at a specified design point
        
        This routine evaluates the objective function and its gradient at a
        single design point.
        
        Inputs
        ------
        design_point = The design point at which to evaluate the function and
                       its gradient. The design point is defined as a list of
                       values, one value for each design variable required by
                       the objective function.
                       
        Outputs
        -------
        objective = The value of the objective function at the specified design
                    point.
                    
        gradient = The gradient of the objective function at the specified
                   design point.
        """
        objective, gradient = self.obj_fcn_with_gradient((design_point, 0))
        self.n_fcn_evals += 1
        self.n_grad_evals += 1

        return objective, gradient
    
    
    def central_difference(self, args):
        """Approximate the gradient of a function using central differencing
        
        This routine approximates the gradient of a specified function with respect
        to all design variables at a specified design point. The gradient is
        approximated using second-order central differencing.
        
        Inputs
        ------
        design_point = A list of design variables defining the design point at
                       which the objective function and its gradient will be
                       evaluated
                       
        case_id = The case ID to use for the objective function evaluation. The
                  case IDs for gradient evaluations will be incremented
                  sequentially starting from (case_id + 1).
        """
        design_point = args[0]
        case_id = args[1]
        # Initialize a list of objective function arguments by perturbing each
        # variable by +/-dx
        n_design_vars = len(design_point)
        argslist = [(design_point[:], i) for i in range(case_id, case_id + 2 * n_design_vars + 1)]
        for i in range(1, n_design_vars + 1):
            argslist[i][0][i - 1] += self.dx
            argslist[i + n_design_vars][0][i - 1] -= self.dx
        
        if self.max_processes > 1:
            # Execute function at multiple design points in parallel
            with multiprocessing.Pool(processes = self.max_processes) as pool:
                results = pool.map(self.obj_fcn, argslist)
        else:
            # Execute function at each design point sequentially
            results = []
            for a in argslist:
                results.append(self.obj_fcn(a))
                
        # Get the objective function value at the specified design point
        objective = results[0]
        
        # Calculate the gradient of the objective function from results
        # at the perturbed design points
        gradient = []
        for i in range(1, n_design_vars + 1):
            gradient.append((results[i] - results[i + n_design_vars]) / (2.0 * self.dx))

        return objective, gradient

        
class settings(object):
    """Defines the various settings used by the optimization algorithm
    
    
    
    """
    def __init__(self):
        self.opt_file = 'optimization.txt'
        self.grad_file = 'gradient.txt'
        self.verbose = False
        
        self.nvars = 0
        self.varnames = []
        self.vars = []
        self.grad = []
        
        self.default_alpha = 0.0
        self.stop_delta = 0.0
        self.nsearch = 8
        self.line_search_type = 'quadratic'
        
        self.nconstraints = 0
        self.constrainttype = []
        self.constraintnames = []
        self.constraintvalues = []
        self.penalty = []
        self.penalty_factor = []


    def load(settings_file):
        self = settings()
        input = myjson(settings_file)
            
        # Read settings from JSON file
        json_settings = input.get('settings', OrderedDict)
        self.default_alpha = json_settings.get('default_alpha', float)
        self.stop_delta = json_settings.get('stop_delta', float)
        self.nsearch = json_settings.get('n_search', int)
        self.line_search_type = json_settings.get('line_search_type', str, 'quadratic')
        self.verbose = json_settings.get('verbose', bool, False)  # optional
        
        # Read variables
        json_variables = input.get('variables', OrderedDict)
        self.nvars = len(json_variables.data)
        self.varnames = []
        self.varsinit = []
        self.opton = []
        for var_name in json_variables.data:
            self.varnames.append(var_name)
            self.varsinit.append(json_variables.get(var_name +'.init', float))
            self.opton.append(json_variables.get(var_name + '.opt', str) == 'on')
        
        # Read constraints
        json_constraints = input.get('constraints', OrderedDict, OrderedDict())
        self.nconstraints = len(json_constraints.data)
        self.nconstraints = 0
        self.contrainttype = []
        self.constraintnames = []
        self.constraintvalues = []
        self.penalty = []
        self.penalty_factor = []
        valid_constraint_types = ['=', '<', '>']
        for const_name in json_constraints.data:
            json_constraint_data = json_constraints.get(const_name, OrderedDict)
            const_type = json_constraint_data.get('type', str)
            if const_type not in valid_constraint_types:
                print('Unknown constraint type: {0}. Constraint {1} skipped.'
                    .format(const_type, const_name))
                print('Valid constraint types are {0}.'.format(valid_constraint_types))
                continue
            
            self.constrainttype.append(const_type)
            self.constraintnames.append(const_name)
            self.constraintvalues.append(json_constraint_data.get('value', float))
            self.penalty.append(json_constraint_data.get('penalty', float))
            self.penalty_factor.append(json_constraint_data.get('factor', float))
            
        return self


def optimize(obj_model, settings):
    """
    """
    header = ('{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}, {5:>20}'
        .format('iter', 'outer', 'inner', 'fitness', 'alpha', 'mag(dx)'))
    for name in settings.varnames: header += ', {0:>20}'.format(name)
    with open(settings.opt_file, 'w') as opt_file:
        opt_file.write(header + '\n')
    
    with open(settings.grad_file, 'w') as grad_file:
        grad_file.write(header + '\n')
    
    print('---------- Variables ----------')
    for i in range(settings.nvars):
        print('{0} = {1}'.format(settings.varnames[i], settings.varsinit[i]))
    print('')
    
    print('---------- Constraints ----------')
    for i in range(settings.nconstraints):
        print('{0}, {1}, {2}'.format(settings.constraintnames[i],
            settings.constrainttype[i], settings.constraintvalues[i]))
    print('')    
    
    print('---------- Settings ----------')
    print('      default alpha: {0}'.format(settings.default_alpha))
    print('     stopping delta: {0}'.format(settings.stop_delta))
    print('')
    
    iter = 0
    o_iter = 0
    mag_dx = 1.0
    design_point = settings.varsinit[:]
    while mag_dx > settings.stop_delta:
        design_point_init = numpy.copy(design_point)
        i_iter = 0
        
        print('Constraint Penalties')
        for i in range(settings.nconstraints):
            print('{0} {1}'.format(settings.constraintnames[i], settings.penalty[i]))
            
        print('Beginning new update matrix')
        print(header)
        
        alpha = 0.0
        while mag_dx > settings.stop_delta:
            obj_value, gradient = obj_model.evaluate_gradient(design_point)
            append_file(iter, o_iter, i_iter, obj_value, alpha, mag_dx, design_point, gradient, settings)
            
            # Initialize N to the identity matrix
            if (i_iter == 0):
                N = numpy.zeros((settings.nvars, settings.nvars))  # n x n
                for i in range(settings.nvars):
                    N[i, i] = 1.0
            else:
                dx = numpy.matrix(design_point - design_point_prev)  # 1 x n
                gamma = numpy.matrix(gradient - gradient_prev)  # 1 x n
                NG = N * numpy.transpose(gamma)  # n x 1
                denom = dx * numpy.transpose(gamma)  # 1 x 1
                N += ((1.0 + numpy.dot(gamma, NG) / denom)[0,0] * (numpy.transpose(dx) * dx) / denom
                      - ((numpy.transpose(dx) * (gamma * N)) + (NG * dx)) / denom
                     )
                
            s = -numpy.dot(N, gradient)
            design_point_prev = numpy.copy(design_point)
            gradient_prev = numpy.copy(gradient)
            
            alpha, design_point = line_search(design_point[:], obj_value, s, obj_model, settings)
            
            dx = design_point - design_point_prev
            mag_dx = numpy.linalg.norm(dx)
            i_iter += 1
            iter += 1
            
#TODO:        save_file_name = 'optix_save.json'
#TODO:        self.write_optix_file(save_file_name)
        
        dx = design_point - design_point_init
        mag_dx = numpy.linalg.norm(dx)
        append_file(iter, o_iter, i_iter, obj_value, alpha, mag_dx, design_point, gradient, settings)
        
        o_iter += 1
        for i in range(settings.nconstraints):
            settings.penalty[i] = settings.penalty[i] * settings.penalty_factor[i]
    
    # Run the final case
    obj_value = obj_model.obj_fcn((design_point, 0))
    append_file(iter, o_iter, i_iter, obj_value, 0.0, mag_dx, design_point, gradient, settings)
    return (obj_value, design_point)
    
    
def line_search(design_point, obj_value, s, obj_model, settings):
    if settings.line_search_type == 'quadratic':
        return line_search_quad(design_point, obj_value, s, obj_model, settings)
    else:
        return line_search_lin(design_point, obj_value, s, obj_model, settings)
    
    
def line_search_lin(design_point, obj_value, s, obj_model, settings):
    if settings.verbose:
        print('line search ----------------------------------------------------------------------------')

    s_norm = numpy.linalg.norm(s)
    alpha = max(settings.default_alpha, 1.1 * settings.stop_delta / s_norm)
    alpha_mult = settings.nsearch / 2.0
    
    found_min = False
    while not found_min:
        xval, yval = run_mult_cases(settings.nsearch, alpha, s, design_point, obj_value, obj_model)
        if settings.verbose:
            for i in range(settings.nsearch + 1):
                print('{0}, {1}, {2}'.format(i, xval[i], yval[i]))
        
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
            else: alpha *= alpha_mult
    
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


def line_search_quad(design_point, obj_value, s, obj_model, settings):
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
    s_norm = numpy.linalg.norm(s)
    alpha = max(settings.default_alpha, 1.1 * settings.stop_delta / s_norm)
    alpha_mult = settings.nsearch / 2.0
    
    # Compute the objective function multiple times in the direction of s
    alphas, obj_vals = run_mult_cases(settings.nsearch, alpha, s, design_point, obj_value, obj_model)
    if settings.verbose:
        for i in range(settings.nsearch + 1):
            print('{0}, {1}, {2}'.format(i, alphas[i], obj_vals[i]))
    
    # Fit the data points to a parabola
    alphas_sq = [a**2 for a in alphas]
    A = numpy.vstack([alphas_sq, alphas, numpy.ones(len(alphas))]).T
    a, b, c = numpy.linalg.lstsq(A, obj_vals)[0]
    
    # Find the minimum of the parabola
    # Make sure alpha_min is a minimum (a > 0) and is positive (b < 0)
    if a > zero and b < zero:
        alpha_min = -0.5 * b / a

    else:
        # Parabolic curve fit did not work, find the minimum data point
        ind = obj_vals.index(min(obj_vals))
        
        # If minimum is at beginning or end of line search, use that data point
        if ind < 1 or ind > settings.nsearch - 2:
            alpha_min = alphas[ind]

        # If minimum and two closest neighbors form a concave-up parabola,
        # fit the parabola and find its minimum. 
        elif alphas[ind - 1] > alphas[ind] or alphas[ind + 1] > alphas[ind]:
            alphas = alphas[ind - 1 : ind + 2]
            alphas_sq = alphas_sq[ind - 1 : ind + 2]
            obj_vals = obj_vals[ind - 1 : ind + 2]
            A = numpy.vstack([alphas_sq, alphas, numpy.ones(len(alphas))]).T
            a, b, c = numpy.linalg.lstsq(A, obj_vals)[0]
            # Already enforced constraints on parabolic curve fit, so use the minimum.
            alpha_min = -0.5 * b / a
        # Can't fit a parabola through the data, so just use the minimum point
        else:
            alpha_min = alphas[ind]
    
    # Update design point based on alpha that minimized objective function
    for i in range(len(design_point)):
        design_point[i] += alpha_min * s[i]

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
