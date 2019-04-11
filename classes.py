from collections import OrderedDict
import numpy as np
import os
import shutil
import time
import multiprocessing as mp

def grad(f,x,dx,central,pool):
    n = len(x)

    if central:
        argslist = []
        for i in range(n):
            dx_v = np.zeros((n,1))
            dx_v[i] = dx
            argslist.append(x-dx_v)
            argslist.append(x+dx_v)
    else:
        argslist = []
        for i in range(n):
            dx_v = np.zeros((n,1))
            dx_v[i] = dx
            argslist.append(x+dx_v)
        argslist.append(x)

    results = pool.map(f,argslist)

    gradient = np.zeros((n,1))
    if central:
        for i in range(n):
            gradient[i] = (results[2*i+1]-results[2*i])/(2*dx)
    else:
        for i in range(n):
            gradient[i] = (results[i]-results[-1])/dx
    return gradient

class Settings:
    """Contains settings used by optimizer"""

    def __init__(self,**kwargs):

        # Objective function args
        self.args = kwargs.get("args",())

        # General args
        self.method = kwargs.get("method")
        self.termination_tol = kwargs.get("termination_tol",1e-9)
        self.grad_tol = kwargs.get("grad_tol",1e-9)
        self.verbose = kwargs.get("verbose",False)
        self.central_diff = kwargs.get("central_diff",True)
        self.file_tag = kwargs.get("file_tag","")
        self.max_processes = kwargs.get("max_processes",1)
        self.dx = kwargs.get("dx",0.01)
        self.max_iterations = kwargs.get("max_iterations",np.inf)

        self.use_finite_diff = kwargs.get("jac") == None

        # BFGS args
        self.n_search = kwargs.get("n_search",8)
        self.alpha_d = kwargs.get("default_alpha",None)
        self.alpha_mult = kwargs.get("alpha_mult",self.n_search-1)
        self.search_type = kwargs.get("line_search","bracket")
        self.rsq_tol = kwargs.get("rsq_tol",0.8)
        self.wolfe_armijo = kwargs.get("wolfe_armijo",1e-4)
        self.wolfe_curv = kwargs.get("wolfe_curv",0.9)
        self.hess_init = kwargs.get("hess_init",1.0)

        if self.wolfe_curv < self.wolfe_armijo:
            raise ValueError("Wolfe conditions improperly specified.")

        # SQP args
        self.strict_penalty = kwargs.get("strict_penalty",True)

        # GRG args
        self.cstr_tol = kwargs.get("cstr_tol",1e-4)

        # Bounds and constraints
        bounds = kwargs.get("bounds")
        constraints = kwargs.get("constraints")

        # Assign method if not specified
        if self.method == None:
            if (bounds != None or constraints != None):
                self.method = "sqp"
            else:
                self.method = "bfgs"

        #Check for issues
        if self.method == "bgfs" and (bounds != None or constraints != None):
            raise ValueError("Bounds or constraints may not be specified for the simple BGFS algorithm.")

class OptimizerResult:
    """Return data from the 'minimize' function"""

    def __init__(self,f,x,success,message,iterations,obj_calls,cstr_calls=[]):
        self.f = f
        self.x = x
        self.success = success
        self.message = message
        self.total_iter = iterations
        self.obj_calls = obj_calls
        self.cstr_calls = cstr_calls

class Constraint:
    """Class defining a constraint"""
    eval_calls = mp.Value('i',0)
    
    def __init__(self,cstr_type,f,pool,queue,settings,**kwargs):
        self.args = kwargs.get("args",())
        self.type = cstr_type
        self.fun = f
        self.gr = kwargs.get("grad")
        self.central_diff = settings.central_diff
        self.max_processes = settings.max_processes
        self.dx = settings.dx
        self.pool = pool
        self.queue = queue
        with self.eval_calls.get_lock():
            self.eval_calls.value = 0
        
    def g(self,x):
        with self.eval_calls.get_lock():
            self.eval_calls.value += 1
        return self.fun(x,*self.args)

    def del_g(self,x):
        if self.gr == None:
            return grad(self.g,x,self.dx,self.central_diff,self.pool)
        else:
            return self.gr(x)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

class Objective:
    """Class defining objective function"""
    eval_calls = mp.Value('i',0)

    def __init__(self,f,pool,queue,settings,**kwargs):
        self.args = settings.args
        self.fun = f
        self.gr = kwargs.get("grad")
        self.hess = kwargs.get("hess")
        self.central_diff = settings.central_diff
        self.max_processes = settings.max_processes
        self.dx = settings.dx
        self.pool = pool
        self.queue = queue
        with self.eval_calls.get_lock():
            self.eval_calls.value = 0

    def f(self,x):
        n = len(x)
        f_val = np.asscalar(self.fun(x,*self.args))
        with self.eval_calls.get_lock():
            self.eval_calls.value += 1
        msg = "{0:>20}".format(f_val)
        for value in x:
            msg += ", {0:>20}".format(np.asscalar(value))
        self.queue.put(msg)
        return f_val

    def del_f(self,x):
        if self.gr == None:
            return grad(self.f,x,self.dx,self.central_diff,self.pool)
        else:
            return self.gr(x)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


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
        self.a, self.b, self.c = np.linalg.lstsq(A,y,rcond=None)[0]
        
        # Calculate the coefficient of determination
        f = [self.f(xx) for xx in x]
        ssres = ((f - y)**2).sum()
        sstot = ((y - y.mean())**2).sum()

        if abs(sstot) < 1e-14:
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
