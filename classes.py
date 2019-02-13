from collections import OrderedDict
import numpy as np
import os
import shutil
import time
import multiprocessing as mp

def grad(f,x,dx,central,max_processes):
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

    with mp.Pool(processes=max_processes) as pool:
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
        self.args = kwargs.get("args",())
        self.method = kwargs.get("method")
        self.termination_tol = kwargs.get("termination_tol",1e-12)
        self.verbose = kwargs.get("verbose",False)
        self.central_diff = kwargs.get("central_diff",True)
        self.file_tag = kwargs.get("file_tag","")
        self.max_processes = kwargs.get("max_processes",1)
        self.dx = kwargs.get("dx",0.001)
        self.alpha_d = kwargs.get("default_alpha",1)
        self.search_type = kwargs.get("line_search","bracket")
        self.n_search = kwargs.get("n_search",8)
        self.max_iterations = kwargs.get("max_iterations",np.inf)
        self.wolfe_armijo = kwargs.get("wolfe_armijo",1e-4)
        self.wolfe_curv = kwargs.get("wolfe_curv",0.9)
        self.plot_path = kwargs.get("plot_path",False)

        self.use_finite_diff = grad == None

        bounds = kwargs.get("bounds")
        constraints = kwargs.get("constraints")

        if self.method == None:
            if (bounds != None or constraints != None):
                self.method = "sqp"
            else:
                self.method = "bgfs"

        #Check for issues
        if self.method == "bgfs" and (bounds != None or constraints != None):
            raise ValueError("Bounds or constraints may not be specified for the simple BGFS algorithm.")

class OptimizerResult:
    obj_calls = 0
    cstr_calls = []

    def __init__(self,f,x,success,message,iterations):
        self.f = f
        self.x = x
        self.success = success
        self.message = message
        self.total_iter = iterations

class Constraint:
    """Class defining a constraint"""
    eval_calls = mp.Value('i',0)
    
    def __init__(self,cstr_type,f,settings,**kwargs):
        self.args = kwargs.get("args",())
        self.type = cstr_type
        self.fun = f
        self.gr = kwargs.get("grad")
        self.central_diff = settings.central_diff
        self.max_processes = settings.max_processes
        self.dx = settings.dx
        
    def g(self,x):
        with self.eval_calls.get_lock():
            self.eval_calls.value += 1
        return self.fun(x,*self.args)

    def del_g(self,x):
        if self.gr == None:
            return grad(self.g,x,self.dx,self.central_diff,self.max_processes)
        else:
            return self.gr(x)

class Objective:
    """Class defining objective function"""
    eval_calls = mp.Value('i',0)
    store_ind = mp.Value('i',0)
    f_points = mp.Array('d',10000)
    x_points = mp.Array('d',1000000)

    def __init__(self,f,n_vars,settings,**kwargs):
        self.args = settings.args
        self.fun = f
        self.n_vars = n_vars
        self.gr = kwargs.get("grad")
        self.hess = kwargs.get("hess")
        self.central_diff = settings.central_diff
        self.max_processes = settings.max_processes
        self.dx = settings.dx

    def f(self,x):
        n = len(x)
        f_val = np.asscalar(self.fun(x,*self.args))
        with self.eval_calls.get_lock():
            self.eval_calls.value += 1
        with self.store_ind.get_lock():
            with self.f_points.get_lock():
                fs = np.frombuffer(self.f_points.get_obj())
                fs[self.store_ind.value] = f_val
            with self.x_points.get_lock():
                xs = np.frombuffer(self.x_points.get_obj())
                xs[self.store_ind.value*n:(self.store_ind.value+1)*n] = x.flatten()
            self.store_ind.value += 1
        return f_val

    def del_f(self,x):
        if self.gr == None:
            return grad(self.f,x,self.dx,self.central_diff,self.max_processes)
        else:
            return self.gr(x)


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
