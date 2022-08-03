from asyncio import constants
from random import random

import optix as opt
import numpy as np


def test_BFGS_quadratic_line_search_8_steps_1_processor_booth():

    x0 = [3.0, -3.0]
    
    optimum = opt.minimize(opt.test_functions.booth, x0, file_tag="_test", n_search=8, line_search="quadratic", termination_tol=1e-12)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(abs(optimum.f) < 1e-12)
    assert(abs(optimum.x[0] - 1.0) < 1e-8)
    assert(abs(optimum.x[1] - 3.0) < 1e-8)


def test_BFGS_bracket_line_search_8_steps_8_processors_booth():

    x0 = [4.0, -4.0]
    
    optimum = opt.minimize(opt.test_functions.booth, x0, file_tag="_test", n_search=8, max_processes=8, line_search="bracket", termination_tol=1e-12)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(abs(optimum.f) < 1e-12)
    assert(abs(optimum.x[0] - 1.0) < 1e-8)
    assert(abs(optimum.x[1] - 3.0) < 1e-8)


def test_GRG_1_ineq_1_eq_cstr_1_processor():

    def f(x):
        return x[0]**2+x[1]+x[2]**2+np.exp(x[0]+x[2])
    
    def g1(x):
        return -(x[0]**2+x[1]**2)+100
    
    def g2(x):
        return -x[0]-x[2]+1
    
    constraints = [{"type":"eq","fun":g1},{"type":"ineq","fun":g2}]
    x0 = [-4.0, 0.0, 1.0]
    
    optimum = opt.minimize(f, x0, constraints=constraints, method='grg', file_tag="_test", max_processes=1, termination_tol=1e-12)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - -9.268089022051072) < 1e-3)
    assert(abs(optimum.x[0] - -0.27461726295264) < 1e-3)
    assert(abs(optimum.x[1] - -9.99623304398555) < 1e-3)
    assert(abs(optimum.x[2] - -0.25923516334851) < 1e-3)


def test_GRG_1_ineq_cstr_1_processor_rosenbrock():

    def g1(x):
        return -x[0]**2 - x[1]**2 + 2.0

    constraints = [{
        "type" : "ineq",
        "fun" : g1
    }]
    x0 = [0.5, 0.0]

    optimum = opt.minimize(opt.test_functions.rosenbrock, x0, constraints=constraints, method='grg', file_tag='_test', max_processes=1, termination_tol=1.0e-12)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f) < 1e-3)
    assert(abs(optimum.x[0] - 2.0) < 1e-2)
    assert(abs(optimum.x[1] - 2.0) < 1e-2)