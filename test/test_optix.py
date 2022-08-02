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


def test_GRG_2_ineq_2_eq_cstr_1_processor():

    def f(x):
        return x[0]**2+x[1]+x[2]**2+np.exp(x[0]+x[2])
    
    def g1(x):
        return -(x[0]**2+x[1]**2)+100
    
    def g2(x):
        return -x[0]-x[2]+1
    
    def g3(x):
        return x[1]+9
    
    def g4(x):
        return x[2]-2
    
    constraints = [{"type":"eq","fun":g1},{"type":"ineq","fun":g2},{"type":"ineq","fun":g3},{"type":"eq","fun":g4}]
    x0 = [1.0, 1.0, 1.0]
    
    optimum = opt.minimize(f,x0,constraints=constraints,method='grg',file_tag="_test",max_processes=1,termination_tol=1e-9,central_diff=False)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f) < 1e-12)
    assert(abs(optimum.x[0] - 1.0) < 1e-8)
    assert(abs(optimum.x[1] - 3.0) < 1e-8)
    assert(abs(optimum.x[2] - 3.0) < 1e-8)