from random import random

import multiprocessing_on_dill as multiprocessing
import optix as opt


def test_BFGS_quad_8_1():

    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5

    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f, x0, file_tag="_test", n_search=8, line_search="quadratic", termination_tol=1e-6, default_alpha=0.01)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(optimum.f - 4.0 < 1e-7)
    assert(abs(optimum.x[0] - 1.0) < 1e-3)
    assert(abs(optimum.x[1] - 1.0) < 1e-3)


def test_BFGS_3_var_brkt_8_8():

    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5
    
    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,max_processes=8,line_search="bracket",termination_tol=1e-6,alpha_mult=1.5,default_alpha=0.1)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(optimum.f - 4.0 < 1e-7)
    assert(abs(optimum.x[0] - 1.0) < 1e-3)
    assert(abs(optimum.x[1] - 1.0) < 1e-3)


def test_BFGS_brkt_4_1():

    def f(x):
        return -x[0]*x[1]+0.5*(x[0]**2+x[1]**2)
    
    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=4,line_search="bracket",termination_tol=1e-6,default_alpha=0.01)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(optimum.f - 4.0 < 1e-7)
    assert(abs(optimum.x[0] - 1.0) < 1e-3)
    assert(abs(optimum.x[1] - 1.0) < 1e-3)

def test_BFGS_brkt_8_8_alpha_mult():

    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5
    
    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,max_processes=8,line_search="bracket",termination_tol=1e-6,alpha_mult=1.5,default_alpha=0.1)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(optimum.f < 1e-7)
    assert(abs(optimum.x[0] - -6.15231132878745) < 1e-3)
    assert(abs(optimum.x[1] - -6.15231162316595) < 1e-3)


if __name__=="__main__":

    pass