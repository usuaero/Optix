import os

import optix as opt


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

    x0 = [4.0, -1.0]
    
    optimum = opt.minimize(opt.test_functions.booth, x0, file_tag="_test", n_search=8, max_processes=8, line_search="bracket", termination_tol=1e-12)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))

    assert(abs(optimum.f) < 1e-12)
    assert(abs(optimum.x[0] - 1.0) < 1e-8)
    assert(abs(optimum.x[1] - 3.0) < 1e-8)


def test_GRG_1_eq_cstr_1_processor_booth():

    def g1(x):
        return x[1] - x[0]
    
    constraints = [{"type":"eq","fun":g1}]
    x0 = [-4.0, 0.0]
    
    optimum = opt.minimize(opt.test_functions.booth, x0, constraints=constraints, method='grg', file_tag="_test", max_processes=1, termination_tol=1e-12)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - 2.0) < 1e-3)
    assert(abs(optimum.x[0] - 2.0) < 1e-3)
    assert(abs(optimum.x[1] - 2.0) < 1e-3)


def test_GRG_1_ineq_cstr_1_processor_booth():

    def g1(x):
        return -x[0]**2 - x[1]**2 + 2.0

    constraints = [{
        "type" : "ineq",
        "fun" : g1
    }]
    x0 = [0.5, 0.0]

    optimum = opt.minimize(opt.test_functions.booth, x0, constraints=constraints, method='grg', file_tag='_test', max_processes=1, termination_tol=1.0e-12)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - 19.80005460153218) < 1e-3)
    assert(abs(optimum.x[0] - 0.89387647186033) < 1e-2)
    assert(abs(optimum.x[1] - 1.09594016855598) < 1e-2)


def test_GRG_1_ineq_1_eq_cstr_1_processor_booth():

    def g1(x):
        return -x[0]**2 - x[1]**2 + 2.0

    def g2(x):
        return x[1] - x[0]

    constraints = [{
        "type" : "ineq",
        "fun" : g1
    },
    {
        "type" : "eq",
        "fun" : g2
    }]
    x0 = [0.5, 0.0]

    optimum = opt.minimize(opt.test_functions.booth, x0, constraints=constraints, method='grg', file_tag='_test', max_processes=1, termination_tol=1.0e-12)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - 20.0) < 1e-3)
    assert(abs(optimum.x[0] - 1.0) < 1e-2)
    assert(abs(optimum.x[1] - 1.0) < 1e-2)


def test_SQP_2_ineq_3_eq_cstr_8_processors():

    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5+x[2]**2
    
    def g1(x):
        return -(x[0]+0.25)**2+0.75*(x[1]+2)
    
    def g2(x):
        return -x[1]+5
    
    def g3(x):
        return x[0]+x[1]
    
    def g4(x):
        return -x[0]+0.75
    
    def g5(x):
        return x[2]-1
    
    x0 = [-8.0, 0.5, 4.0]
    constraints = [{"type":"ineq","fun":g1},{"type":"eq","fun":g2},{"type":"ineq","fun":g3},{"type":"eq","fun":g4},{"type":"eq","fun":g5}]
    
    optimum = opt.minimize(f, x0, constraints=constraints, file_tag="_test", max_processes=8, termination_tol=1e-12)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - 24.75390625) < 1e-3)
    assert(abs(optimum.x[0] - 0.75) < 1e-2)
    assert(abs(optimum.x[1] - 5.0) < 1e-2)
    assert(abs(optimum.x[2] - 1.0) < 1e-2)


def test_SQP_4_ineq_cstr_8_processors():

    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5
    
    def g1(x):
        return -(x[0]+0.25)**2+0.75*(x[1]+2)
    
    def g2(x):
        return -x[1]+5
    
    def g3(x):
        return x[0]+x[1]
    
    def g4(x):
        return -x[0]+0.75
    
    x0 = [-8.0, -8.0]

    constraints = [{"type":"ineq","fun":g1},{"type":"ineq","fun":g2},{"type":"ineq","fun":g3},{"type":"ineq","fun":g4}]
    
    optimum = opt.minimize(f,x0,constraints=constraints,file_tag="_test",max_processes=8,termination_tol=1e-9,)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - 4.0625) < 1e-3)
    assert(abs(optimum.x[0] - 0.75) < 1e-2)
    assert(abs(optimum.x[1] - 0.5625) < 1e-2)


def test_SQP_5_ineq_cstr_8_processors():

    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5+x[2]**2
    
    def g1(x):
        return -(x[0]+0.25)**2+0.75*(x[1]+2)
    
    def g2(x):
        return -x[1]+5
    
    def g3(x):
        return x[0]+x[1]
    
    def g4(x):
        return -x[0]+0.75
    
    def g5(x):
        return x[2]-1
    
    x0 = [-10.0, 5.0, 8.0]
    constraints = [{"type":"ineq","fun":g1},{"type":"ineq","fun":g2},{"type":"ineq","fun":g3},{"type":"ineq","fun":g4},{"type":"ineq","fun":g5}]
    
    optimum = opt.minimize(f,x0,constraints=constraints,file_tag="_test",max_processes=8,termination_tol=1e-9)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f - 5.0625) < 1e-3)
    assert(abs(optimum.x[0] - 0.75) < 1e-2)
    assert(abs(optimum.x[1] - 0.5625) < 1e-2)
    assert(abs(optimum.x[2] - 1.0) < 1e-2)


def test_nelder_mead_3_dims_rosenbrock():

    x0 = [-4.0, 0.0, 2.0]
    
    optimum = opt.minimize(opt.test_functions.rosenbrock, x0, method='nelder-mead', file_tag="_test", max_processes=1, termination_tol=1e-12)

    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))

    assert(abs(optimum.f) < 1e-12)
    assert(abs(optimum.x[0] - 1.0) < 1e-6)
    assert(abs(optimum.x[1] - 1.0) < 1e-6)
    assert(abs(optimum.x[2] - 1.0) < 1e-6)


def test_cleanup():
    # Removes generated files

    files = os.listdir('.')
    for file in files:
        if "_test.txt" in file:
            os.remove(file)