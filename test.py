import optix as opt
from random import random
import sys

message = []

try:
    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5
    
    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,max_processes=8,line_search="bracket",termination_tol=1e-6,default_alpha=0.01)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    message.append( "Passed bracketting, 8 search point, 8 process BGFS test.\n")
except Exception:
    message.append( "Failed bracketting, 8 search point, 8 process BGFS test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))
    

try:
    def f(x):
        return -x[0]*x[1]+0.5*(x[0]**2+x[1]**2)
    
    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=4,line_search="bracket",termination_tol=1e-6,default_alpha=0.01)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    message.append( "Passed bracketting, 4 search point, single process BGFS test.\n")
except:
    message.append( "Failed bracketting, 4 search point, single process BGFS test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

try:
    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5

    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,line_search="quadratic",termination_tol=1e-6,default_alpha=0.01)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    message.append( "Passed quadratic, 8 search point, single process BGFS test.\n")
except:
    message.append( "Failed quadratic, 8 search point, single process BGFS test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

try:
    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5
    
    x0 = [-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,max_processes=8,line_search="bracket",termination_tol=1e-6,alpha_mult=1.5,default_alpha=0.1)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    message.append( "Passed bracketting, 8 search point, 8 process alpha multiplier BGFS test.\n")
except:
    message.append( "Failed bracketting, 8 search point, 8 process alpha multiplier BGFS test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

try:
    def f(x):
        return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+x[2]**2+5
    
    x0 = [-10+20*random(),-10+20*random(),-10+20*random()]
    
    optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,max_processes=8,line_search="bracket",termination_tol=1e-6,default_alpha=0.01)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    message.append( "Passed three vaiable, bracketting, 8 search point, 8 process BGFS test.\n")
except:
    message.append( "Failed three vaiable, bracketting, 8 search point, 8 process BGFS test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

try:
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
    
    x0 = [-10+20*random(),-10+20*random()]
    constraints = [{"type":"ineq","fun":g1},{"type":"ineq","fun":g2},{"type":"ineq","fun":g3},{"type":"ineq","fun":g4}]
    
    optimum = opt.minimize(f,x0,constraints=constraints,file_tag="_test",max_processes=8,termination_tol=1e-9,)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    cstr_calls = 0
    for calls in optimum.cstr_calls:
        cstr_calls += calls
    print("Constraint calls: {0}".format(cstr_calls))
    message.append( "Passed 4 inequality constraint, 8 process SQP test.\n")
except:
    message.append( "Failed 4 inequality constraint, 8 process SQP test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

try:
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
    
    x0 = [-10+20*random(),-10+20*random(),-10+20*random()]
    constraints = [{"type":"ineq","fun":g1},{"type":"ineq","fun":g2},{"type":"ineq","fun":g3},{"type":"ineq","fun":g4},{"type":"ineq","fun":g5}]
    
    optimum = opt.minimize(f,x0,constraints=constraints,file_tag="_test",max_processes=8,termination_tol=1e-9)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    cstr_calls = 0
    for calls in optimum.cstr_calls:
        cstr_calls += calls
    print("Constraint calls: {0}".format(cstr_calls))
    message.append( "Passed 3 variable, 5 inequality constraint, 8 process SQP test.\n")
except:
    message.append( "Failed 3 variable, 5 inequality constraint, 8 process SQP test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

optimum.obj_calls = -1000000

try:
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
    
    x0 = [-10+20*random(),-10+20*random(),-10+20*random()]
    constraints = [{"type":"ineq","fun":g1},{"type":"eq","fun":g2},{"type":"ineq","fun":g3},{"type":"eq","fun":g4},{"type":"eq","fun":g5}]
    
    optimum = opt.minimize(f,x0,constraints=constraints,file_tag="_test",max_processes=8,termination_tol=1e-9,central_diff=False)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x))
    print("Function calls: {0}".format(optimum.obj_calls))
    cstr_calls = 0
    for calls in optimum.cstr_calls:
        cstr_calls += calls
    print("Constraint calls: {0}".format(cstr_calls))
    message.append( "Passed 3 variable, 2 inequality constraint, 3 equality constraint, forward differencing 8 process SQP test.\n")
except:
    message.append( "Failed 3 variable, 2 inequality constraint, 3 equality constraint, forward differencing 8 process SQP test.\n")
    message.append( "Unexpected error: {0}\n".format(sys.exc_info()[0]))

print(" ".join(message))
