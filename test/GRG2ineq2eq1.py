import optix as opt
from random import random
import sys
import numpy as np
import traceback

message = []
try:
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
    x0 = [-4.0, 0.0, 2.0]
    
    optimum = opt.minimize(f,x0,constraints=constraints,method='grg',file_tag="_test",max_processes=1,termination_tol=1e-9,central_diff=False, verbose=True)
    print("Optimum value: {0}".format(optimum.f))
    print("Optimum point: {0}".format(optimum.x.flatten()))
    print("Return message: {0}".format(optimum.message))
    print("Function calls: {0}".format(optimum.obj_calls))
    cstr_calls = 0
    for calls in optimum.cstr_calls:
        cstr_calls += calls
    print("Constraint calls: {0}".format(cstr_calls))
    message.append("Passed 3 variable, 2 inequality constraint, 2 equality constraint, forward differencing, 1 process GRG test.\n")
except:
    message.append("Failed 3 variable, 2 inequality constraint, 2 equality constraint, forward differencing, 1 process GRG test.\n")
    message.append( "Unexpected error: {0}\n".format(traceback.format_exc()))

print("".join(message))
