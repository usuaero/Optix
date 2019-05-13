import optix as opt
from random import random
import sys
import numpy as np
import traceback

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
    message.append( "Unexpected error: {0}\n".format(traceback.format_exc()))

print("".join(message))
