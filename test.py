import optix as o
import numpy as np

def myFun(arg):
    x = arg[0]
    return x[0]**2-2*x[0]*x[1]+4*x[1]**2

settings_file = 'settings.json'

my_model = o.objective_model(myFun)
my_settings = o.settings.load(settings_file)
print(o.optimize(my_model,my_settings))
