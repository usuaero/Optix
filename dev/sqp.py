# A simple program for me to learn how to use the SQP algorithm
import numpy as np

def f(x):
    return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5

def g(x):
    return -(x[0]+0.25)**2+0.75*x[1]

def grad(func,x0,dx):
    del_f = np.zeros((len(x0),1))
    for i in range(len(x0)):
        delta_x = np.zeros((len(x0),1))
        delta_x[i] += dx
        del_f[i] = (func(x0+delta_x)-func(x0-delta_x))/(2*dx)
    return del_f

stopping_delta = 1e-12
dx = 0.01
delta_x = stopping_delta+1
iter = 0
n_vars = 2
n_cstr = 1

x_start = np.matrix([[-1],[4]])
x0 = np.copy(x_start)

iter += 1
print("Iteration {0} -----------".format(iter))

# Create quadratic approximation
f0 = f(x0)
del_f0 = grad(f,x0,dx)
g0 = g(x0)
del_g0 = grad(g,x0,dx)
del_2_L0 = np.identity(len(x0))

print("x0 = {0}".format(x0))
print("f0 = {0}".format(f0))
print("del_f0 = {0}".format(del_f0))
print("del_2_L0 = {0}".format(del_2_L0))
print("g0 = {0}".format(g0))
print("del_g0 = {0}".format(del_g0))
    
# Create the system of equations to solve for delta_x and lambda
n_eqns = n_vars+n_cstr # At first assume all constraints are binding
A = np.zeros((n_eqns,n_eqns))
b = np.zeros((n_eqns,1))
for i in range(n_eqns):
    for j in range(n_eqns):
        if i<n_vars and j<n_vars:
            print(del_2_L0[i][j])
            A[i][j] = del_2_L0[i][j]
        elif i<n_vars:
            A[i][j] = -del_g0[i]
        elif j<n_vars:
            A[i][j] = del_g0[j]
        else:
            A[i][j] = 0
    if i<n_vars:
        b[i] = -del_f0[i]
    else:
        b[i] = -g0

print("A:",A)
print("b:",b)

x_lambda = np.linalg.solve(A,b)
delta_x = x_lambda[0:n_vars]
l = x_lambda[n_vars:n_eqns]
print("delta x:",delta_x)
print("lambda:",l)

if l < 0: # Constraint is not binding
    print("Constraint is not binding!")
    A = np.zeros((n_vars,n_vars))
    b = np.zeros((n_vars,1))
    for i in range(n_vars):
        for j in range(n_vars):
            A[i][j] = del_2_L0[i][j]
        b[i] = -del_f0[i]
    
    print("A:",A)
    print("b:",b)

    delta_x = np.linalg.solve(A,b)
    l = np.zeros((n_cstr,1))
    print("delta x:",delta_x)

x1 = x0+delta_x
print("x1: {0}".format(x1))
P1 = f(x1)
for i in range(n_cstr):
    P1 += l[i]*abs(g(x1))
print("P1: {0}".format(P1))

while np.linalg.norm(delta_x) > stopping_delta:
    iter += 1
    print("Iteration {0} -----------".format(iter))

    # Create quadratic approximation
    f1 = f(x1)
    del_f1 = grad(f,x1,dx)
    g1 = g(x1)
    del_g1 = grad(g,x1,dx)

    # Update the Lagrangian Hessain
    del_L0 = del_f0-np.asscalar(l)*del_g0
    del_L1 = del_f1-np.asscalar(l)*del_g1
    gamma_0 = np.matrix(del_L1-del_L0)
    first = gamma_0*gamma_0.T/(gamma_0.T*np.matrix(delta_x))
    second = del_2_L0*(np.matrix(delta_x)*np.matrix(delta_x).T)*del_2_L0/(np.matrix(delta_x).T*del_2_L0*np.matrix(delta_x))
    del_2_L1 = np.asarray(del_2_L0+first-second)

    print("x1 = {0}".format(x1))
    print("f1 = {0}".format(f1))
    print("del_f1 = {0}".format(del_f1))
    print("del_2_L1 = {0}".format(del_2_L1))
    print("g1 = {0}".format(g1))
    print("del_g1 = {0}".format(del_g1))
        
    # Create the system of equations to solve for delta_x and lambda
    n_eqns = n_vars+n_cstr # At first assume all constraints are binding
    A = np.zeros((n_eqns,n_eqns))
    b = np.zeros((n_eqns,1))
    for i in range(n_eqns):
        for j in range(n_eqns):
            if i<n_vars and j<n_vars:
                A[i][j] = del_2_L1[i][j]
            elif i<n_vars:
                A[i][j] = -del_g1[i]
            elif j<n_vars:
                A[i][j] = del_g1[j]
            else:
                A[i][j] = 0
        if i<n_vars:
            b[i] = -del_f1[i]
        else:
            b[i] = -g1
    
    print("A:",A)
    print("b:",b)

    x_lambda = np.linalg.solve(A,b)
    delta_x = x_lambda[0:n_vars]
    l = x_lambda[n_vars:n_eqns]
    print("delta x:",delta_x)
    print("lambda:",l)

    if l < 0: # Constraint is not binding
        print("Constraint is not binding!")
        A = np.zeros((n_vars,n_vars))
        b = np.zeros((n_vars,1))
        for i in range(n_vars):
            for j in range(n_vars):
                A[i][j] = del_2_L1[i][j]
            b[i] = -del_f1[i]
        
        print("A:",A)
        print("b:",b)
    
        delta_x = np.linalg.solve(A,b)
        l = np.zeros((n_cstr,1))
        print("delta x:",delta_x)
    
    x2 = x1+delta_x
    print("x2: {0}".format(x2))
    P2 = f(x2)
    for i in range(n_cstr):
        P2 += l[i]*abs(g(x2))
    print("P2: {0}".format(P2))

    while P2 > P1:
        print("Stepped too far! Cutting step in half.")
        delta_x /= 2
        x2 = x1+delta_x
        print("x2: {0}".format(x2))
        P2 = f(x2)
        for i in range(n_cstr):
            P2 += l[i]*abs(g(x2))
        print("P2: {0}".format(P2))

    x0 = x1
    x1 = x2
    f0 = f1
    del_f0 = del_f1
    g0 = g1
    del_g0 = del_g1
    del_2_L0 = del_2_L1
    P0 = P1

print("\n---Optimum Point Found---")
print("x: {0}".format(x2))
print("objective: {0}".format(P2))
