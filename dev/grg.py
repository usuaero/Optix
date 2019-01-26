# A simple program for me to learn how to use the GRG algorithm
import numpy as np

def f(x):
    return x[0]**2+x[1]

def g_1(x): # Less-than constraint
    return x[0]**2+x[1]**2-25

def g_2(x): # Less-than constraint
    return x[0]+x[1]-1

def grad(func,x0,dx):
    del_f = np.zeros((len(x0),1))
    for i in range(len(x0)):
        delta_x = np.zeros((len(x0),1))
        delta_x[i] += dx
        del_f[i] = (func(x0+delta_x)-func(x0-delta_x))/(2*dx)
    return del_f

stopping_delta = 1e-16
dx = 0.001
delta_x = stopping_delta+1
iter = 0
n_vars = 2
n_cstr = 2

x_start = np.matrix([[11527.0/4500.0],[-7027.0/4500.0]])
x0 = np.copy(x_start)

while np.linalg.norm(delta_x) > stopping_delta:
    iter += 1
    print("Iteration {0} -----------".format(iter))
    print("x0 = {0}".format(x0))
    
    # Evaluate functions
    f0 = f(x0)
    g0 = np.zeros((n_cstr,1))
    g0[0] = g_1(x0)
    g0[1] = g_2(x0)
    del_f0 = grad(f,x0,dx)
    del_g_10 = grad(g_1,x0,dx)
    del_g_20 = grad(g_2,x0,dx)
    
    print("f0 = {0}".format(f0))
    print("g0 = {0}".format(g0))
    
    # Determine binding constraints, add slack variables, and partition variables
    cstr_b = (g0>=0) # All constraints are less-than
    n_binding = np.asscalar(sum(cstr_b))
    print("{0} binding constraints".format(n_binding))
    d_psi_d_x0 = np.concatenate((del_g_10,del_g_20),axis=1).T[np.repeat(cstr_b,2,axis=1)].reshape((n_binding,n_vars))
    print(d_psi_d_x0)
    cstr_b = cstr_b.flatten()
    
    s0 = g0[cstr_b].reshape((n_binding,1))
    print("s = {0}".format(s0))
    variables0 = np.concatenate((s0,x0),axis=0) # We place the slack variables first since we would prefer those be the independent variables
    
    # Search for independent variables and determine gradients
    z0 = np.zeros((n_vars,1))
    del_f_z0 = np.zeros((n_vars,1))
    d_psi_d_z0 = np.zeros((n_binding,n_vars))
    z_ind0 = []
    var_ind = -1
    for i in range(n_vars):
        while True:
            var_ind += 1
            if var_ind < n_binding and abs(variables0[var_ind])<1e-4: # Slack variable at limit
                    z0[i] = variables0[var_ind]
                    del_f_z0[i] = 0 # df/ds is always 0
                    d_psi_d_z0[i,i] = 1 # dg/ds is always 1
                    z_ind0.append(var_ind)
                    break
            else: # Design variable
                z0[i] = variables0[var_ind]
                del_f_z0[i] = del_f0[var_ind-n_binding]
                d_psi_d_z0[:,i]  = d_psi_d_x0[:,var_ind-n_binding]
                z_ind0.append(var_ind)
                break
    print("z = {0}".format(z0))
    
    # Search for dependent variables and determine gradients
    # Note the number of dependent variables is equal to the number of binding constraints
    y0 = np.zeros((n_binding,1))
    del_f_y0 = np.zeros((n_binding,1))
    d_psi_d_y0 = np.zeros((n_binding,n_binding))
    y_ind0 = []
    var_ind = -1
    for i in range(n_binding):
        while True:
            var_ind += 1
            if not var_ind in z_ind0: # The variable is not independent
                y0[i] = variables0[var_ind]
                del_f_y0[i] = del_f0[var_ind-n_binding]
                d_psi_d_y0[:,i] = d_psi_d_x0[:,var_ind-n_binding]
                y_ind0.append(var_ind)
                break
    
    print("y = {0}".format(y0))
    
    # Compute reduced gradient
    print("del_f(z) = {0}".format(del_f_z0))
    print("del_f(y) = {0}".format(del_f_y0))
    print("dpsi/dz = {0}".format(d_psi_d_z0))
    print("dpsi/dy = {0}".format(d_psi_d_y0))
    
    if n_binding != 0:
        del_f_r0 = (np.matrix(del_f_z0).T-np.matrix(del_f_y0).T*np.linalg.inv(d_psi_d_y0)*np.matrix(d_psi_d_z0)).T
    else:
        del_f_r0 = np.matrix(del_f_z0)
    
    # For now, the search direction is simply the direction of the reduced gradient
    s = -del_f_r0/np.linalg.norm(del_f_r0)
    print("Search Direction: {0}".format(s))
    
    # Conduct line search
    alpha = 0.5
    alpha_mult = 1.1
    n_search = 8
    while alpha>stopping_delta:
        z_search = np.zeros((n_vars,n_search))
        y_search = np.zeros((n_binding,n_search))
        x_search = np.zeros((n_vars,n_search))
        f_search = np.zeros(n_search)
        g_search = np.zeros((n_cstr,n_search))
        for i in range(n_search):
            z_search[:,i] = (z0+i*alpha*s).flatten()
            if n_binding != 0:
                y_search[:,i] = (y0-np.linalg.inv(d_psi_d_y0)*np.matrix(d_psi_d_z0)*np.matrix(i*alpha*s)).flatten()
            var_i = np.concatenate((z_search[:,i],y_search[:,i]))
            x_i = var_i[np.where(np.concatenate((z_ind0,y_ind0))>=n_binding)]
            x_search[:,i] = x_i
            f_search[i] = f(x_i)
            g_search[:,i] = [g_1(x_i),g_2(x_i)]
            # Drive dependent variables back to where violated constraints are satisfied
            iterations = 0
            while n_binding != 0 and (g_search[:,i]>0).any() and iterations<100:
                iterations += 1 # To avoid divergence of the N-R method
                g_search[:,i][np.where(g_search[:,i]<0)] = 0
                y_search[:,i] = y_search[:,i] - (np.linalg.inv(d_psi_d_y0)*np.matrix(g_search[cstr_b,i]).T).flatten()
                var_i = np.concatenate((z_search[:,i],y_search[:,i]))
                x_i = var_i[np.where(np.concatenate((z_ind0,y_ind0))>=n_binding)]
                x_search[:,i] = x_i
                f_search[i] = f(x_i)
                g_search[:,i] = [g_1(x_i),g_2(x_i)]
        min_ind = np.argmin(f_search)
        while (g_search[:,min_ind]>0).any():
            min_ind -= 1 # Step back to feasible space

        if min_ind == n_search-1: # Minimum at end of line search, step size must be increased
            alpha *= alpha_mult
            continue
        if min_ind == 0: # Minimum at beginning of line search, step size must be reduced
            alpha /= alpha_mult
            continue
        else: # Minimum is found in the middle of the line search
            print(f_search)
            x1 = x_search[:,min_ind].reshape((n_vars,1))
            break
    
    delta_x = x1-x0
    x0 = x1

print("---Optimum Found---")
print("x_opt = {0}".format(x0))
print("f_opt = {0}".format(f(x0)))
