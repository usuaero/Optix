import numpy as np

import optix.classes as c


def print_setup(n_vars, x_start, n_cstr, n_ineq_cstr, settings):
    # Starts the terminal output

    print()
    print("Optix.py from USU AeroLab")

    print()
    print('---------- Variables ----------')
    print("Optimizing in {0} variables.".format(n_vars))
    print("Initial guess:\n{0}".format(x_start))

    if settings.method == "grg" or settings.method == "sqp":
        print()
        print('---------- Constraints ----------')
        print('{0} total constraints'.format(n_cstr))
        print('{0} inequality constraints'.format(n_ineq_cstr))
        print('{0} equality constraints'.format(n_cstr-n_ineq_cstr))
        print('***Please note, Optix will rearrange constraints so the inequality constraints are listed first.')
        print('***Otherwise, the order is maintained.')

    print()
    print('---------- Settings ----------')
    print('            method: {0}'.format(settings.method))
    print('     obj func args: {0}'.format(settings.args))
    if settings.method != "nelder-mead":
        print('     initial alpha: {0}'.format(settings.alpha_init))
    print('    stopping delta: {0}'.format(settings.termination_tol))
    print('     max processes: {0}'.format(settings.max_processes))
    print('          file tag: {0}'.format(settings.file_tag))
    print('           verbose: {0}'.format(settings.verbose))
    if settings.use_finite_diff:
        if settings.central_diff:
            print('using central difference approximation')
        else:
            print('using forward difference approximation')
        print('  finite diff step: {0}'.format(settings.dx))
    print()


def get_constraints(constraints, pool, queue, settings):
    if constraints != None:
        n_cstr = len(constraints)
        n_ineq_cstr = 0
        g = []
        # Inequality constraints are stored first
        for constraint in constraints:
            if constraint["type"] == "ineq":
                n_ineq_cstr += 1
                grad = constraint.get("grad")
                constr = c.Constraint(
                    constraint["type"], constraint["fun"], pool, queue, settings, grad=grad)
                g.append(constr)
        for constraint in constraints:
            if constraint["type"] == "eq":
                grad = constraint.get("grad")
                constr = c.Constraint(
                    constraint["type"], constraint["fun"], pool, queue, settings, grad=grad)
                g.append(constr)
        g = np.array(g)
    else:
        g = None
        n_cstr = 0
        n_ineq_cstr = 0
    return g, n_cstr, n_ineq_cstr


def append_file(iter, o_iter, i_iter, obj_fcn_value, mag_dx, design_point, settings, **kwargs):
    # Writes a new iteration to the output files

    g = kwargs.get("g")
    del_g = kwargs.get("del_g")
    gradient = kwargs.get("gradient")

    msg = '{0:4d}, {1:5d}, {2:5d}, {3: 20.13E}, {4: 20.13E}'.format(
        iter, o_iter, i_iter, obj_fcn_value, mag_dx)
    values_msg = msg
    for value in design_point:
        values_msg = ('{0}, {1: 20.13E}'.format(
            values_msg, value))
    if not g is None:
        for cstr in g:
            values_msg = ('{0}, {1:20.13E}'.format(
                values_msg, cstr))
    print(values_msg)
    with open(settings.opt_file, 'a') as opt_file:
        print(values_msg, file=opt_file)

    if isinstance(gradient, np.ndarray):
        grad_msg = msg
        for grad in gradient:
            grad_msg = ('{0}, {1:20.13E}'.format(grad_msg, grad))
        if not del_g is None:
            for i in range(settings.n_cstr):
                for j in range(len(design_point)):
                    grad_msg = ('{0}, {1:20.13E}'.format(grad_msg, del_g[j,i]))
        with open(settings.grad_file, 'a') as grad_file:
            print(grad_msg, file=grad_file)


def eval_write(filename, header, q):
    with open(filename, 'w') as f:
        f.write(header+"\n")
        f.flush()
        while True:
            try:
                msg = q.get()
            except:
                continue
            if msg == 'kill':
                break
            f.write(msg+"\n")
            f.flush()
    return True


def format_output_files(n_vars, n_cstr, settings, pool, queue):
    # Sets up output files

    # Set up header
    opt_header = '{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}'.format('iter', 'outer', 'inner', 'fitness', 'mag(dx)')
    for i in range(n_vars):
        opt_header += ', {0:>20}'.format('x'+str(i))
    for i in range(n_cstr):
        opt_header += ', {0:>20}'.format('g'+str(i))

    # Open file
    opt_filename = "iterations"+settings.file_tag+".txt"
    settings.opt_file = opt_filename

    # Write header
    with open(opt_filename, 'w') as opt_file:
        opt_file.write(opt_header + '\n')

    # Set up gradient header
    grad_header = '{0:>84}  {1:>20}'.format(' ', 'df')
    for i in range(n_cstr):
        grad_header += (', {0:>'+str(21*n_vars)+'}').format('dg'+str(i))
    grad_header += '\n{0:>4}, {1:>5}, {2:>5}, {3:>20}, {4:>20}'.format(
        'iter', 'outer', 'inner', 'fitness', 'mag(dx)')
    for j in range(n_cstr+1):
        for i in range(n_vars):
            grad_header += ', {0:>20}'.format('dx'+str(i))

    # Open gradient file
    grad_filename = "gradient"+settings.file_tag+".txt"
    settings.grad_file = grad_filename

    # Write gradient header
    with open(grad_filename, 'w') as grad_file:
        grad_file.write(grad_header + '\n')

    # Print header to command line
    print(opt_header)


def eval_grad(x0, f, g, n_vars, n_cstr):
    # Evaluate gradients at specified point
    del_f0 = f.del_f(x0)
    del_g0 = np.zeros((n_vars, n_cstr))
    for i in range(n_cstr):
        del_g0[:,i] = g[i].del_g(x0)
    return del_f0, del_g0


def eval_constr(g, x1):
    n_cstr = len(g)
    g1 = np.zeros(n_cstr)
    for i in range(n_cstr):
        g1[i] = g[i].g(x1)
    return g1