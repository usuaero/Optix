import numpy as np


def sphere(x):
    """N-dimensional sphere function.
    Minimum is at f(0,...,0) = 0."""

    return np.sum(x*x)


def rosenbrock(x):
    """N-dimensional Rosenbrock function.
    Minimum is at f(1,...,1) = 0."""

    f = 0.0
    for i in range(len(x)-1):
        f += 100.0*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

    return f


def beale(z):
    """2-dimensional Beale function.
    Minimum is at f(3,0.5) = 0."""

    x = z[0]
    y = z[1]
    return (1.5 - x + x*y)**2 + (2.25 - x - x*y**2)**2 + (2.625 - x + x*y**3)**2


def booth(z):
    """2-dimensional Booth function.
    Minimum is at f(1,3) = 0."""

    x = z[0]
    y = z[1]
    return (x + 2.0*y - 7.0)**2 + (2.0*x + y - 5.0)**2


def matyas(z):
    """2-dimensional Matyas function.
    Minimum is at f(0,0) = 0."""

    x = z[0]
    y = z[1]
    return 0.26(x**2 + y**2) - 0.48*x*y