import numpy as np
import matplotlib.pyplot as plt

def root_finder(func, guess1:float, guess2:float, tol:float):
    '''
    Finds the approximate root of a function using the secant method.
    Parameters:
        func: A scalar function whose root is to be found
        guess: Initial guesses for the root
        tol: The difference between the current and last iteration. A lower number corresponds to a higher precision.
    Returns:
        The root of the function
    '''
    z_0, z_1 = guess1, guess2
    g_0, g_1 = func(z_0), func(z_1)

    diff = 1000
    while diff >= tol:
        z_new = (z_0*g_1 - z_1*g_0)/(g_1 - g_0)     # The secant method
        diff = np.abs(z_new-z_1)

        # Update the parameters
        z_0 = z_1
        z_1 = z_new
        g_0, g_1 = func(z_0), func(z_1)

    return z_1

def g(z):
    return z + np.sin(z) + np.cos(z)

print(root_finder(g, -2, 2, 1e-4))