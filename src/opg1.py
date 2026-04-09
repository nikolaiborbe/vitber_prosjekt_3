import numpy as np
import matplotlib.pyplot as plt

# Oppgave 1c
def f(x, y):
    y_1 = np.sin(2*x)
    y_2 = 2 * np.cos(2*x)
    dy = [y_2, -4*y_1]
    return np.array(dy)

def BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha):
    x_n = x_init
    y_n = y_init

    X = [x_n]
    Y = [y_n.copy()]
    H = [h0]
    n_accept = 0
    n_reject = 0
    
    k1 = f(x_n, y_n)

    while x_end - x_n > 0:
        h = min(h0, x_end - x_n)
        k2 = f(x_n + h / 2, y_n + h * k1 / 2)
        k3 = f(x_n + 3 * h / 4, y_n + 3 * h * k2 / 4)
        y_n1 = y_n + h * (2 * k1 + 3 * k2 + 4 * k3) / 9
        x_n1 = x_n + h
        k4 = f(x_n + h, y_n1)
        z_n1 = y_n + h * (7 * k1 + 6 * k2 + 8 * k3 + 3 * k4) / 24
        est = np.linalg.norm(y_n1 - z_n1)

        if est < tol:
            x_n = x_n1
            y_n = y_n1
            k1 = k4

            X.append(x_n)
            Y.append(y_n.copy())
            H.append(h)
            n_accept += 1

        else:
            n_reject += 1

        # oppdater steglengde
        h = alpha * h * (tol / est)**(1/3)
            
    stats = {
        'n_accept': n_accept,
        'n_reject': n_reject,
        'step_length': h
    }

    return np.array(X), np.array(Y), np.array(H), stats

X, Y, H, stats = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)

alpha = 0.8
tol = 1e-3
y_init = np.array([0, 2])
x_init = 0
x_end = 2 * np.pi
h0 = 0.1

# Plotting y(x) and y'(x)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(X, Y[:, 0], label='y(x)')
ax1.plot(X, Y[:, 1], label="y'(x)")
ax1.set_xlabel('x')
ax1.set_ylabel('y(x)')
ax1.set_title('y(x) and y\'(x) as function of x')
ax1.grid()
ax1.legend()

# Plotting the step size h
ax2.plot(X, H, label='Step length h')
ax2.set_xlabel('x')
ax2.set_ylabel('h')
ax2.set_title('Step length varies as a function of x')
ax2.grid() 
ax2.legend() 

plt.show()




# Oppgave 1e
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

