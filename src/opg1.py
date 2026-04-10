import numpy as np
import matplotlib.pyplot as plt

# Oppgave 1c)
def f(x, y):
    y_1, y_2 = y
    dy = [y_2, -4*np.sin(2*x)]
    return np.array(dy)

def BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha):
    x_n = x_init
    y_n = y_init

    X = [x_n]
    Y = [y_n.copy()]
    H = [h0]
    n_accept = 0
    n_reject = 0
    h = h0

    k1 = f(x_n, y_n)

    while x_end - x_n > 0:
        h = min(h, x_end - x_n)
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
        'final_step_length': h
    }

    return np.array(X), np.array(Y), np.array(H), stats
"""
alpha = 0.8
tol = 1e-7
y_init = np.array([0, 2])
x_init = 0
x_end = 2 * np.pi
h0 = 0.1

X, Y, H, stats = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)


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
def root_finder(func, guess1:float, guess2:float, tol:float, params:tuple=(), max_iter:float=1e6):
    '''
    Finds the approximate root of a function using the secant method.
    Parameters:
        func: A scalar function whose root is to be found
        guess: Initial guesses for the root
        tol: The difference between the current and last iteration. A lower number corresponds to a higher precision.
        params (optional): Extra arguments to be passed to the function.
        max_iter: Maximum amount of iterations, prevents infinite while loop.
    Returns:
        The root of the function
    '''

    z_0, z_1 = guess1, guess2
    g_0, g_1 = func(z_0, *params), func(z_1, *params)

    diff = 1000
    iterations = 0
    while (diff >= tol):
        z_new = (z_0*g_1 - z_1*g_0)/(g_1 - g_0)     # The secant method
        diff = np.abs(z_new-z_1)

        # Update the parameters
        z_0 = z_1
        z_1 = z_new
        g_0, g_1 = func(z_0, *params), func(z_1, *params)

        iterations += 1
        if iterations > max_iter:
            print('Max iterations exceeded')
            return None

    return z_1

def g(z):
    return z + np.sin(z) + np.cos(z)

alpha = 0.8
tol = 1e-7
y_init = np.array([0, 2])
x_init = 0
x_end = 2 * np.pi
y_0 = 0
y_end = 0
h0 = 0.1

args = (x_init, x_end, y_0, y_end, f, h0, tol, alpha)

init_b = root_finder(err_as_func_of_b, -10, 10, 1e-5, params = (args), max_iter = 1e6)
print(init_b)

"""

def root_finder(func, guess1:float, guess2:float, tol:float, params:tuple=(), max_iter:float=1e6)->np.ndarray:
    '''
    Finds the approximate root of a function using the secant method and saves the solution for each iteration.
    Parameters:
        func: A scalar function whose root is to be found
        guess: Initial guesses for the root
        tol: The difference between the current and last iteration. A lower number corresponds to a higher precision.
        params (optional): Extra arguments to be passed to the function
        max_iter: Maximum amount of iterations, prevents infinite while loop
    Returns:
        An array containing the approximated roots for each iteration
    '''

    z_0, z_1 = guess1, guess2
    g_0, g_1 = func(z_0, *params), func(z_1, *params)

    diff = 1000
    iterations = 0
    root_vals = [z_0, z_1]
    while (diff >= tol):
        z_new = (z_0*g_1 - z_1*g_0)/(g_1 - g_0)     # The secant method
        diff = np.abs(z_new-z_1)

        # Update the parameters
        z_0 = z_1
        z_1 = z_new
        g_0, g_1 = func(z_0, *params), func(z_1, *params)

        root_vals.append(z_1)

        iterations += 1
        if iterations > max_iter:
            print('Max iterations exceeded')
            return np.array([])

    return np.array(root_vals)

def F(b:float, f, y_left:float, y_right:float, x_left:float, x_right:float, h0:float, alpha:float, tol:float)->float:
    '''
    Solves the IVP and evaluates the error of the solution at the right boundary.
    Parameters:
        b: y'(0)
        f: The right side of the differential equation
        y_left: The boundary value at the start of the interval
        y_right: The boundary value at the end of the interval
        x_left: The left boundary
        x_right: The right boundary
        h0: The initial step size for the IVP solver
        alpha: Optimism parameter for dynamic step size adjustment
        tol: The tolerated error of the IVP solver
    Returns:
        The error of the solution at the right boundary
    '''
    y_init = np.array([y_left, b])
    Y = BogackiShampine(x_left, x_right, y_init, f, h0, tol, alpha)[1]
    y_right_approx = Y[-1,0]
    err = abs(y_right_approx - y_right)
    return err

"""
def BVP_solver(f, y_left:np.ndarray, y_right:np.ndarray, x_left:float, x_right:float,
        b1:float, b2:float, alpha:float, tol:float)->np.ndarray:
    '''
    Solves the BVP using the shooting method, and saves the solution for each iteration.
    Parameters:
        f: The right hand side of the differential equation
        y_left: The boundary value at the start of the interval
        y_right: The boundary value at the end of the interval
        x_left: The left boundary
        x_right: The right boundary
        b1,b2: Initial guesses for y'(0)
        alpha: A parameter for dynamic step size adjustment in the IVP solver
        tol: The tolerated error of the appoximate solution
    Returns:
        The solution for each iteration of the solver
    '''

    diff = 1000
    while diff > tol:
        # First solve the IVP
        init_1 = np.array([y_left, b1])
        init_2 = np.array([y_left, b2])

        y1 = BogackiShampine(x_left, x_right, init_1, f, 0.01, tol, alpha)[1]
        y2 = BogackiShampine(x_left, x_right, init_2, f, 0.01, tol, alpha)[1]

        y1_right = y1[-1,0]
        y2_right = y2[-1,0]

        # Minimize the error at the right boundary using the secant method


alpha = 0.8
tol = 1e-7
h0 = 0.01
x_left = 0
x_right = 2 * np.pi
y_left = 0
y_right = 0
b1 = -1
b2 = 1

# Minimize F
args = (f, y_left, y_right, x_left, x_right, h0, alpha, tol)
b_list = root_finder(F, b1, b2, 1e-7, params = args)

# Plot the solution for each iteration of the solver
for i, b in enumerate(b_list):
    y_init = np.array([y_left, b])
    X, Y, H, stats = BogackiShampine(x_left, x_right, y_init, f, h0, tol, alpha)
    solution = Y[:,0]

    if i==0:
        lab = f'b={b:.3f} (Guess 1)'
    elif i==1:
        lab = f'b={b:.3f} (Guess 2)'
    else:
        lab = f'b={b:.3f}'

    plt.plot(X, solution, label = lab)

plt.xlabel('x')
plt.ylabel('y', rotation = 'horizontal')
plt.legend()
plt.grid('both')
plt.title('BVP solution for each iteration')

ticks = np.arange(0, 2*np.pi+np.pi/2, np.pi/2)
ticklabels = [f'{tick/np.pi}pi' for tick in ticks]
plt.xticks(ticks, ticklabels)

plt.show()
"""
def g(x, y):
    y_1, y_2 = y
    dy = [y_2, y_1 + np.sin(x)]

    return np.array(dy)
"""
alpha = 0.8
tol = 1e-7
h0 = 0.01
x_left = 0
x_right = 12
y_left = 0
y_right = 0
b1 = -1
b2 = 1

# Minimize F
args = (g, y_left, y_right, x_left, x_right, h0, alpha, tol)
b_list = root_finder(F, b1, b2, 1e-7, params = args)
y_init = np.array([y_left, b_list[-1]])

# Plot the solution
X, Y, H, stats = BogackiShampine(x_left, x_right, y_init, g, h0, tol, alpha)
y_sol = Y[:,0]

plt.plot(X, y_sol)
plt.xlabel('x')
plt.ylabel('y', rotation = 'horizontal')
plt.title('Solution to eq. 7')
plt.show()
"""

# Exercise 1g
from scipy.integrate import solve_bvp

def BVP_solver(func, bc_left, bc_right, x_left, x_right, y_init, x_size = 100):
    '''
    Uses scipy to solve a BVP
    Parameters:
      func: The right hand side of the BVP
      bc: The boundary conditions
      x_left/x_right: The left/right of the interval
      y_init: An initial guess for the solution
      x_size: The amount of points on the discretized x-axis
    Returns:
      The final x-axis and the solution at the nodes of this axis
    '''

    def bc_residuals(y_left:np.ndarray, y_right:np.ndarray)->np.ndarray:
        res_left = y_left[0] - bc_left
        res_right = y_right[0] - bc_right

        return np.array([res_left, res_right])

    assert len(y_init[0]) == x_size, 'y_init must have shape (n, x_size)'
    x = np.linspace(x_left, x_right, x_size)
    solution = solve_bvp(func, bc_residuals, x, y_init)

    return solution.x, solution.y

x_left = 0
x_right = 12
bc_left = 0
bc_right = 0
y_init = np.zeros((2, 100))

scipy_sol = BVP_solver(g, bc_left, bc_right, x_left, x_right, y_init)
x, Y = scipy_sol
y = Y[0,:]

plt.plot(x, y)
plt.show()