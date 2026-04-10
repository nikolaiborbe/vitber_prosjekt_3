import numpy as np
import matplotlib.pyplot as plt

# Oppgave 1c)
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

alpha = 0.8
tol = 1e-7
y_init = np.array([0, 2])
x_init = 0
x_end = 2 * np.pi
h0 = 0.1

X, Y, H, stats = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)


ax1.plot(X, Y[:, 0], label=f'y(x), tol={tol}')
ax1.plot(X, Y[:, 1], label="y'(x)")
ax1.set_xlabel('x')
ax1.set_title('y(x) and y\'(x) as function of x')
ax1.vlines(np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2), ymin=-2, ymax=2, colors='r', linestyles='dashed', label='x = n * pi/2')
ax1.grid()
ax1.legend()

# Plotting the step size h
ax2.plot(X, H, label='Step length h')
ax2.set_xlabel('x')
ax2.set_ylabel('h')
ax2.set_title('Step length varies as a function of x')
ax2.vlines(np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2), ymin=0, ymax=max(H), colors='r', linestyles='dashed', label='x = n * pi/2')
ax2.grid() 
ax2.legend() 



#Exercise 1 d) 

ax3 = fig.add_subplot(223)

tol_array = np.linspace(1e-8,1e-1,100)
err_array = np.zeros(len(tol_array))

for i,tol in enumerate(tol_array):
    X, Y, H, stats = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)
    err_array[i] = np.mean(abs(np.sin(2*X) - Y[:,0]))

ax3.plot(tol_array, err_array)
ax3.legend()
ax3.set_title("Error as a function of tolerance")
ax3.set_xlabel("Error")
ax3.set_ylabel("Tol")
ax3.grid()

tol = 1e-7  

ax4 = fig.add_subplot(224)

alpha_array = np.linspace(0.5,0.9,10)
step_array = np.zeros(len(alpha_array))
for i, alpha in enumerate(alpha_array):
    X, Y, H, stats = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)
    step_array[i] = len(X)

ax4.plot(alpha_array, step_array)
ax4.set_title("Number of time steps as a function of alpha")
ax4.set_xlabel("alpha")
ax4.set_ylabel("number of time steps")
ax4.grid()
plt.show()

#Oppgave 1e

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


#Exercise 1 f)

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
print(f'{b_list[-1]:.10f}')


"""
def err_as_func_of_b(b,x_init, x_end,y_0, y_end, f, h0, tol, alpha):
    
    Calculates the error in the endpoint of a function from a referance, 
    with a "guess" b for the start value of the derivative.
    
    y_init = np.array([y_0,b])
    Y = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)[1]
    err = abs(Y[-1,0] - y_end)
    return err

def solve_boundary_value_problem():
    b_true = root_finder(err_as_func_of_b,-1,0,1e-6,params=parameters)
    return b_true

print("Oppgave 1f):",solve_boundary_value_problem())
"""


#Exercise 1 g)







#Exercise 1 h)

print("1h:")

def h(x, y, z):
    return np.sin(x+y+z)

print(root_finder(h, -1, 1, 1e-5, params=(5, 456)))