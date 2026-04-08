import numpy as np
import matplotlib.pyplot as plt

# Oppgave 1 c)

alpha = 0.8
tol = 1e-3
y_init = np.array([0, 2])
x_init = 0
x_end = 2 * np.pi
h0 = 0.1

def f(x, y):
    y_1 = np.sin(2*x)
    y_2 = 2 * np.cos(2*x)
    dy = [y_2, -4*y_1]
    return np.array(dy), y


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
            n_accept += 1

        else:
            n_reject += 1

        # oppdater steglengde
        if est == 0:
            h = 2*h 
        else:
            h = alpha * h * (tol / est)**(1/3)
            
        stats = {
            'n_accept': n_accept,
            'n_reject': n_reject,
            'final_step_size': h
        }

    return X, Y, H , stats

X, Y, H, stats = BogackiShampine(x_init, x_end, y_init, f, h0, tol, alpha)

# Plotting y(x) and y'(x)
plt.plot(X, Y[:, 0], label='y(x)')
plt.plot(X, Y[:, 1], label="y'(x)")
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('y(x) and y\'(x) as function of x')
plt.grid()

# Plotting the step size h
plt.plot(X, H, label='Step length h')
plt.xlabel('x')
plt.ylabel('h')
plt.title('Step length varies as a function of x')
plt.grid()  
plt.show()


