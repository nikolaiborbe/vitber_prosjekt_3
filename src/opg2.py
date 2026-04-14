import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import tqdm
from utils import (
    calculate_Ns
)


# Oppgave 2 a)
def transform_matrix_to_vector(M):
    """
    Transforming a 2x2 complex matrix M into a real vector m.
    """
    m = []
    real = np.real(M)
    imag = np.imag(M)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            m.append(real[i][j])

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            m.append(imag[i][j])

    return np.array(m)


def transform_vector_to_matrix(m):
    """
    Transforming a real vector m back into a 2x2 complex matrix M.
    """
    n = len(m) // 2
    M = np.zeros((2, 2))
    real = m[:n]
    imag = m[n:]

    M = real + 1j * imag
    return M.reshape((2, 2))


# Oppgave 2 b)
def transform_four_8comp_vectors_to_32comp_vector(m1, m2, m3, m4):
    """
    Transforming four 8-component real vectors into a single 32-component real vector v.
    """
    return np.concatenate((m1, m2, m3, m4))

def transform_32comp_vector_to_four_8comp_vectors(v):
    """
    Transforming a 32-component real vector v back into four 8-component real vectors.
    """
    m1 = v[:8]
    m2 = v[8:16]
    m3 = v[16:24]
    m4 = v[24:32]
    return m1, m2, m3, m4

# Oppgave 2 c) 
def transform_matrices_to_32comp_vector(gamma, gamma_tilde, omega, omega_tilde):
    """
    Transforming four unknown 2x2 complex matrices into a single 32-component real vector v.
    """
    # Transforming each matrix to a 8-component real vector
    m_gamma = transform_matrix_to_vector(gamma)
    m_gamma_tilde = transform_matrix_to_vector(gamma_tilde)
    m_omega = transform_matrix_to_vector(omega)
    m_omega_tilde = transform_matrix_to_vector(omega_tilde)

    # Transforming the four 8-component vectors into a single 32-component vector
    v = transform_four_8comp_vectors_to_32comp_vector(m_gamma, m_gamma_tilde, m_omega, m_omega_tilde)

    return v


def transform_32comp_vector_to_matrices(v):
    """
    Transforms a 32-component real vector v into four 2x2 complex matrices.
    """
    # Transforming the 32-component vector back to four 8-component vectors
    m_gamma, m_gamma_tilde, m_omega, m_omega_tilde = transform_32comp_vector_to_four_8comp_vectors(v)

    # Transforming each 8-component vector back to a 2x2 complex matrix
    gamma = transform_vector_to_matrix(m_gamma)
    gamma_tilde = transform_vector_to_matrix(m_gamma_tilde)
    omega = transform_vector_to_matrix(m_omega)
    omega_tilde = transform_vector_to_matrix(m_omega_tilde)

    return gamma, gamma_tilde, omega, omega_tilde

# Exercise 2d
def dv(v:np.ndarray, epsilon:float)->np.ndarray:
    '''
    Computes the derivative of the 32 component vector v wrt. x.
    '''
    gamma, gamma_tilde, omega, omega_tilde = transform_32comp_vector_to_matrices(v)

    # Gamma and gamma_tilde have trivial derivatives
    d_gamma, d_gamma_tilde = omega, omega_tilde

    # The identity matrix
    I = np.identity(2)

    # Find N and N_tilde
    N_inv = I - np.matmul(gamma, gamma_tilde)
    N = np.linalg.inv(N_inv)

    N_tilde_inv = I - np.matmul(gamma_tilde, gamma)
    N_tilde = np.linalg.inv(N_tilde_inv)

    # Multiply necessary matrices
    product1 = np.matmul(omega, N_tilde)
    product2 = np.matmul(gamma_tilde, omega)
    product3 = np.matmul(omega_tilde, N)
    product4 = np.matmul(gamma, omega_tilde)

    # The derivatives of omega and omega_tilde
    d_omega = -2j*(epsilon + 0.01j)*gamma - 2*np.matmul(product1, product2)
    d_omega_tilde = -2j*(epsilon + 0.01j)*gamma_tilde - 2*np.matmul(product3, product4)

    dv = transform_matrices_to_32comp_vector(d_gamma, d_gamma_tilde, d_omega, d_omega_tilde)

    return dv

# Exercise 2f
def boundary_conditions(v_left, v_right, gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R, l):
    '''
    The boundary conditions for the system.
    '''
    gamma_0, gamma_tilde_0, omega_0, omega_tilde_0 = transform_32comp_vector_to_matrices(v_left)
    gamma_1, gamma_tilde_1, omega_1, omega_tilde_1 = transform_32comp_vector_to_matrices(v_right)


    # Find N and N_tilde for each interface metal L and R
    N_L, N_tilde_L = calculate_Ns(gamma_L, gamma_tilde_L)
    N_R, N_tilde_R = calculate_Ns(gamma_R, gamma_tilde_R)

    # The identity matrix
    I = np.identity(2)

    # Define some more matrices
    M1 = I - np.matmul(gamma_0, gamma_tilde_L)
    M2 = gamma_L - gamma_0

    M3 = I - np.matmul(gamma_tilde_0, gamma_L)
    M4 = gamma_tilde_L - gamma_tilde_0

    M5 = I - np.matmul(gamma_1, gamma_tilde_R)
    M6 = gamma_R - gamma_1

    M7 = I - np.matmul(gamma_tilde_1, gamma_R)
    M8 = gamma_tilde_R - gamma_tilde_1

    # The boundary conditions
    bc1 = omega_0 + (1/(3*l)) * M1 @ N_L @ M2
    bc2 = omega_tilde_0 + (1/(3*l)) * M3 @ N_tilde_L @ M4
    bc3 = omega_1 - (1/(3*l)) * M5 @ N_R @ M6
    bc4 = omega_tilde_1 - (1/(3*l)) * M7 @ N_tilde_R @ M8

    # Vectorize
    res = transform_matrices_to_32comp_vector(bc1, bc2, bc3, bc4)

    return res

def bc_residuals_normal_metal(v_left, v_right, l):
    # In this case the ricatti matrices for the interface metals are all zero
    gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))

    # Compute the residuals at the boundaries
    res = boundary_conditions(v_left, v_right, gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R, l)

    return res

# Exercise 2e
def h(x: np.ndarray, vec: np.ndarray, epsilon:float=1) -> np.ndarray:
    """
    The right hand side of the differential equation
    Parameters:
        x: Vector with m components
        vec: 32 x m matrix that contains the vector v at each position in x

    Returns:
        a 32 x m matrix that contains d/dx(v) at each position in x
    """
    dv_vec = np.zeros_like(vec)

    for i in range(vec.shape[1]): #iterate through each column of vec
        dv_vec[:, i] = dv(vec[:, i], epsilon) 

    return np.array(dv_vec)


# Exercise 2g
'''
m = 101
x = np.linspace(0, 1, m)
y = np.zeros((32, m))
epsilon_list = [0,1,2]

sol_list = []
for epsilon in epsilon_list:
  # Use lambda to remove epsilon as a parameter, such that h works along with the BVP solver
  solution = solve_bvp(lambda x, vec: h(x, vec, epsilon = epsilon), lambda x, y: bc_residuals_normal_metal(x, y, l), x, y)
  sol_list.append((solution.x, solution.y))
'''
# Exercise 2h
def green_function(gamma:np.ndarray, gamma_tilde:np.ndarray):
    N, N_tilde = calculate_Ns(gamma, gamma_tilde)
    I = np.identity(2)

    g11 = 2*N - I
    g12 = 2*np.matmul(N, gamma)
    g21 = -2*np.matmul(N_tilde, gamma_tilde)
    g22 = -2*N_tilde + I

    g = np.block([[g11,g12],[g21,g22]])

    return g

def density_of_states(g:np.ndarray)->float:
    '''
    Computes the normalized density of stated
    Parameters:
        The green function
    '''
    rho_hat = np.diag([1,1,-1,-1])
    product = np.matmul(rho_hat, g)
    trace = np.trace(product)

    D = np.real(trace)/4

    return D

def from_solution_to_density_of_states(x:np.ndarray, y:np.ndarray)->np.ndarray:
    '''
    Finds the density of states as a function of position x, given a solution (x,y) from the BVP solver.
    Parameters:
        x: The x-array returned form the BVP solver (solution.x)
        y: The y-array returned form the BVP solver (solution.y)
    Returns:
        The density of states for each position along x
    '''
    D_array = np.zeros(len(x))
    for i in range(len(x)):
        v = y[:,i]
        gamma, gamma_tilde, omega, omega_tilde = transform_32comp_vector_to_matrices(v)
        g = green_function(gamma, gamma_tilde)
        D = density_of_states(g)
        D_array[i] = D

    return D_array

'''
# Plotting
fig2h, axs2h = plt.subplots(3, 1, sharex='all', sharey='all')
for i, sol in enumerate(sol_list):
    x, y = sol
    D = from_solution_to_density_of_states(x, y)
    axs2h[i].plot(x, D, label = f'$\\epsilon={i}$', linewidth = 1.3)
    axs2h[i].legend(fontsize = 12)
    axs2h[i].grid(axis = 'both')

axs2h[2].set_xlabel('$x/l$', size = 15)
axs2h[1].set_ylabel('$\\frac{D}{D_0}$', size = 18, rotation = 'horizontal', labelpad = 25)
fig2h.suptitle('The normalized density of states as a function of position')
plt.show()
'''

# Exercise 2i
def bc_residuals_superconductors(v_left, v_right, phi_L, phi_R, epsilon, l):
    t_plus, t_minus = np.atanh(1/(epsilon+0.01j)), np.atanh(-1/(epsilon+0.01j))
    s_plus, s_minus = np.sinh(t_plus), np.sinh(t_minus)
    c_plus, c_minus = np.cosh(t_plus), np.cosh(t_minus)

    a = s_plus/(1+c_plus)
    b = s_minus/(1+c_minus)

    gamma_L = np.array([[0,a],[b,0]]) * np.exp(phi_L*1j)
    gamma_tilde_L = np.array([[0,b],[a,0]]) * np.exp(-phi_L*1j)

    gamma_R = np.array([[0,a],[b,0]]) * np.exp(phi_R*1j)
    gamma_tilde_R = np.array([[0,b],[a,0]]) * np.exp(-phi_R*1j)

    # Compute the residuals at the boundaries
    res = boundary_conditions(v_left, v_right, gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R, l)

    return res

# Exercise 2j
'''
m = 101
epsilon = 2
x = np.linspace(0,1,m)
y = np.zeros((32,m))
l = 1
phi_L, phi_R = 0, 0

solution = solve_bvp(lambda x,vec: h(x,vec,epsilon), lambda v_left, v_right: bc_residuals_superconductors(v_left, v_right, phi_L, phi_R, epsilon, l), x, y)
x_sol, y_sol = solution.x, solution.y

D = from_solution_to_density_of_states(x_sol, y_sol)
plt.plot(x_sol, D, label = f'$\\epsilon={epsilon}$', linewidth = 1.3)
plt.legend(fontsize = 12)
plt.grid(axis = 'both')

plt.xlabel('$x/l$', size = 15)
plt.ylabel('$\\frac{D}{D_0}$', size = 18, rotation = 'horizontal', labelpad = 25)
plt.title('The normalized density of states as a function of position')
plt.show()
'''

# Exercise 2k
'''
m = 101
epsilons = np.linspace(0, 2, 101)
l = [0.5, 1, 1.5]

phi_L, phi_R = 0, 0

solution_dict_2k = {} # Dictionary for storing solutions
for L in l:
    x = np.linspace(0, L, m)
    y = np.zeros((32, m))

    for j in tqdm.trange(len(epsilons)):
        epsilon = np.flip(epsilons)[j] # Flip the array to start at epsilon=2
        solution = solve_bvp(lambda x,vec: h(x,vec,epsilon), lambda v_left, v_right: bc_residuals_superconductors(v_left, v_right, phi_L, phi_R, epsilon, L), x, y)
        x_sol, y_sol = solution.x, solution.y
        x, y = x_sol, y_sol # Use the solution as the initial guess for the next epsilon

        # Store the solution
        label = f'(l, epsilon) = ({L}, {epsilon})'
        solution_dict_2k[label] = (x_sol, y_sol)

# Plotting
for L in l:
    D_energy = np.zeros((len(epsilons)))
    for j in range(len(epsilons)):
        epsilon = epsilons[j]
        label = f'(l, epsilon) = ({L}, {epsilon})'

        x_sol, y_sol = solution_dict_2k[label]

        # Find the density of states in the middle of the normal metal
        D_energy[j] = from_solution_to_density_of_states(x_sol, y_sol)[np.argmin(np.abs(x_sol - L/2))]


    plt.plot(epsilons, D_energy, label = f'$l={L}$')

plt.legend(fontsize = 12)
plt.grid('both')
plt.xlabel('$\\epsilon$', size = 12)
plt.ylabel('$\\frac{D}{D_0}$', size = 16, rotation = 'horizontal', labelpad = 10)
plt.title('The normalized density of states as a function of energy')
plt.show()
'''

# OLD 2k
'''
m = 101
epsilons = np.linspace(0, 2, 101)
l = [0.5, 1, 2]
phi_L, phi_R = 0, 0

for L in l:
    x = np.linspace(0, L, m)
    y = np.zeros((32, m))

    D_energy = np.zeros((len(epsilons)))

    for j in tqdm.trange(len(epsilons)):
        epsilon = epsilons[j]
        solution = solve_bvp(lambda x,vec: h(x,vec,epsilon), lambda v_left, v_right: bc_residuals_superconductors(v_left, v_right, phi_L, phi_R, epsilon, L), x, y)
        x_sol, y_sol = solution.x, solution.y
        D_energy[j] = from_solution_to_density_of_states(x_sol, y_sol)[np.argmin(np.abs(x_sol - L/2))] # Take the density of states at the middle of the normal metal
        y = y_sol # Use the solution as the initial guess for the next epsilon

    plt.plot(epsilons, D_energy, label = f'$l={L}$', linewidth = 1.3)

plt.legend(fontsize = 10)
plt.grid(axis = 'both')
plt.xlabel('$\\epsilon$', size = 12)
plt.ylabel('$\\frac{D}{D_0}$', size = 15, rotation = 'horizontal', labelpad = 10)
plt.title('The normalized density of states as a function of energy')
plt.show()
'''

# Exercise 2l
def differentiate_Ns(gamma, gamma_tilde, omega, omega_tilde):
    '''
    Calculates the matrices N and N_tilde differentiated wrt. x.
    '''

    N, N_tilde = calculate_Ns(gamma, gamma_tilde)

    dN = N @ ((omega @ gamma_tilde) + (gamma @ omega_tilde)) @ N
    dN_tilde = N_tilde @ ((omega_tilde @ gamma) + (gamma_tilde @ omega)) @ N_tilde

    return dN, dN_tilde

def differentiate_green_function(gamma, gamma_tilde, omega, omega_tilde):
    '''
    Calculates the green function differentiated wrt. x.
    '''
    N, N_tilde = calculate_Ns(gamma, gamma_tilde)
    dN, dN_tilde = differentiate_Ns(gamma, gamma_tilde, omega, omega_tilde)

    dg11 = dN
    dg12 = (N @ omega) + (dN @ gamma)
    dg21 = (-N_tilde @ omega_tilde) - (dN_tilde @ gamma_tilde)
    dg22 = -dN_tilde

    dg = 2 * np.block([[dg11, dg12],[dg21, dg22]])

    return dg

def current_integrand(gamma, gamma_tilde, omega, omega_tilde):
    N, N_tilde = calculate_Ns(gamma, gamma_tilde)
    dN, dN_tilde = differentiate_Ns(gamma, gamma_tilde, omega, omega_tilde)

    g = green_function(gamma, gamma_tilde)
    dg = differentiate_green_function(gamma, gamma_tilde, omega, omega_tilde)

    rho_hat = np.diag([1,1,-1,-1])

    product = rho_hat @ ((g @ dg) - (dg @ g))
    trace = np.trace(product)

    j = np.real(trace)

    return j

def from_solution_to_current_integrand(x:np.ndarray, y:np.ndarray)->np.ndarray:
    '''
    Finds the current integrand as a function of position x, given a solution (x,y) from the BVP solver.
    Parameters:
        x: The x-array returned form the BVP solver (solution.x)
        y: The y-array returned form the BVP solver (solution.y)
    Returns:
        The current integrand for each position along x
    '''

    j_array = np.zeros(len(x))
    for i in range(len(x)):
        v = y[:,i]
        gamma, gamma_tilde, omega, omega_tilde = transform_32comp_vector_to_matrices(v)
        j = current_integrand(gamma, gamma_tilde, omega, omega_tilde)
        j_array[i] = j

    return j_array

'''
epsilon_list = [2.0, 1.5, 1, 0.5, 0.0]
epsilons = np.linspace(0, 2, 51) # The epsilon values for which we have stored solutions, 
                                 # corresponds to the array in exercise 2k

for eps in epsilon_list:
    # Find the element in epsilons closest to the element eps
    index = np.argmin(np.abs(epsilons-eps))
    epsilon = epsilons[index]

    label = f'(l, epsilon) = ({1}, {epsilon})'
    x, y = solution_dict_2k[label]
    j = from_solution_to_current_integrand(x, y)

    plt.plot(x, j, label = f'$\\epsilon={epsilon}$', linewidth = 1.2)

plt.legend(fontsize = 12)
plt.grid('both')
plt.xlabel('$\\epsilon$', size = 12)
plt.ylabel('$j$', size = 12, rotation = 'horizontal', labelpad = 8)
plt.title('The current integrand as a function of position')
plt.show()
'''

# Exercise 2m

m = 101
epsilons = np.linspace(0, 2, 101)
l = 1
phi_L, phi_R = 1, 0
x = np.linspace(0, l, m)
y = np.zeros((32, m))

solution_dict_2m = {}
for i in tqdm.trange(len(epsilons)):
    epsilon = np.flip(epsilons)[i] # Flip the array to start at epsilon=2
    solution = solve_bvp(lambda x,vec: h(x,vec,epsilon), lambda v_left, v_right: bc_residuals_superconductors(v_left, v_right, phi_L, phi_R, epsilon, L), x, y)
    x_sol, y_sol = solution.x, solution.y
    x, y = x_sol, y_sol # Use the solution as the initial guess for the next epsilon

    # Store the solution
    label = f'(l, epsilon) = ({l}, {epsilon})'
    solution_dict_2m[label] = (x_sol, y_sol)

# Plotting
j_array = np.zeros((len(epsilons)))
for i, epsilon in enumerate(epsilons):
    label = f'(l, epsilon)=({l}, {epsilon})'
    x, y = solution_dict_2m[label]

    # Take the solution in the middle of the normal metal
    j_array[i] = from_solution_to_current_integrand(x, y)[np.argmin(np.abs(x - l/2))]

plt.plot(epsilons, j_array)
plt.grid(axis = 'both')
plt.xlabel('$\\epsilon$', size = 12)
plt.ylabel('$j$', size = 12, rotation = 'horizontal', labelpad = 10)
plt.title('The current integrand as a function of energy')
plt.show()

'''
j_array = np.zeros((len(epsilons)))
for i in tqdm.trange(len(epsilons)):
    epsilon = np.flip(epsilons)[i] # Flip the array to start at epsilon=2
    solution = solve_bvp(lambda x,vec: h(x,vec,epsilon), lambda v_left, v_right: bc_residuals_superconductors(v_left, v_right, phi_L, phi_R, epsilon, l), x, y)
    x_sol, y_sol = solution.x, solution.y
    j_array[i] = from_solution_to_current_integrand(x_sol, y_sol)[np.argmin(np.abs(x_sol - l/2))] # Take the solution in the middle of the normal metal
    x, y = x_sol, y_sol # Use the solution as the initial guess for the next epsilon


plt.plot(epsilons, np.flip(j_array))
plt.grid(axis = 'both')
plt.xlabel('$\\epsilon$', size = 12)
plt.ylabel('$j$', size = 12, rotation = 'horizontal', labelpad = 10)
plt.title('The current integrand as a function of energy')
plt.show()
'''