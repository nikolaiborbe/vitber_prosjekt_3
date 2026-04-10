import numpy as np


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
    Transforming a 32-component real vector v back into four unknown 2x2 complex matrices.
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
def boundary_conditions(v_left, v_right, gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R):
    '''
    The boundary conditions for the system.
    '''
    gamma_0, gamma_tilde_0, omega_0, omega_tilde_0 = transform_32comp_vector_to_matrices(v_left)
    gamma_1, gamma_tilde_1, omega_1, omega_tilde_1 = transform_32comp_vector_to_matrices(v_right)

    # The identity matrix
    I = np.identity(2)

    # Find N and N_tilde for each interface metal L and R
    N_L_inv = I - np.matmul(gamma_L, gamma_tilde_L)
    N_L = np.linalg.inv(N_L_inv)

    N_tilde_L_inv = I - np.matmul(gamma_tilde_L, gamma_L)
    N_tilde_L = np.linalg.inv(N_tilde_L_inv)

    N_R_inv = I - np.matmul(gamma_R, gamma_tilde_R)
    N_R = np.linalg.inv(N_R_inv)

    N_tilde_R_inv = I - np.matmul(gamma_tilde_R, gamma_R)
    N_tilde_R = np.linalg.inv(N_tilde_R_inv)

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
    bc1 = omega_0 + (1/3)*np.matmul(np.matmul(M1, N_L), M2)
    bc2 = omega_tilde_0 + (1/3)*np.matmul(np.matmul(M3, N_tilde_L), M4)
    bc3 = omega_1 - (1/3)*np.matmul(np.matmul(M5, N_R), M6)
    bc4 = omega_tilde_1 - (1/3)*np.matmul(np.matmul(M7, N_tilde_R), M8)

    # Vectorize
    res = transform_matrices_to_32comp_vector(omega_0, omega_tilde_0, omega_1, omega_tilde_1)

    return res

def bc_residuals_normal_metal(v_left, v_right):
    # In this case the ricatti matrices for the interface metals are all zero
    gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))

    # Compute the residuals at the boundaries
    res = boundary_conditions(v_left, v_right, gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R)

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
from scipy.integrate import solve_bvp

m = 101
x = np.linspace(0, 1, m)
y = np.zeros((32, m))

# Use lambda to remove epsilon as a parameter, such that h works along with the BVP solver
solution = solve_bvp(lambda x, vec: h(x, vec, epsilon = -1), bc_residuals_normal_metal, x, y)
sol_x, sol_y = solution.x, solution.y

print(sol_x.shape)
print(sol_y.shape)