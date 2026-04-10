import numpy as np


# Exercise 2 a)

# testvalue for matrix M
M = np.array([[1 + 2j, 3 + 4j],
              [6 + 9j, 7 + 8j]])


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

m = transform_matrix_to_vector(M)
print("Vector m:\n", m)


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

M_reconstructed = transform_vector_to_matrix(m)
print("Reconstructed Matrix M:\n", M_reconstructed)



# Exercise 2 b)

# testvalue for four 8-component real vectors
m1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
m2 = np.array([9, 10, 11, 12, 13, 14, 15, 16])
m3 = np.array([17, 18, 19, 20, 21, 22, 23, 24])
m4 = np.array([25, 26, 27, 28, 29, 30, 31, 32])

def transform_four_8comp_vectors_to_32comp_vector(m1, m2, m3, m4):
    """
    Transforming four 8-component real vectors into a single 32-component real vector v.
    """
    return np.concatenate((m1, m2, m3, m4))

v = transform_four_8comp_vectors_to_32comp_vector(m1, m2, m3, m4)
print("Vector v:\n", v)

def transform_32comp_vector_to_four_8comp_vectors(v):
    """
    Transforming a 32-component real vector v back into four 8-component real vectors.
    """
    m1 = v[:8]
    m2 = v[8:16]
    m3 = v[16:24]
    m4 = v[24:32]
    return m1, m2, m3, m4

m1_reconstructed, m2_reconstructed, m3_reconstructed, m4_reconstructed = transform_32comp_vector_to_four_8comp_vectors(v)
print("Reconstructed m1:", m1_reconstructed)
print("Reconstructed m2:", m2_reconstructed)
print("Reconstructed m3:", m3_reconstructed)
print("Reconstructed m4:", m4_reconstructed)


# Exercise 2 c)

# testvalue for unknown 2x2 complex matrices
gamma = np.array([[1 + 2j, 3 + 4j],
              [6 + 9j, 7 + 8j]])
gamma_tilde = np.array([[5 + 6j, 7 + 8j],
              [9 + 10j, 11 + 12j]])
omega = np.array([[13 + 14j, 15 + 16j],
              [17 + 18j, 19 + 20j]])
omega_tilde = np.array([[21 + 22j, 23 + 24j],
              [25 + 26j, 27 + 28j]])    

def transform_matrixes_to_32comp_vector(gamma, gamma_tilde, omega, omega_tilde):
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

print("Vector v:\n", transform_matrixes_to_32comp_vector(gamma, gamma_tilde, omega, omega_tilde))

def transform_32comp_vector_to_matrixes(v):
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

gamma_reconstructed, gamma_tilde_reconstructed, omega_reconstructed, omega_tilde_reconstructed = transform_32comp_vector_to_matrixes(transform_matrixes_to_32comp_vector(gamma, gamma_tilde, omega, omega_tilde))
print("Reconstructed gamma:\n", gamma_reconstructed)
print("Reconstructed gamma_tilde:\n", gamma_tilde_reconstructed)
print("Reconstructed omega:\n", omega_reconstructed)
print("Reconstructed omega_tilde:\n", omega_tilde_reconstructed)



# Exercise 2 e)

# function fun, that will be a input in solve_bvp
def fun(x, vec):
    """
    x: Vector with m components
    vec: 32 x m matrix, that conatains the vector v at each position on x

    the function returns a 32 x m matrix that contains d/dx(v) at each position on x
    """
    
