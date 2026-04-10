import numpy as np


# Oppgave 2 a)
print("Oppgave 2a:")

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



# Oppgave 2 b)
print("Oppgave 2b:")

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



#Exercise 2d)  

import jax
from jax import grad 

def derivative_of_vector_func(v: list ,epsilon: float):
    """
    takes in vector,  and epsilon and returns 
    the derivative of v with respects to x
    """
    der = []
    for i in range(len(v)):
        der += grad(v[i](epsilon))