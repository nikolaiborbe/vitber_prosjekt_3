import numpy as np

def calculate_Ns(gamma, gamma_tilde):
    I = np.identity(2)

    # Find N and N_tilde
    N_inv = I - np.matmul(gamma, gamma_tilde)
    N = np.linalg.inv(N_inv)

    N_tilde_inv = I - np.matmul(gamma_tilde, gamma)
    N_tilde = np.linalg.inv(N_tilde_inv)

    return N_tilde_inv, N_tilde