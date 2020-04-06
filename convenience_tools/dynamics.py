import numpy as np
from numba import jit

@jit
def dynamical_protein(y, t, k, masses, friction):
    x_prime, x = np.split(np.array(y), 2)

    first_derivative = x_prime
    x_repeated = np.repeat(x[:, np.newaxis], len(x), axis=1)
    diffs_x = x_repeated - x_repeated.T
    second_derivative = (- friction * x_prime - np.sum(k * diffs_x, axis=1)) / masses

    dydt = np.concatenate([second_derivative, first_derivative])
    return dydt