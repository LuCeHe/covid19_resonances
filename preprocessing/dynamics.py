import numpy as np
from numba import jit

def dynamical_protein(y, t, k, masses, friction, y_eq, charges, amplitude, dir_beam, freq_beam):
    x_prime, x = np.split(np.array(y), 2)

    # xray
    positions = x + y_eq
    a = np.transpose(np.repeat(positions[:, np.newaxis], 3, axis=1), [1, 0])
    xrays_beam = charges * amplitude * np.sin(np.sum(dir_beam * a.T, axis=1) + freq_beam * t)

    # net of springs
    first_derivative = x_prime
    x_repeated = np.repeat(x[:, np.newaxis], len(x), axis=1)
    diffs_x = x_repeated - x_repeated.T
    spring_term = - np.sum(k * diffs_x, axis=1)

    #newton law
    second_derivative = (- friction * x_prime + spring_term + xrays_beam) / masses

    dydt = np.concatenate([second_derivative, first_derivative])
    return dydt


@jit
def dynamical_protein_noXray(y, t, k, masses, friction):
    x_prime, x = np.split(np.array(y), 2)

    # net of springs
    first_derivative = x_prime
    x_repeated = np.repeat(x[:, np.newaxis], len(x), axis=1)
    diffs_x = x_repeated - x_repeated.T

    #newton law
    second_derivative = (- friction * x_prime - np.sum(k * diffs_x, axis=1)) / masses

    dydt = np.concatenate([second_derivative, first_derivative])
    return dydt