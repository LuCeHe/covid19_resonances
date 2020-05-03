import numpy as np
import tensorflow as tf
#from numba import jit

def dynamical_protein(y, t, k, masses, friction, y_eq, charges, amplitude_1, amplitude_2, dir_beam, freq_beam_1, freq_beam_2):
    x_prime, x = np.split(np.array(y), 2)

    # xray
    positions = x + y_eq
    a = np.transpose(np.repeat(positions[:, np.newaxis], 3, axis=1), [1, 0])
    xrays_beam_1 = charges * amplitude_1 * np.sin(np.sum(dir_beam * a.T, axis=1) + freq_beam_1 * t)
    xrays_beam_2 = charges * amplitude_2 * np.sin(np.sum(dir_beam * a.T, axis=1) + freq_beam_2 * t + np.pi/2)

    xrays_beam = xrays_beam_1 + xrays_beam_2
    # net of springs
    first_derivative = x_prime
    x_repeated = np.repeat(x[:, np.newaxis], len(x), axis=1)
    diffs_x = x_repeated - x_repeated.T
    spring_term = - np.sum(k * diffs_x, axis=1)

    #newton law
    second_derivative = (- friction * x_prime + spring_term + xrays_beam) / masses

    dydt = np.concatenate([second_derivative, first_derivative])
    return dydt


#@jit
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


def proteinODE_old(y_0, y_eq, masses, friction, k, freq_beam_1, freq_beam_2, dir_beam, charges, amplitude_1, amplitude_2):
    def ode_system(t, xpx):
        """ODE system.
        dy1/dx = y2
        dy2/dx = -y1
        """
        x_prime, x = tf.split(xpx, 2, axis=1)

        first_derivative = x_prime
        x_repeated = tf.repeat(x[:, tf.newaxis], x.shape[1], axis=1)
        diffs_x = x_repeated - tf.transpose(x_repeated, perm=[0, 2, 1])

        # compute absolute position to calculate the influence of the xray
        positions = x + y_eq
        a = tf.transpose(tf.repeat(positions[:, :, tf.newaxis], 3, axis=2), [0, 2, 1])
        absolute_positions = tf.concat(a, axis=1)
        kx = tf.reduce_sum(dir_beam[tf.newaxis, :, tf.newaxis] * absolute_positions, axis=1)
        xrays_beam_1 = charges * amplitude_1 * tf.sin(kx + freq_beam_1 * t)
        xrays_beam_2 = charges * amplitude_2 * tf.sin(kx + freq_beam_2 * t + tf.constant(np.pi/2))

        xrays_beam = xrays_beam_1 + xrays_beam_2
        # compute differences in distances for spring model
        diffs_x = tf.cast(diffs_x, dtype=tf.float64)
        k_diff = tf.math.multiply(k, diffs_x)
        k_diff = tf.cast(k_diff, dtype=tf.float32)

        second_derivative = (- friction * x_prime - tf.reduce_sum(k_diff, axis=1) + xrays_beam) / masses

        dx_t = tf.gradients(x, t)[0]
        dx_tt = tf.gradients(x_prime, t)[0]
        concatenation = tf.concat([dx_t - first_derivative, dx_tt - second_derivative], axis=1)
        list_eqs = tf.split(concatenation, xpx.shape[1], axis=1)
        # print(concatenation)
        return list_eqs
    return ode_system


def proteinODE(y_0, y_eq, masses, friction, k, freq_beam_1, freq_beam_2, dir_beam, charges, amplitude_1, amplitude_2):
    def ode_system(t, xpx):

        print(tf.keras.backend.int_shape(t))
        print(tf.keras.backend.int_shape(xpx))
        """ODE system.
        dy1/dx = y2
        dy2/dx = -y1
        """
        x_prime, x = tf.split(xpx, 2, axis=1)

        first_derivative = x_prime
        x_repeated = tf.repeat(x[:, tf.newaxis], x.shape[1], axis=1)
        diffs_x = x_repeated - tf.transpose(x_repeated, perm=[0, 2, 1])

        # compute absolute position to calculate the influence of the xray
        positions = x + y_eq
        a = tf.transpose(tf.repeat(positions[:, :, tf.newaxis], 3, axis=2), [0, 2, 1])
        absolute_positions = tf.concat(a, axis=1)
        kx = tf.reduce_sum(dir_beam[tf.newaxis, :, tf.newaxis] * absolute_positions, axis=1)
        xrays_beam_1 = charges * amplitude_1 * tf.sin(kx + freq_beam_1 * t)
        xrays_beam_2 = charges * amplitude_2 * tf.sin(kx + freq_beam_2 * t + tf.constant(np.pi/2))

        xrays_beam = xrays_beam_1 + xrays_beam_2
        # compute differences in distances for spring model
        diffs_x = tf.cast(diffs_x, dtype=tf.float64)
        k_diff = tf.math.multiply(k, diffs_x)
        k_diff = tf.cast(k_diff, dtype=tf.float32)

        second_derivative = (- friction * x_prime - tf.reduce_sum(k_diff, axis=1) + xrays_beam) / masses

        dx_t = tf.gradients(x, t)[0]
        dx_tt = tf.gradients(x_prime, t)[0]
        concatenation = tf.concat([dx_t - first_derivative, dx_tt - second_derivative], axis=1)
        list_eqs = tf.split(concatenation, xpx.shape[1], axis=1)
        # print(concatenation)
        return list_eqs
    return ode_system