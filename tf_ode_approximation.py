from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde

from covid19_resonances.convenience_tools.preprocessing import getData


def main():
    y_0, y_eq, masses, friction, k = getData(pdb_name='toy')
    def ode_system(t, xpx):
        """ODE system.
        dy1/dx = y2
        dy2/dx = -y1
        """
        x_prime, x = tf.split(xpx, 2, axis=1)

        first_derivative = x_prime
        x_repeated = tf.repeat(x[:, tf.newaxis], x.shape[1], axis=1)
        diffs_x = x_repeated - tf.transpose(x_repeated, perm=[0, 2, 1])

        diffs_x = tf.cast(diffs_x, dtype=tf.float64)
        k_diff = tf.math.multiply(k, diffs_x)
        k_diff = tf.cast(k_diff, dtype=tf.float32)
        second_derivative = (- friction * x_prime - tf.reduce_sum(k_diff, axis=1)) / masses

        dx_t = tf.gradients(x, t)[0]
        dx_tt = tf.gradients(x_prime, t)[0]
        concatenation = tf.concat([dx_t - first_derivative, dx_tt - second_derivative], axis=1)
        list_eqs = tf.split(concatenation, xpx.shape[1], axis=1)
        #print(concatenation)
        return list_eqs

    def boundary(_, on_initial):
        return on_initial

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x), np.cos(x)))

    n_variables = len(masses)
    print('n_variables: ', n_variables)
    geom = dde.geometry.TimeDomain(0, 10)
    ic_list = [dde.IC(geom, np.sin, boundary, component=i) for i in range(n_variables)]
    #ic2 = dde.IC(geom, np.cos, boundary, component=1)
    data = dde.data.PDE(geom, 2*n_variables, ode_system, ic_list, 35, 2*n_variables, func=func, num_test=100)

    layer_size = [1] + [50] * 3 + [2*n_variables]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
