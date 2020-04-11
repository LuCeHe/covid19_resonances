from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import deepxde as dde
import numpy as np
import tensorflow as tf

from GenericTools.PlotTools.deepxde_modified import saveplot
from GenericTools.SacredTools.VeryCustomSacred import CustomExperiment

tf.logging.set_verbosity(tf.logging.FATAL)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

from covid19_resonances.convenience_tools.preprocessing import getData

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('tf covid19 resonances', base_dir=CDIR, GPU=1)


@ex.config
def cfg():
    epochs = 1  # 20000
    freq_beam = 20
    dir_beam = np.random.rand(3)
    dir_beam = dir_beam / np.linalg.norm(dir_beam)


@ex.automain
def main(epochs, dir_beam, freq_beam, _log):
    y_0, y_eq, masses, friction, k, amplitude, charges = getData(pdb_name='toy')

    n_variables = len(masses)
    n_atoms = int(n_variables/3)
    print('n_variables: ', n_variables)

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
        xrays_beam = charges * amplitude * tf.sin(tf.reduce_sum(dir_beam[tf.newaxis, :, tf.newaxis] * absolute_positions, axis=1) + freq_beam * t)

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

    def boundary(_, on_initial):
        return on_initial

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x),) * 2 * n_variables)

    geom = dde.geometry.TimeDomain(0, 10)
    ic_list = [dde.IC(geom, np.sin, boundary, component=i) for i in range(n_variables)]
    # ic2 = dde.IC(geom, np.cos, boundary, component=1)
    data = dde.data.PDE(
        geom=geom,
        num_outputs=2 * n_variables,
        pde=ode_system,
        bcs=ic_list,
        num_domain=35,
        num_boundary=2 * n_variables,
        func=func,
        num_test=100)

    layer_size = [1] + [50] * 3 + [2 * n_variables]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=epochs)

    plotspath = os.path.join(*[CDIR, ex.observers[0].basedir, r'images/'])
    outputspath = os.path.join(*[CDIR, ex.observers[0].basedir, r'other_outputs/'])
    saveplot(losshistory, train_state, issave=True, isplot=True, plotspath=plotspath, outputspath=outputspath)
