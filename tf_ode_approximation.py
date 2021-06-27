from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

import deepxde.deepxde as dde
import numpy as np

from GenericTools.PlotTools.deepxde_modified import saveplot
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment
from covid19_resonances.preprocessing.dynamics import proteinODE
from covid19_resonances.preprocessing.protein_constants import getProteinConstants

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('tf covid19 resonances', base_dir=CDIR, GPU=1)


@ex.config
def cfg():
    epochs = 1  # 20000

    n_atoms = np.random.choice(range(3, 8))
    pdb_name = 'toy{}'.format(n_atoms)  # '6lu7'  # '6M03'
    stoptime = 20
    temporal_resolution = 2000  # 200
    print_every = int(temporal_resolution / 20)

    # beam properties
    freq_beam_1 = 2000 * np.random.rand()
    freq_beam_2 = 2000 * np.random.rand()
    freq_beam_2 = np.random.choice([freq_beam_2, freq_beam_1])

    dir_beam = np.random.rand(3)
    amplitude_1 = np.array([1, 2, 2])
    amplitude_2 = np.cross(dir_beam, amplitude_1)
    amplitude_2 = amplitude_2 / np.linalg.norm(amplitude_2) * np.linalg.norm(amplitude_1) * .2

    amplitude_1 = np.repeat(amplitude_1[:, np.newaxis], n_atoms, axis=1).T.flatten()
    amplitude_2 = np.repeat(amplitude_2[:, np.newaxis], n_atoms, axis=1).T.flatten()
    amplitude_2 = np.random.choice([0, amplitude_1, amplitude_2])

    seed = np.random.choice(2000)


@ex.automain
def main(epochs, n_atoms, pdb_name, stoptime, temporal_resolution, print_every, freq_beam_1, freq_beam_2, dir_beam,
         amplitude_1, amplitude_2, seed, _log):
    y_0, y_eq, masses, friction, k, charges = getProteinConstants(pdb_name=pdb_name)

    n_variables = 3 * n_atoms
    print('number atoms: {}, number variables: {}'.format(n_atoms, 3 * n_atoms))

    ode_system = proteinODE(y_0, y_eq, masses, friction, k, freq_beam_1, freq_beam_2, dir_beam, charges, amplitude_1,
                            amplitude_2)

    """
    def boundary(_, on_initial):
        return on_initial

    geom = dde.geometry.TimeDomain(0, 10)
    ic_list = [dde.IC(geom, lambda X: np.zeros(X.shape), boundary, component=i) for i in range(n_variables)]

    # ic2 = dde.IC(geom, np.cos, boundary, component=1)
    data = dde.data.PDE(
        geometry=geom,
        pde=ode_system,
        bcs=ic_list,
        num_domain=10,
        num_boundary=10,
        train_distribution='random',
        solution=None,
        num_test=None)
    """

    #geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 10)
    geomtime = timedomain #dde.geometry.GeometryXTime(geom, timedomain)

    bc = [dde.DirichletBC(geom=geomtime,
                          func=lambda x: np.zeros((x.shape[0],1)),
                          on_boundary=lambda _, on_boundary: on_boundary,
                          component=i) for i in range(2*n_variables)]
    ic = [dde.IC(geom=geomtime,
                 func=lambda x: np.zeros(x.shape),
                 on_initial=lambda _, on_initial: on_initial,
                 component=i + n_variables) for i in range(2*n_variables)]

    data = dde.data.TimePDE(
        geometryxtime=geomtime, pde=ode_system, ic_bcs=ic+bc, num_domain=300, num_boundary=80, num_initial=160
    )

    layer_size = [1] + [50] * 3 + [2 * n_variables]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=epochs)

    plotspath = os.path.join(*[CDIR, ex.observers[0].basedir, 'images'])
    outputspath = os.path.join(*[CDIR, ex.observers[0].basedir, 'other_outputs'])
    saveplot(losshistory, train_state, issave=True, isplot=True, plotspath=plotspath, outputspath=outputspath)
