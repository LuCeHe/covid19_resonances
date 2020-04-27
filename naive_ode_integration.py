"""
1. charge location
https://www.youtube.com/watch?v=DP4yk0_A828
http://nbcr-222.ucsd.edu/pdb2pqr_2.0.0/:
his server enables a user to convert PDB files into PQR files. PQR
files are PDB files where the occupancy and B-factor columns have
been replaced by per-atom charge and radius.
2. get the expected springs
"""

import os
from pylab import *
from scipy.integrate import odeint

from GenericTools.SacredTools.VeryCustomSacred import CustomExperiment
from covid19_resonances.preprocessing.dynamics import dynamical_protein
from covid19_resonances.preprocessing.protein_constants import getProteinConstants
from covid19_resonances.postprocessing.visualization import positions2gif

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('noi', base_dir=CDIR, GPU=1)


@ex.config
def cfg():
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
def main(pdb_name, stoptime, temporal_resolution, print_every, dir_beam, freq_beam_1, freq_beam_2, amplitude_1,
         amplitude_2, _log):
    sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'images'])
    files_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'other_outputs'])

    _log.warn('Getting the Data...')
    y_0, y_eq, masses, friction, k, charges = getProteinConstants(pdb_name=pdb_name)
    protein_constants = {'y_0': y_0, 'y_eq': y_eq, 'masses': masses, 'friction': friction, 'k': k,
                         'charges': charges}
    beam_constants = {'dir_beam': dir_beam, 'freq_beam_1': freq_beam_1, 'freq_beam_2': freq_beam_2,
                      'amplitude_1': amplitude_1[:3], 'amplitude_2': amplitude_2[:3]}
    np.save(files_dir + r'/protein_constants', protein_constants)
    np.save(files_dir + r'/beam_constants', beam_constants)

    _log.warn('Solving the Dynamical Equations...')
    t = [stoptime * float(i) / (temporal_resolution - 1) for i in range(temporal_resolution)]
    moving_dots = odeint(dynamical_protein, y_0, t,
                         args=(k, masses, friction, y_eq, charges, amplitude_1, amplitude_2, dir_beam, freq_beam_1, freq_beam_2))
    np.save(files_dir + r'/dynamics', moving_dots)

    _log.warn('Plotting...')
    speeds, positions = np.split(moving_dots, 2, axis=1)
    max_disruption = np.max(np.std(positions, axis=1))  # np.mean(np.std(positions, axis=0))
    plt.plot(t, positions.round(3))
    plt.savefig(images_dir + r'/two_springs_{}.png'.format(max_disruption.round(3)), dpi=100)
    positions = positions + y_eq
    gifpath = images_dir + r'/movie_md_{}.gif'.format(max_disruption.round(3))
    positions2gif(positions=positions,
                  timesteps=temporal_resolution,
                  print_every=print_every,
                  gifpath=gifpath)

    return max_disruption
