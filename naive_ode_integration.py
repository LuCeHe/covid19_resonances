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
from covid19_resonances.convenience_tools.dynamics import dynamical_protein
from covid19_resonances.convenience_tools.preprocessing import getData
from covid19_resonances.visualization.visualization import positions2gif

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('guinea_cleaner_CPC', base_dir=CDIR, GPU=1)


@ex.config
def cfg():
    pdb_name = 'toy5'  # '6lu7'  # '6M03'
    stoptime = 20
    temporal_resolution = 20  # 200
    print_every = int(temporal_resolution / 20)
    freq_beam = 20
    dir_beam = np.random.rand(3)


@ex.automain
def main(pdb_name, stoptime, temporal_resolution, print_every, dir_beam, freq_beam, _log):
    sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'images'])
    files_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'other_outputs'])

    _log.warn('Getting the Data...')
    y_0, y_eq, masses, friction, k, amplitude, charges = getData(pdb_name=pdb_name)
    protein_constants = {'y_0': y_0, 'y_eq': y_eq, 'masses': masses, 'friction': friction, 'k': k,
                         'amplitude': amplitude, 'charges': charges}
    np.save(files_dir + r'/protein_constants', protein_constants)

    _log.warn('Solving the Dynamical Equations...')
    t = [stoptime * float(i) / (temporal_resolution - 1) for i in range(temporal_resolution)]
    moving_dots = odeint(dynamical_protein, y_0, t,
                         args=(k, masses, friction, y_eq, charges, amplitude, dir_beam, freq_beam))
    np.save(files_dir + r'/dynamics', moving_dots)

    _log.warn('Plotting...')
    speeds, positions = np.split(moving_dots, 2, axis=1)
    max_disruption = np.mean(np.std(positions, axis=0))
    plt.plot(t, positions.round(3))
    plt.savefig(images_dir + r'/two_springs.png', dpi=100)
    positions = positions + y_eq
    gifpath = images_dir + r'/movie_md_{}.gif'.format(max_disruption.round(3))
    positions2gif(positions=positions,
                  timesteps=temporal_resolution,
                  print_every=print_every,
                  gifpath=gifpath)

    return max_disruption
