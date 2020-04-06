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
    pdb_name = 'toy'  # '6lu7'  #
    stoptime = 20
    numpoints = 200
    print_every = int(numpoints / 20)


@ex.automain
def main(pdb_name, stoptime, numpoints, print_every, _log):
    sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir])

    _log.warn('Getting the Data...')
    y_0, y_eq, masses, friction, k = getData(pdb_name=pdb_name)

    _log.warn('Solving the Dynamical Equations...')
    # dynamic and save
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

    moving_dots = odeint(dynamical_protein, y_0, t, args=(k, masses, friction))

    np.save(sacred_dir + r'/dynamics', moving_dots)

    _log.warn('Plotting...')
    # plot and git of dynamic
    speeds, positions = np.split(moving_dots, 2, axis=1)
    plt.plot(t, positions.round(3))
    plt.savefig(sacred_dir + r'/two_springs.png', dpi=100)
    positions = positions + y_eq
    gifpath = sacred_dir + r'/movie.gif'
    positions2gif(positions=positions,
                  timesteps=numpoints,
                  print_every=print_every,
                  gifpath=gifpath)
