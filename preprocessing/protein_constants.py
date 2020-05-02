from prody import *
from pylab import *
import os

CDIR = os.path.dirname(os.path.realpath(__file__))
CDIR_, _ = os.path.split(CDIR)


def getProteinSprings(protein_path, spring_path):
    if not os.path.isfile(spring_path):
        covid_part = parsePDB(protein_path)  # .select('calpha')

        anm = ANM('covid_part ANM analysis')
        anm.buildHessian(covid_part)
        k = anm.getHessian()
        print(anm.getHessian().round(3))
        print(anm.getHessian().shape)
        np.save(spring_path, k)
    else:
        k = np.load(spring_path)
    return k


def get_content_data(data_path):
    files = os.listdir(data_path)
    print(files)

def getProteinConstants(pdb_name='toy'):
    # the reference frame is from the equilibrium position, so y_eq = 0

    if 'toy' in pdb_name:
        # be careful, odeint seems to scale quadratically with n_atoms:
        # 20 n_atoms -> 10s / 10s
        # 40 n_atoms -> 34s / 35s
        # 60 n_atoms -> 112s
        n_atoms = int(pdb_name.replace('toy', ''))

        y_0 = np.concatenate(
            [np.zeros(3 * n_atoms), np.random.rand(3 * n_atoms)])  # initial velocities followed by initial positions
        y_0 = np.concatenate(
            [np.zeros(3 * n_atoms), np.zeros(3 * n_atoms)])  # initial velocities followed by initial positions
        y_eq = np.random.rand(3 * n_atoms)
        #friction = np.random.rand(3 * n_atoms)
        friction = np.zeros(3*n_atoms)
        k = np.random.rand(3 * n_atoms, 3 * n_atoms)
        masses = np.random.rand(3 * n_atoms)
        charges = np.random.rand(3 * n_atoms)
    else:
        data_path = r'data/deepmind_prediction'
        data_path = r'data/cristal_structure'
        precomputed_path = r'data/precomputed'

        protein_path = os.path.join(*[CDIR_, data_path, pdb_name + '.pdb'])
        spring_path = os.path.join(*[CDIR_, precomputed_path, 'k_' + pdb_name + '.npy'])

        covid_part = parsePDB(protein_path)  # .select('calpha')

        # check size whole protein
        # check size only Calpha N=7500->927

        k = getProteinSprings(protein_path, spring_path)

        masses = [atom.getMasses() for atom in covid_part.iterAtoms()]
        coords = [atom.getCoords() for atom in covid_part.iterAtoms()]
        y_eq = np.concatenate(coords)
        masses = np.repeat(masses, 3)

        n_atoms = int(len(masses) / 3)
        friction = np.random.rand(3 * n_atoms)
        y_0 = np.concatenate(
            [np.zeros(3 * n_atoms), np.random.rand(3 * n_atoms)])  # initial velocities followed by initial positions

    return y_0, y_eq, masses, friction, k, charges
