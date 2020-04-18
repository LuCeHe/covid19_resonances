import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import shutil
from mpl_toolkits.mplot3d import Axes3D


def pdb2plot(protein_path):
    Store_all = []
    with open(protein_path) as protein:
        for lines in protein:
            if "ATOM   " in lines:
                lines = lines.split()
                # 'ATOM', '1', 'N', 'LEU', 'A', '125', '4.329', '-12.012', '2.376', '1.00', '0.00', 'N'
                Store_all.append(map(float, lines[6:9]))

    x, y, z = zip(*Store_all)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot(x, y, z, "-")
    ax.axis("off")

    plt.show()


def positions2gif(positions, timesteps, print_every=1, gifpath='movie.gif'):
    max_pos = np.max(positions)
    min_pos = np.min(positions)

    giffolderpath = 'gif'
    if os.path.isdir(giffolderpath):
        shutil.rmtree(giffolderpath, ignore_errors=True)
    os.mkdir(giffolderpath)

    filenames = []
    for i in range(0, timesteps, print_every):
        pos_t = positions[i]

        n_atoms = len(pos_t) / 3
        coordinates = np.concatenate([l[:, np.newaxis] for l in np.split(pos_t, n_atoms)], axis=1).T
        x, y, z = np.split(coordinates.T, 3)
        x, y, z = x[0], y[0], z[0]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim3d(min_pos, max_pos)
        ax.set_ylim3d(min_pos, max_pos)
        ax.set_zlim3d(min_pos, max_pos)

        ax.plot(x, y, z, '.')
        ax.grid(False)

        # ax.axis("off")
        filename = 'gif/protein_t_{}.png'.format(i)
        plt.savefig(filename)
        filenames.append(filename)

    with imageio.get_writer(gifpath, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    shutil.rmtree(giffolderpath, ignore_errors=True)


if __name__ == '__main__':
    timesteps = 10
    print_every = 2
    num_atoms = 4

    moving_dots = np.random.rand(timesteps, 2 * 3 * num_atoms)

    speeds, positions = np.split(moving_dots, 2, axis=1)

    positions2gif(positions, print_every)
