"""

from data generated from simulations of random nets of springs, construct a dataset that will be used to
predict the disruption of a protein given it's constants and the frequency of EM applied

"""

import os, json, h5py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from GenericTools.SacredTools.VeryCustomSacred import CustomExperiment

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('cnd', base_dir=CDIR, GPU=1)


@ex.config
def cfg():
    initial_time = 'experiment-2020-04-18-at-22-03-14'
    final_time = 'experiment-2020-04-18-at-22-16-46'
    train_split = .80
    val_split = .05


@ex.automain
def main(initial_time, final_time, train_split, val_split):
    content = [file for file in os.listdir('experiments') if 'experiment' in file]
    content = [file for file in content if '-noi_' in file]
    # content = [file for file in content if not '-cnd_' in file]
    print(content)

    content = [c for c in content if c > initial_time]
    content = [c for c in content if c < final_time]
    print(content)

    ks = []
    charges = []
    masses = []
    y_eqs = []
    maxi_dis = []
    freqs = []
    for file in content:
        print('')
        maximal_disruption_path = os.path.join(*[CDIR, 'experiments', file, '1', 'run.json'])
        freq_path = os.path.join(*[CDIR, 'experiments', file, '1', 'config.json'])
        protein_constants_path = os.path.join(*[CDIR, 'experiments', file, 'other_outputs', 'protein_constants.npy'])
        beam_constants_path = os.path.join(*[CDIR, 'experiments', file, 'other_outputs', 'beam_constants.npy'])

        try:
            # get maximal_disruption
            with open(maximal_disruption_path) as json_file:
                data = json.load(json_file)
                maximal_disruption = data['result']['value']
                print(maximal_disruption)

            # get freq
            with open(freq_path) as json_file:
                data = json.load(json_file)
                freq = data['freq_beam']
                print(freq)

            # save protein constants
            protein_constants = np.load(protein_constants_path, allow_pickle=True)[()]
            print(protein_constants.keys())
            ks.append(np.array(protein_constants['k'])[np.newaxis, ...])
            charges.append(np.array(protein_constants['charges'])[np.newaxis, ...])
            masses.append(np.array(protein_constants['masses'])[np.newaxis, ...])
            y_eqs.append(np.array(protein_constants['y_eq'])[np.newaxis, ...])
            maxi_dis.append(np.array([maximal_disruption])[np.newaxis, ...])
            freqs.append(np.array([freq])[np.newaxis, ...])

            # save beam constants
            beam_constants = np.load(beam_constants_path, allow_pickle=True)[()]
            print(beam_constants.keys())
            dir_beam.append(np.array(beam_constants['dir_beam'])[np.newaxis, ...])
            freq_beam_1.append(np.array(beam_constants['freq_beam_1'])[np.newaxis, ...])
            freq_beam_2.append(np.array(beam_constants['freq_beam_2'])[np.newaxis, ...])
            ks.append(np.array(beam_constants['k'])[np.newaxis, ...])
            ks.append(np.array(beam_constants['k'])[np.newaxis, ...])

            'dir_beam': dir_beam, 'freq_beam_1': freq_beam_1, 'freq_beam_2': freq_beam_2,
            'amplitude_1': amplitude_1, 'amplitude_2': amplitude_2}
        except:
            pass

    ks = pad_sequences(ks, padding='pre', truncating='pre', value=-1)
    charges = pad_sequences(charges, padding='pre', truncating='pre', value=-1)
    masses = pad_sequences(masses, padding='pre', truncating='pre', value=-1)
    y_eqs = pad_sequences(y_eqs, padding='pre', truncating='pre', value=-1)

    maxi_dis = np.concatenate(maxi_dis, axis=0)
    freqs = np.concatenate(freqs, axis=0)

    print(ks.shape, charges.shape, masses.shape, y_eqs.shape)  # , )
    print(maxi_dis.shape)
    print(freqs.shape)

    n_samples = freqs.shape[0]
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)

    if n_val == 0:
        n_val = 1
        n_train = n_samples - 2

    for set, limits in zip(['train', 'test', 'val'], [[0, n_train], [n_train, -n_val], [-n_val, None]]):
        print('Save {} into a .h5...'.format(set))
        TOY_DISRUPTION_DATA = os.path.join(*[CDIR, 'data', 'toy_data_{}.h5'.format(set)])
        hf = h5py.File(TOY_DISRUPTION_DATA, 'w')

        hf.create_dataset('ks', data=ks[limits[0]:limits[1]])
        hf.create_dataset('charges', data=charges[limits[0]:limits[1]])
        hf.create_dataset('masses', data=masses[limits[0]:limits[1]])
        hf.create_dataset('y_eqs', data=y_eqs[limits[0]:limits[1]])
        hf.create_dataset('maxi_dis', data=maxi_dis[limits[0]:limits[1]])
        hf.create_dataset('freqs', data=freqs[limits[0]:limits[1]])
        hf.close()
