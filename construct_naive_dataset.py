"""

from data generated from simulations of random nets of springs, construct a dataset that will be used to
predict the disruption of a protein given it's constants and the frequency of EM applied

"""


import os, json, h5py
import numpy as np

CDIR = os.path.dirname(os.path.realpath(__file__))

initial_time_string = 'experiment-2020-04-18-at-11-14-14'
final_time_string = 'experiment-2020-04-18-at-11-16-46'



content = [file for file in os.listdir('experiments') if 'experiment' in file]
print(content)

content = [c for c in content if c > initial_time_string]
content = [c for c in content if c < final_time_string]
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

        protein_constants = np.load(protein_constants_path, allow_pickle=True)[()]
        #print(protein_constants)
        print(protein_constants.keys())

        # collect info for dataset
        # save protein constants

        ks.append(np.array(protein_constants['k'])[np.newaxis, ...])
        charges.append(np.array(protein_constants['charges'])[np.newaxis, ...])
        masses.append(np.array(protein_constants['masses'])[np.newaxis, ...])
        y_eqs.append(np.array(protein_constants['y_eq'])[np.newaxis, ...])
        maxi_dis.append(np.array([maximal_disruption])[np.newaxis, ...])
        freqs.append(np.array([freq])[np.newaxis, ...])
    except:
        pass

ks = np.concatenate(ks, axis=0)
charges = np.concatenate(charges, axis=0)
masses = np.concatenate(masses, axis=0)
y_eqs = np.concatenate(y_eqs, axis=0)
maxi_dis = np.concatenate(maxi_dis, axis=0)
freqs = np.concatenate(freqs, axis=0)

print(ks.shape, charges.shape, masses.shape, y_eqs.shape) #, )
print(maxi_dis.shape)
print(freqs.shape)


print('Save into a .h5...')
hf = h5py.File(r'data/toy_data.h5', 'w')

hf.create_dataset('ks', data=ks)
hf.create_dataset('charges', data=charges)
hf.create_dataset('masses', data=masses)
hf.create_dataset('y_eqs', data=y_eqs)
hf.create_dataset('maxi_dis', data=maxi_dis)
hf.create_dataset('freqs', data=freqs)
hf.close()