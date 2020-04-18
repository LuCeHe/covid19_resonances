import h5py
import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 h5path,
                 batch_size=2):
        self.__dict__.update(h5path=h5path,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.h5path, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        return self.batch_generation()

    def on_epoch_end(self):
        self.data_file = h5py.File(self.h5path, 'r')

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        ks = self.data_file['ks'][batch_start:batch_stop, ::]
        charges = self.data_file['charges'][batch_start:batch_stop, ::]
        masses = self.data_file['masses'][batch_start:batch_stop, ::]
        y_eqs = self.data_file['y_eqs'][batch_start:batch_stop, ::]
        freqs = self.data_file['freqs'][batch_start:batch_stop, ::]

        maxi_dis = self.data_file['maxi_dis'][batch_start:batch_stop, ::]

        input_batch = (ks, charges, masses, y_eqs, freqs)
        input_batch = [np.reshape(tensor, (self.batch_size, -1, 1)) for tensor in input_batch]
        output_batch = (maxi_dis,)

        return input_batch, output_batch


if __name__ == '__main__':
    h5path = r'../data/toy_data.h5'
    data_generator = DataGenerator(h5path=h5path)

    batch = data_generator.__getitem__()
    print(len(batch[0]), len(batch[1]))