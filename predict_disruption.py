import os, h5py

from covid19_resonances.neural_models.neural_models import MemN2N_model
from covid19_resonances.preprocessing.data_generator import DataGenerator

CDIR = os.path.dirname(os.path.realpath(__file__))
DISRUPTION_DATA = os.path.join(*[CDIR, r'data/toy_data.h5'])

#def disruption_predictor_model():

data_generator = DataGenerator(h5path=DISRUPTION_DATA)


model = MemN2N_model()

model.fit(data_generator)
