import os

from GenericTools.KerasTools.convenience_tools import plot_history
from covid19_resonances.neural_models.neural_models import MemN2N_model
from covid19_resonances.preprocessing.data_generator import DataGenerator

CDIR = os.path.dirname(os.path.realpath(__file__))
DISRUPTION_DATA = os.path.join(*[CDIR, r'data/toy_data.h5'])

epochs = 30

data_generator = DataGenerator(h5path=DISRUPTION_DATA, batch_size=1)


model = MemN2N_model(mem_dim=2, num_hops=1)

#model.summary()

history = model.fit(data_generator, epochs=epochs)
plot_filename = os.path.join(*[CDIR, r'experiments/history.png'])
plot_history(history, plot_filename, epochs)
