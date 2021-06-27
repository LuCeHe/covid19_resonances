import os

from GenericTools.KerasTools.convenience_tools import plot_history
from covid19_resonances.neural_models.neural_models import MemN2N_model
from covid19_resonances.preprocessing.data_generator import DataGenerator

from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('pd', base_dir=CDIR, GPU=1)
TOY_DISRUPTION_DATA_train = os.path.join(*[CDIR, 'data', 'toy_data_train.h5'])
TOY_DISRUPTION_DATA_test = os.path.join(*[CDIR, 'data', 'toy_data_test.h5'])
TOY_DISRUPTION_DATA_val = os.path.join(*[CDIR, 'data', 'toy_data_val.h5'])


@ex.config
def cfg():
    epochs = 30


@ex.automain
def main(epochs):
    config_dir = os.path.join(*[CDIR, ex.observers[0].basedir, '1'])
    images_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'images'])

    train_generator = DataGenerator(h5path=TOY_DISRUPTION_DATA_train, batch_size=1)
    val_generator = DataGenerator(h5path=TOY_DISRUPTION_DATA_test, batch_size=1)
    model = MemN2N_model(mem_dim=2, num_hops=1)

    # model.summary()
    history = model.fit(train_generator, epochs=epochs)
    plot_filename = os.path.join(*[images_dir, 'history.png'])
    plot_history(history, plot_filename, epochs)

    test_generator = DataGenerator(h5path=TOY_DISRUPTION_DATA_val, batch_size=1)
    evaluation = model.evaluate(test_generator)
    prediction = model.predict(test_generator)

    print(evaluation)
    print(prediction)