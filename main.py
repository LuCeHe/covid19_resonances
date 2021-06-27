import os
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment
from ariel_tests.interpolations import timeStructured

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('main', base_dir=CDIR, GPU=1, seed=None)
TOY_DISRUPTION_DATA = os.path.join(*[CDIR, 'data', 'toy_data_train.h5'])

@ex.automain
def main():

    if not os.path.isfile(TOY_DISRUPTION_DATA):
        initial_time = 'experiment-' + timeStructured()
        os.system('python run_many_naive.py')
        final_time = 'experiment-' + timeStructured()
        os.system('python construct_naive_dataset.py with initial_time={} final_time={}'.format(initial_time, final_time))

    os.system('python run_many_naive.py')




