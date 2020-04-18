import os
from GenericTools.SacredTools.VeryCustomSacred import CustomExperiment

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment(' covid19 resonances ', base_dir=CDIR, GPU=1)
TOY_DISRUPTION_DATA = os.path.join(*[CDIR, r'data/toy_data.h5'])

@ex.config
def cfg():
    a = 0

@ex.automain
def main():

    if not os.path.isfile(TOY_DISRUPTION_DATA):
        # day-hour
        # run_many_naive.py
        # day-hour
        # construct_naive_dataset.py
        pass

    # load TOY_DISRUPTION_DATA
    # run predict_disruption.py
    pass




