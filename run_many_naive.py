import os, json
import numpy as np

CDIR = os.path.dirname(os.path.realpath(__file__))

n_experiments = 10 #100

# loop to produce different experiments
# TODO: optimize it to have it in different processes
for _ in range(n_experiments):
    print('')
    os.system('python naive_ode_integration.py')
    content = [file for file in os.listdir('experiments') if 'experiment' in file]
    print(content[-1])

    results = os.path.join(*[CDIR, 'experiments', content[-1], '1', 'run.json'])
    with open(results) as json_file:
        data = json.load(json_file)
        print(data['result'])