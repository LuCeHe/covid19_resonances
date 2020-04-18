import os, json
import numpy as np

CDIR = os.path.dirname(os.path.realpath(__file__))

freqs = np.linspace(1, 200, 10).round(2).tolist()
# loop for different freqs
for freq in freqs:
    print('')
    os.system('python naive_ode_integration.py with freq_beam={}'.format(freq))
    content = [file for file in os.listdir('experiments') if 'experiment' in file]
    print(content[-1])

    results = os.path.join(*[CDIR, 'experiments', content[-1], '1', 'run.json'])
    print(results)
    # get result from run.json
    with open(results) as json_file:
        data = json.load(json_file)
        print(data['result'])
        #for k in data.keys():


    # collect info for dataset
    # save protein constants