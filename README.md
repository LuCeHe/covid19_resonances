# covid19_resonances


I've defined a system of ODEs to model covid's proteins, since one can find online the .pbd of the protein (and of all its 
other proteins), and from there one can extract the springs that model the interactions of its atoms. The idea is to find 
to which freq in the X-ray range it moves the most, maybe it's not necessary to have a resonance, and just an excitation 
that is high enough to deform the protein in such a way that it stops being functional. What is convenient of X-rays is 
that (1) hospitals have already the machines, and (2) the body is mostly transparent to it, apart from bones, so you can 
access all of its interior from the exterior. Similar reasoning would go for ultrasound in the 10-100 MHz, and this same code could be applied: X-rays would multiply a sinusoid by the charge of each atom, ultrasound would just need a sinusoid as the external force.

1. scipy.integrate.odeint won't be enough, since it is extremely slow for a system of 7500 
equations (the spike has 2500 atoms, each with 3 space coords). I'm using Deep Learning to approximate the system dynamic
and there's several possibilities: [here](https://github.com/frankhan91/DeepBSDE/blob/master/main.py), [here](https://github.com/lululxvi/deepxde) and [here](https://github.com/analysiscenter/pydens). 
2. The way I model the X-ray wave is by making each atom oscillate with the freq in X-rays, and with the spatial difference given by the speed of sound within the body. It might be dumbish given that I was not planning to take in consideration diffractions of the photons and other complex interactions of photons with the protein.
3. Once we can simulate large networks of springs using DL, we can infer the maximal disruption given an incident frequency. We can use that to train another DL network to predict what's the maximal disruption frequency given the constants of the proteins.
4. This might be irrelevant in the short term, but there might be an optimal waveform to disrupt a specific protein, but I doubt one can get sophisticated waveforms with current X-rays machines. I was planning on doing a Gaussian Process search.
5. Any feedback is very appreciated + if anybody wants to jump in, it would be very cool. Hopefully I can get some theoretical predictions by the end of the week. If so, it would be cool to publish it.

## TODO
- [x] visualize pdb in 3d
- [x] pdb to pqr for [charge visualization](https://www.youtube.com/watch?v=DP4yk0_A828)
- [x] protein to springs, spring constants and atom masses, using Aisotropic Network Models: [here](https://pymolwiki.org/index.php/PyANM) and [here](http://prody.csb.pitt.edu/)
- [x] small toy dynamical system to use before I manage to optimize the code for very high dimensional dynamical systems needed for proteins
- [x] automatically download the .pdb
- [x] successfully train a DL to approximate the system of equations with and without X-Ray model
- [x] define a metric of maximal disruption: the incident wave would be maximally disruptive if it maximizes the variance around the equilibrium position
- [ ] search for the frequency in the X-Ray that maximally disrupts the system

## How to Use it
1. create a `data` folder and a `cristal_structure` folder inside
2. run `get_data.py` to download a covid19 protein in that folder
3. run `naive_ode_integration.py` for `pdb_name = 'toy'` to have an impression of the final result for a random network. This code will be prohibitively slow to try with a true `pdb_name`
4. run `tf_ode_approximation` to approximate the dynamical system with DL. Under construction.
