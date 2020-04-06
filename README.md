# covid19_resonances


I've defined a system of ODEs to model covid's spike, since one can find online the .pbd of the protein (and of all its 
other proteins), and from there one can extract the springs that model the interactions of its atoms. The idea is to find 
to which freq in the X-ray range it moves the most, maybe it's not necessary to have a resonance, and just an excitation 
that is high enough to deform the protein in such a way that it stops being functional. What is convenient of X-rays is 
that (1) hospitals have already the machines, and (2) the body is mostly transparent to it, apart from bones, so you can 
access all of its interior from the exterior.

1. Dumb me thought scipy.integrate.odeint would be good enough, but it is extremely slow for a system of 7500 
eqs (the spike has 2500 atoms, each with 3 space coords). I'm planning to use this DL to approximate the system dynamic
and there's several possibilities: [here](https://github.com/frankhan91/DeepBSDE/blob/master/main.py), [here](https://github.com/lululxvi/deepxde) 
and [here](https://github.com/analysiscenter/pydens). 
2. The way I was going to model the X-ray wave was just to make oscillate each atom with the freq in X-rays, and with the spatial difference given by the speed of sound within the body. It might be dumbish given that I was not planning to take in consideration diffractions of the photons etc
3. This might be irrelevant in the short term, but there might be an optimal waveform to disrupt a specific protein, but I doubt one can get sophisticated waveforms with current X-rays machines. I was planning on doing a Gaussian Process search.
4. Any feedback is very appreciated + if anybody wants to jump in, it would be very cool. Hopefully I can get some theoretical predictions by the end of the week. If so, it would be cool to publish it.

## TODO
- [x] visualize pdb
- [x] make it a system of springs
- [x] 3d drawing
- [x] pdb to pqr for [charge visualization](https://www.youtube.com/watch?v=DP4yk0_A828)
- [x] protein to springs, spring constants and atom masses, using Aisotropic Network Models: [here](https://pymolwiki.org/index.php/PyANM) and [here](http://prody.csb.pitt.edu/)
- [x] small toy dynamical system to use before I manage to optimize the code for very high dimensional dynamical systems needed for proteins
- [x] approximate dynamical system without x-rays
- [ ] model X-ray influence to the charge
- [ ] automatically download the .pdb
- [ ] define a metric of maximal disruption: the incident wave would be maximally disruptive if it maximizes the variance around the equilibrium position