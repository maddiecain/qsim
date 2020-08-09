# QSim

QSim is an optimized Python package for simulating noisy quantum computers, with a focus on quantum graph algorithms
and error correcting codes. It's still in development, but the hope is that future versions will be stable.
The following simulation methods are supported:
- Lindblad master equation solver,
- Schrodinger equation integrator, which supports time-dependent and time-independent Hamiltonians,
- Stochastic wavefunction Monte Carlo solver, and
- Quantum channel evolution, including noisy channels.

QSim supports simulating and performing outer loop optimization for the 
Quantum Approximate Optimization Algorithm (QAOA), and simulating the quantum adiabatic algorithm with a flexible 
annealing schedule. Analytic gradients
can be calculated under noisy channel evolution and noiseless evolution.
Maximum independent set (MIS; includes Rydberg MIS) and MaxCut Hamiltonians can be generated from
networkx Graph objects. Classical Monte Carlo and simulated annealing scripts are also included for comparison.

QSim also includes functionality for 
simulating the following error-correcting and error-detecting codes:
- Two qubit phase flip error-detecting code,
- Three qubit bit flip error-correcting code (as well as an error-transparent variant), and
- Jordan-Farhi-Shor four qubit error-detecting code.

## Installation

Clone the repository then run the setup script to install qsim.

```bash
git clone git@github.com:maddiecain/qsim.git
python setup.py install
```

The following packages are required to run QSim:
- networkx
- matplotlib
- numpy
- scipy
- odeintw

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
All pull requests must include proper Sphinx documentation and updated unit tests.

## License
[MIT](https://choosealicense.com/licenses/mit/)
