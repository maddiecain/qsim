from scipy.integrate import ode
from qsim.tools.operations import *
from qsim.tools import tools
from odeintw import odeintw
import numpy as np
__all__ = ['MasterEquation']

class MasterEquation(object):
    def __init__(self, hamiltonian = None, noise_model = None):
        # Noise is a list of LindbladNoise objects
        # Hamiltonian is a function of time
        self.hamiltonian = hamiltonian
        self.noise_model = noise_model


    def run_ode_solver(self, s, t0, tf, dt = .1, func = None):
        if func is None:
            def f(s, t):
                # hbar is set to one
                return -1j*tools.commutator(self.hamiltonian(t), s)+self.noise_model.all_qubit_liouvillian(s)

            func = f
        # s is a density matrix
        # tf is the total simulation time
        z, infodict = odeintw(func, s, np.linspace(t0, tf, num=(tf-t0)/dt), full_output=True)
        return z
