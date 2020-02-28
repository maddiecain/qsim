from scipy.integrate import ode
from qsim.tools.operations import *
from qsim.tools import tools
from odeintw import odeintw
import numpy as np
__all__ = ['MasterEquation']

class MasterEquation(object):
    def __init__(self, hamiltonian, noise_model):
        # Noise is a list of LindbladNoise objects
        # Hamiltonian is a function of time
        self.hamiltonian = hamiltonian
        self.noise_model = noise_model


    def run_ode_solver(self, s, t0, tf, dt = .1):
        def f(s, t):
            # hbar is set to one
            return -1j*tools.commutator(self.hamiltonian(t), s)+self.noise_model.all_qubit_liouvillian(s)
        # s is a density matrix
        # tf is the total simulation time
        z, infodict = odeintw(f, s, np.linspace(t0, tf, num=(tf-t0)/dt), full_output=True)
        return z

    def run_basic_integrator(self, s, t0, tf, dt = 1):
        rho_dot()
