from scipy.integrate import ode
from qsim.tools.operations import *
from qsim.tools import tools
from odeintw import odeintw
import numpy as np
__all__ = ['MasterEquation']

class MasterEquation(object):
    def __init__(self, hamiltonians = [], noise_models = []):
        # Noise is a list of LindbladNoise objects
        # Hamiltonian is a function of time
        self.hamiltonians = hamiltonians
        self.noise_models = noise_models


    def run_ode_solver(self, s, t0, tf, num = 2, func = None):
        if func is None:
            def f(state, t):
                # hbar is set to one
                res = np.zeros(state.shape)
                for hamiltonian in self.hamiltonians:
                    res = res + -1j*(hamiltonian.left_multiply(s, overwrite = False) - hamiltonian.right_multiply(s, overwrite = False))
                for noise_model in self.noise_models:
                    res = res + noise_model.all_qubit_liouvillian(state)
                return res

            func = f
        # s is a State object
        # tf is the total simulation time
        z, infodict = odeintw(func, s.state, np.linspace(t0, tf, num=num), full_output=True)
        return z
