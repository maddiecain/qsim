from scipy.integrate import ode
from qsim.tools.operations import *
from qsim.tools import tools
from odeintw import odeintw
import numpy as np
__all__ = ['SchrodingerEquation']

class SchrodingerEquation(object):
    def __init__(self, hamiltonian):
        # Hamiltonian is a function of time
        self.hamiltonian = hamiltonian

    def run_ode_solver(self, s, t0, tf, dt = .1):
        def f(s, t):
            # hbar is set to one
            return -1j*tools.commutator(self.hamiltonian(t), s)
        # s is a density matrix
        # tf is the total simulation time
        z, infodict = odeintw(f, s, np.linspace(t0, tf, num=(tf-t0)/dt), full_output=True)
        return z

