from qsim.tools import tools
from odeintw import odeintw
import numpy as np
__all__ = ['SchrodingerEquation']

class SchrodingerEquation(object):
    def __init__(self, hamiltonian = None):
        # Hamiltonian is a function of time
        self.hamiltonian = hamiltonian

    def run_ode_solver(self, s, t0, tf, dt = .1, func=None):
        if func is None:
            def f(s, t):
                # hbar is set to one
                return -1j*tools.commutator(self.hamiltonian(t), s)
            func = f
        # s is a density matrix
        # tf is the total simulation time
        z, infodict = odeintw(func, s, np.linspace(t0, tf, num=(tf-t0)/dt), full_output=True)
        return z

