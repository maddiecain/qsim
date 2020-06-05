from qsim.tools import tools
from odeintw import odeintw
import numpy as np
__all__ = ['SchrodingerEquation']

class SchrodingerEquation(object):
    def __init__(self, hamiltonians = None, is_ket = False):
        # Hamiltonian is a function of time
        self.hamiltonians = hamiltonians
        self.is_ket = is_ket

    def run_ode_solver(self, s, t0, tf, dt = .1, func=None):
        if func is None:
            if not self.is_ket:
                def f(s, t):
                    # hbar is set to one
                    return -1j*tools.commutator(self.hamiltonian(t), s)
            else:
                def f(s, t):
                    res = np.zeros(s.shape)
                    for hamiltonian in self.hamiltonians:
                        res = res -1j*hamiltonian.left_multiply(s)
                    return res
            func = f
        # s is a density matrix
        # tf is the total simulation time
        z, infodict = odeintw(func, s, np.arange(t0, tf, dt), full_output=True)
        return z

    def run_mc_solver(self, s, t0, tf, dt = .1, func=None):
        pass

