from qsim.tools import tools
from odeintw import odeintw
import numpy as np

__all__ = ['SchrodingerEquation']


class SchrodingerEquation(object):
    def __init__(self, hamiltonians=None):
        # Hamiltonian is a function of time
        if hamiltonians is None:
            hamiltonians=[]
        self.hamiltonians = hamiltonians

    def run_time_independent_solver(self, t0, tf, dt=.1, func=None):
        """Returns the state of the system under unitary time evolution of a time
        independent Hamiltonian"""
        pass

    def run_ode_solver(self, s, t0, tf, dt=.1, func=None):
        """Numerically integrates the Schrodinger equation"""
        is_ket = tools.is_ket(s)
        if func is None:
            if not is_ket:
                print('Are you really sure you want to solve the Schrodinger equation with a density matrix input?')
                def f(s, t):
                    res = np.zeros(s.shape)
                    for hamiltonian in self.hamiltonians:
                        res = res - 1j * hamiltonian.left_multiply(s, is_ket=is_ket) + 1j * hamiltonian.right_multiply(s, is_ket=is_ket)
                    return res
            else:
                def f(s, t):
                    res = np.zeros(s.shape)
                    for hamiltonian in self.hamiltonians:
                        res = res - 1j * hamiltonian.left_multiply(s)
                    return res
            func = f
        # s is a density matrix
        # tf is the total simulation time
        z, infodict = odeintw(func, s, np.arange(t0, tf, dt), full_output=True)
        return z

    def run_mc_solver(self, s, t0, tf, dt=.1, func=None):
        pass
