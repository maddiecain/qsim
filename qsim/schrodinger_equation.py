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

    def run_ode_solver(self, s, t0, tf, dt=.1, schedule=None, return_infodict=False):
        """Numerically integrates the Schrodinger equation"""
        assert not tools.is_ket(s)
        if schedule is None:
            def schedule(t):
                return [1]*len(self.hamiltonians)

        def f(s, t):
            coefficients = schedule(t)
            res = np.zeros(s.shape)
            for i in range(len(self.hamiltonians)):
                res = res - 1j * coefficients[i] * self.hamiltonians.left_multiply(s, is_ket=False)
            return res
        # s is a ket specifying the initial state
        # tf is the total simulation time
        z, infodict = odeintw(f, s, np.arange(t0, tf, dt), full_output=True)
        if return_infodict: return z, infodict
        return z


