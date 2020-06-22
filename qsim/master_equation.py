from scipy.integrate import ode
from qsim.tools.operations import *
from qsim.tools import tools
from qsim.schrodinger_equation import SchrodingerEquation
from odeintw import odeintw
import numpy as np
import numpy.random
__all__ = ['MasterEquation', 'StochasticWavefunction']

class MasterEquation(object):
    def __init__(self, hamiltonians = None, jump_operators = None):
        # Jump operators is a list of LindbladNoise objects
        # Hamiltonian is a function of time
        if hamiltonians is None:
            hamiltonians = []
        if jump_operators is None:
            jump_operators = []
        self.hamiltonians = hamiltonians
        self.jump_operators = jump_operators


    def run_ode_solver(self, s, t0, tf, num = 2, func = None):
        is_ket = tools.is_ket(s)
        if func is None:
            def f(state, t):
                # hbar is set to one
                res = np.zeros(state.shape)
                for hamiltonian in self.hamiltonians:
                    res = res + -1j*(hamiltonian.left_multiply(s, is_ket=is_ket) - hamiltonian.right_multiply(s, is_ket=is_ket))
                for jump_operator in self.jump_operators:
                # TODO: make this selective in which qubits are operated on. Perhaps it should not default to everything
                    res = res + jump_operator.all_qubit_liouvillian(state)
                return res

            func = f
        # s is a State object
        # tf is the total simulation time
        z, infodict = odeintw(func, s, np.linspace(t0, tf, num=num), full_output=True)
        return z


class StochasticWavefunction(object):
    def __init__(self, hamiltonians=None, jump_operators=None):
        if hamiltonians is None:
            hamiltonians = []
        self.hamiltonians = hamiltonians
        if jump_operators is None:
            jump_operators = []
        self.hamiltonians = hamiltonians
        self.jump_operators = jump_operators

    def run(self, s, t0, tf, dt, func=None):
        # Compute probability that we have a jump
        times = np.arange(t0, tf, dt)
        outputs = np.zeros((times.shape[0], s.shape[0], s.shape[1]), dtype=np.complex128)
        se = schrodinger_equation.SchrodingerEquation(hamiltonians=self.hamiltonians)
        for (j, time) in zip(range(times.shape[0]), times):
            output = se.run_ode_solver(s, time, time+2*dt, dt)[-1]
            # TODO: make this work for multiple jumps
            # Make a list that allows you to select the possible jumps in a random order
            order = list(range(len(self.jump_operators)))
            np.random.shuffle(order)
            for i in order:
                jump = self.jumps[i]
                jump_probability = jump.jump_rate(s)*dt
                if np.random.uniform() < jump_probability:
                    s = jump.random_jump(s)
                    # Renormalize state
                    s = s / np.linalg.norm(s)
                    break # Don't consider any other jumps in this round
                else:
                    jump_probability_integrated = 1 - np.linalg.norm(np.squeeze(output.T)) ** 2
                    s = output/(1-jump_probability_integrated)**.5
            outputs[j,...] = s
        return outputs


