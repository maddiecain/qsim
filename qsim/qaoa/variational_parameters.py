import numpy as np

from qsim.state import State
from qsim import tools, operations

EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1

class VariationalParameter(object):
    def __init__(self, evolve, multiply, param=None):
        self.evolve = evolve
        self.multiply = multiply
        self.param = param

class HamiltonianB(VariationalParameter):
    def __init__(self, param=None):
        super().__init__(self.evolve_B, self.multiply_B, param)

    def multiply_B(self, s: State):
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(s.N):
            if s.is_ket:
                out = out + operations.single_qubit_operation(s.state, i, tools.SIGMA_X_IND, is_pauli=True, is_ket=s.is_ket)
            else:
                state = s.state
                state = state.reshape((2 ** i, 2, -1), order='F').copy()
                state = np.flip(state, 1)
                state = state.reshape(s.state.shape, order='F')
                out = out + state
        s.state = out

    def evolve_B(self, s: State, beta):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        s.all_qubit_rotation(beta, tools.SX)


class HamiltonianC(VariationalParameter):
    def __init__(self, C, param=None):
        super().__init__(self.evolve_C, self.multiply_C, param)
        self.C = C

    def evolve_C(self, s, gamma):
        if s.is_ket:
            s.state = np.exp(-1j * gamma * self.C) * s.state
        else:
            s.state = np.exp(-1j * gamma * self.C) * s.state * np.exp(1j * gamma * self.C).T

    def multiply_C(self, s: State):
        s.state = self.C * s.state

