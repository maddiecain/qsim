import numpy as np

from qsim.state import State
from qsim import tools, operations

EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1


class VariationalParameter(object):
    def __init__(self, evolve, multiply, param=None, error=False):
        self.evolve = evolve
        self.multiply = multiply
        self.param = param
        self.error = error


class HamiltonianB(VariationalParameter):
    def __init__(self, param=None, error=False):
        super().__init__(self.evolve_B, self.multiply_B, param, error=error)

    def multiply_B(self, s: State):
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(s.N):
            if s.is_ket:
                out = out + operations.single_qubit_operation(s.state, i, tools.SIGMA_X_IND, is_pauli=True,
                                                              is_ket=s.is_ket)
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
    def __init__(self, C, param=None, error=False):
        super().__init__(self.evolve_C, self.multiply_C, param, error=error)
        self.C = C

    def evolve_C(self, s, gamma):
        if s.is_ket:
            s.state = np.exp(-1j * gamma * self.C) * s.state
        else:
            s.state = np.exp(-1j * gamma * self.C) * s.state * np.exp(1j * gamma * self.C).T

    def multiply_C(self, s: State):
        s.state = self.C * s.state


class HamiltonianPauli(VariationalParameter):
    def __init__(self, pauli: int, param = None, error=False):
        super().__init__(self.evolve_pauli, self.multiply_pauli, param=param, error=error)
        self.pauli = pauli
        if self.pauli == tools.SIGMA_X_IND:
            self.operator = tools.X
        elif self.pauli == tools.SIGMA_Y_IND:
            self.operator = tools.Y
        elif self.pauli == tools.SIGMA_Z_IND:
            self.operator = tools.Z

    def evolve_pauli(self, s: State, alpha):
        # TODO: make this more efficient
        s.multiply(np.cos(alpha) * np.identity(2**s.N) - 1j * np.sin(alpha) * self.operator(n=s.N))

    def multiply_pauli(self, s: State):
        # TODO: make this more efficient
        s.state = self.operator(n=s.N)@s.state
