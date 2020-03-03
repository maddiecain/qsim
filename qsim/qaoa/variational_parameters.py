import numpy as np
import scipy.linalg as sc

from qsim.state import *
from qsim import tools, operations

EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1
__all__ = ['Hamiltonian', 'HamiltonianBookatzPenalty', 'HamiltonianPauli', 'HamiltonianB', 'HamiltonianC']


class Hamiltonian(object):
    """:class:`Hamiltonian` defines a new variational parameter for QAOA.

    :param evolve: evolves :class:`Hamiltonian`, defaults to [DefaultParamVal]
    :type evolve: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    def __init__(self, hamiltonian = None):
        self.hamiltonian = hamiltonian

    def evolve(self, s: State, time):
        # Diagonalize then apply operation
        s.multiply(sc.expm(-1j*self.hamiltonian*time))

    def multiply(self, s: State):
        s.multiply(self.hamiltonian)


class HamiltonianB(Hamiltonian):
    def __init__(self):
        super().__init__()

    def multiply(self, s: State):
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if s.is_ket:
                out = out + operations.single_qubit_operation(s.state, i, 'X', is_pauli=True,
                                                              is_ket=s.is_ket, d=2 ** s.n)
            else:
                # This is just a single qubit operation
                state = s.state
                state = state.reshape((2 ** (s.N - (s.n * i) - 1), 2 ** (s.n * i), -1), order='F').transpose([1, 0, 2])
                state = np.dot(s.X, state.reshape((2 ** s.n, -1), order='F'))
                state = state.reshape((2 ** (s.n * i), 2 ** (s.N - (s.n * i) - 1), -1), order='F').transpose([1, 0, 2])
                state = state.reshape(s.state.shape, order='F')
                out = out + state
        s.state = out

    def evolve(self, s: State, beta):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        s.all_qubit_rotation(beta, s.X)


class HamiltonianC(Hamiltonian):
    def __init__(self, C):
        super().__init__(hamiltonian=C)

    def evolve(self, s: State, gamma):
        if s.is_ket:
            s.state = np.exp(-1j * gamma * self.hamiltonian) * s.state
        else:
            s.state = np.exp(-1j * gamma * self.hamiltonian) * s.state * np.exp(1j * gamma * self.hamiltonian).T

    def multiply(self, s: State):
        s.state = self.hamiltonian * s.state


class HamiltonianPauli(Hamiltonian):
    def __init__(self, pauli: str):
        super().__init__()
        self.pauli = pauli
        if self.pauli == 'X':
            self.operator = tools.X
        elif self.pauli == 'Y':
            self.operator = tools.Y
        elif self.pauli == 'Z':
            self.operator = tools.Z

    def evolve(self, s: State, alpha):
        # TODO: make this more efficient
        s.multiply(np.cos(alpha) * np.identity(2 ** s.N) - 1j * np.sin(alpha) * self.operator(n=s.N))

    def multiply(self, s: State):
        # TODO: make this more efficient
        if self.pauli == 'X':
            s.state = np.flip(s.state, 0)
        else:
            self.operator(n=s.N) @ s.state


class HamiltonianBookatzPenalty(Hamiltonian):
    def __init__(self):
        super().__init__()

    def evolve(self, s: State, penalty):
        projector = np.identity(2 ** s.n) - (tools.outer_product(s.basis[0], s.basis[0]) +
                                             tools.outer_product(s.basis[1], s.basis[1]))
        op = np.exp(-1j * penalty) * projector - projector + np.identity(2 ** s.n)
        for i in range(int(s.N / s.n)):
            s.single_qubit_operation(i, op)

    def multiply(self, s: State):
        projector = np.identity(2 ** s.n) - tools.outer_product(s.basis[0], s.basis[0]) + \
                    tools.outer_product(s.basis[1], s.basis[1])
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if s.is_ket:
                out = out + operations.single_qubit_operation(s.state, i, projector, is_pauli=False,
                                                              is_ket=s.is_ket, d=2 ** s.n)
            else:
                # This is just a single qubit operation
                # TODO: clean this up!
                state = s.state
                state = state.reshape((2 ** (s.N - (s.n * i) - 1), 2 ** (s.n * i), -1), order='F').transpose([1, 0, 2])
                state = np.dot(projector, state.reshape((2 ** s.n, -1), order='F'))
                state = state.reshape((2 ** (s.n * i), 2 ** (s.N - (s.n * i) - 1), -1), order='F').transpose([1, 0, 2])
                state = state.reshape(s.state.shape, order='F')
                out = out + state
        s.state = out
