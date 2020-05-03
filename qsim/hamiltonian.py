import numpy as np
import scipy.linalg as sc
import networkx as nx

from qsim.state import *
from qsim import tools, operations

__all__ = ['Hamiltonian', 'HamiltonianBookatzPenalty', 'HamiltonianGlobalPauli', 'HamiltonianB', 'HamiltonianC',
           'HamiltonianMarvianPenalty']


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

    def __init__(self, hamiltonian=None):
        # Actual representation of the Hamiltonian as a matrix
        self.hamiltonian = hamiltonian

    def evolve(self, s: State, time):
        # Diagonalize then apply operation
        s.multiply(sc.expm(-1j * self.hamiltonian * time))

    def left_multiply(self, s: State):
        s.state = self.hamiltonian @ s.state

    def right_multiply(self, s: State):
        s.state = s.state @ self.hamiltonian.conj().T


class HamiltonianB(Hamiltonian):
    def __init__(self):
        super().__init__()

    def left_multiply(self, s: State):
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if s.is_ket:
                # Use opX because it's a little bit faster
                out = out + s.opX(i, overwrite=False)
            else:
                out = out + operations.left_multiply(s.state, i, s.X, is_ket=s.is_ket)
        s.state = out

    def right_multiply(self, s: State):
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            out = out + operations.right_multiply(s.state, i, s.X, is_ket=s.is_ket)
        s.state = out

    def evolve(self, s: State, beta):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        s.all_qubit_rotation(beta, s.X)


class HamiltonianC(Hamiltonian):
    def __init__(self, graph: nx.Graph, mis=True, code=State):
        # If MIS is true, create an MIS Hamiltonian. Otherwise, make a MaxCut Hamiltonian
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.
        """
        self.code = code
        self.mis = mis
        self.graph = graph
        self.N = self.graph.number_of_nodes()
        C = np.zeros([2 ** (self.code.n * self.N), 1])

        Z = np.expand_dims(np.diagonal(self.code.Z), axis=0).T
        myeye = lambda n: np.ones(np.asarray(Z.shape) ** n)

        for a, b in self.graph.edges:
            if b < a:
                temp = a
                a = b
                b = temp
            C = C + self.graph[a][b]['weight'] * tools.tensor_product(
                [myeye(a), Z, myeye(b - a - 1), Z, myeye(self.N - b - 1)])
        if self.mis:
            for c in self.graph.nodes:
                C = C + tools.tensor_product([myeye(c), Z, myeye(self.N - c - 1)])
        self.hamiltonian_diag = C
        super().__init__(hamiltonian=np.diag(np.squeeze(C.T)))

    def evolve(self, s: State, gamma):
        if s.is_ket:
            s.state = np.exp(-1j * gamma * self.hamiltonian_diag) * s.state
        else:
            s.state = np.exp(-1j * gamma * self.hamiltonian_diag) * s.state * np.exp(
                1j * gamma * self.hamiltonian_diag).T

    def left_multiply(self, s: State):
        s.state = self.hamiltonian_diag * s.state

    def right_multiply(self, s: State):
        # Already real, so you don't need to conjugate
        s.state = s.state * self.hamiltonian_diag.T


class HamiltonianGlobalPauli(Hamiltonian):
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
        # TODO: make this more efficient (low priority)
        s.multiply(np.cos(alpha) * np.identity(2 ** s.N) - 1j * np.sin(alpha) * self.operator(n=s.N))

    def left_multiply(self, s: State):
        # TODO: make this more efficient (low priority)
        if self.pauli == 'X':
            s.state = np.flip(s.state, 0)
        else:
            s.state = self.operator(n=s.N) @ s.state

    def right_multiply(self, s: State):
        s.state = s.state @ self.operator(n=s.N).conj().T


class HamiltonianBookatzPenalty(Hamiltonian):
    def __init__(self):
        super().__init__()

    def evolve(self, s: State, penalty):
        # Term for a single qubit
        projector = np.identity(2 ** s.n) - s.proj
        op = np.exp(-1j * penalty) * projector - projector + np.identity(2 ** s.n)
        for i in range(int(s.N / s.n)):
            s.single_qubit_operation(i, op)

    def left_multiply(self, s: State):
        projector = np.identity(2 ** s.n) - s.proj
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if s.is_ket:
                out = out + s.single_qubit_operation(i, projector, overwrite=False)
            else:
                out = out + operations.left_multiply(s.state, i, projector, is_ket=s.is_ket)
        s.state = out

    def right_multiply(self, s: State):
        projector = np.identity(2 ** s.n) - s.proj
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            out = out + operations.right_multiply(s.state, i, projector, is_ket=s.is_ket)
        s.state = out


class HamiltonianMarvianPenalty(Hamiltonian):
    def __init__(self, Nx, Ny):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.N = 3 * Nx * Ny
        # Generate Hamiltonian
        # Two by two geometry (can be generalized in the future)
        hp = np.zeros([2 ** self.N, 2 ** self.N])
        for i in range(int(self.Nx * self.Ny)):
            # Add gauge interactions within a single logical qubit
            hp = hp + tools.tensor_product(
                [tools.identity(i * 3), tools.Z(), tools.Z(),
                 tools.identity(self.N - i * 3 - 2)]) + tools.tensor_product(
                [tools.identity(i * 3 + 1), tools.X(), tools.X(), tools.identity(self.N - i * 3 - 3)])
        # Between rows
        for j in range(self.Ny):
            # j is the number of rows
            for k in range(self.Nx):
                # k is the number of columns
                # Need to deal with edge effects
                # Add gauge interactions within a single logical qubit
                if k != self.Nx - 1:
                    # Along the same row
                    hp = hp + tools.tensor_product(
                        [tools.identity(j * self.Nx * 3 + k * 3), tools.X(), tools.identity(2), tools.X(),
                         tools.identity(self.N - (j * self.Nx * 3 + k * 3) - 4)]) + \
                         tools.tensor_product(
                             [tools.identity(j * self.Nx * 3 + k * 3 + 2), tools.Z(), tools.identity(2), tools.Z(),
                              tools.identity(self.N - (j * self.Nx * 3 + k * 3 + 2) - 4)])
                    # Along the same column
                if j != self.Ny - 1:
                    hp = hp + tools.tensor_product(
                        [tools.identity(j * self.Nx * 3 + k * 3), tools.X(), tools.identity(3 * self.Nx - 1),
                         tools.X(),
                         tools.identity(self.N - (j * self.Nx * 3 + k * 3) - 3 * self.Nx - 1)]) + \
                         tools.tensor_product(
                             [tools.identity(j * self.Nx * 3 + k * 3 + 2), tools.Z(),
                              tools.identity(3 * self.Nx - 1),
                              tools.Z(), tools.identity(self.N - (j * self.Nx * 3 + k * 3 + 2) - 3 * self.Nx - 1)])
        self.hamiltonian = -1 * hp
