import numpy as np
import networkx as nx
from qsim.state import State
from qsim import tools, operations

__all__ = ['HamiltonianBookatzPenalty', 'HamiltonianGlobalPauli', 'HamiltonianB', 'HamiltonianC',
           'HamiltonianMarvianPenalty']


class HamiltonianB(object):
    def __init__(self, pauli='X', code=State):
        super().__init__()
        self.pauli = pauli
        self.time_independent = True

    def left_multiply(self, s: State, overwrite=True):
        out = np.zeros_like(s.state, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if s.is_ket:
                # Use op because it's a little bit faster
                if self.pauli == 'X':
                    out = out + s.single_qubit_pauli(i, 'X', overwrite=False)
                elif self.pauli == 'Y':
                    out = out + s.single_qubit_pauli(i, 'Y', overwrite=False)
                elif self.pauli == 'Z':
                    out = out + s.single_qubit_pauli(i, 'Z', overwrite=False)
            else:
                if self.pauli == 'X':
                    out = out + operations.left_multiply(s.state, i, s.X, is_ket=s.is_ket)
                elif self.pauli == 'Y':
                    out = out + operations.left_multiply(s.state, i, s.Y, is_ket=s.is_ket)
                elif self.pauli == 'Z':
                    out = out + operations.left_multiply(s.state, i, s.Z, is_ket=s.is_ket)
        if overwrite:
            s.state = out
        return out

    def right_multiply(self, s: State, overwrite=True):
        out = np.zeros_like(s.state, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if self.pauli == 'X':
                out = out + operations.right_multiply(s.state, i, s.X, is_ket=s.is_ket)
            elif self.pauli == 'Y':
                out = out + operations.right_multiply(s.state, i, s.Y, is_ket=s.is_ket)
            elif self.pauli == 'Z':
                out = out + operations.right_multiply(s.state, i, s.Z, is_ket=s.is_ket)
        if overwrite:
            s.state = out
        return out

    def evolve(self, s: State, beta, overwrite=True):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        if self.pauli == 'X':
            s.all_qubit_rotation(beta, s.X, overwrite=overwrite)
        elif self.pauli == 'Y':
            s.all_qubit_rotation(beta, s.Y, overwrite=overwrite)
        elif self.pauli == 'Z':
            s.all_qubit_rotation(beta, s.Z, overwrite=overwrite)


class HamiltonianC(object):
    def __init__(self, graph: nx.Graph, mis=True, code=State):
        # If MIS is true, create an MIS Hamiltonian. Otherwise, make a MaxCut Hamiltonian
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.
        """
        self.code = code
        self.mis = mis
        self.time_independent = True
        self.graph = graph
        self.N = self.graph.number_of_nodes()
        C = np.zeros([2 ** (self.code.n * self.N), 1])
        # TODO: make this work for non-diagonal Z_L
        Z = np.expand_dims(np.diagonal(self.code.Z), axis=0).T
        myeye = lambda n: np.ones(np.asarray(Z.shape) ** n)

        for a, b in self.graph.edges:
            if b < a:
                temp = a
                a = b
                b = temp
            # TODO: change this to be the correct MIS hamiltonian
            C = C + self.graph[a][b]['weight'] * tools.tensor_product(
                [myeye(a), Z, myeye(b - a - 1), Z, myeye(self.N - b - 1)])
        if self.mis:
            for c in self.graph.nodes:
                C = C + tools.tensor_product([myeye(c), Z, myeye(self.N - c - 1)])
        self.hamiltonian_diag = C

    def evolve(self, s: State, gamma, overwrite=True):
        if s.is_ket:
            if overwrite:
                s.state = np.exp(-1j * gamma * self.hamiltonian_diag) * s.state
            return np.exp(-1j * gamma * self.hamiltonian_diag) * s.state
        else:
            if overwrite:
                s.state = np.exp(-1j * gamma * self.hamiltonian_diag) * s.state * np.exp(
                    1j * gamma * self.hamiltonian_diag).T
            return np.exp(-1j * gamma * self.hamiltonian_diag) * s.state * np.exp(
                1j * gamma * self.hamiltonian_diag).T

    def left_multiply(self, s: State, overwrite=True):
        if overwrite:
            s.state = self.hamiltonian_diag * s.state
            return s.state
        return self.hamiltonian_diag * s.state

    def right_multiply(self, s: State, overwrite=True):
        # Already real, so you don't need to conjugate
        if overwrite:
            s.state = s.state * self.hamiltonian_diag.T
            return s.state
        return s.state * self.hamiltonian_diag.T


class HamiltonianGlobalPauli(object):
    def __init__(self, pauli: str, code=State):
        super().__init__()
        self.code = code
        self.pauli = pauli
        if self.pauli == 'X':
            self.operator = code.X
        elif self.pauli == 'Y':
            self.operator = code.Y
        elif self.pauli == 'Z':
            self.operator = code.Z

    def evolve(self, s: State, alpha, overwrite=True):
        if overwrite:
            s.state = s.multiply(np.cos(alpha) * np.identity(2 ** s.N) - 1j * np.sin(alpha) * tools.tensor_product([self.operator]*int(s.N/s.n)))
            return s.state
        return s.multiply(np.cos(alpha) * np.identity(2 ** s.N) - 1j * np.sin(alpha) * tools.tensor_product([self.operator]*int(s.N/s.n)))


    def left_multiply(self, s: State, overwrite=True):
        if overwrite:
            if self.pauli == 'X' and isinstance(s, State):
                s.state = np.flip(s.state, 0)
            else:
                for i in range(int(s.N/s.n)):
                    if self.pauli == 'X':
                        s.opX(i, overwrite=overwrite)
                    elif self.pauli == 'Y':
                        s.opY(i, overwrite=overwrite)
                    elif self.pauli == 'Z':
                        s.opZ(i, overwrite=overwrite)
            return s.state
        else:
            if self.pauli == 'X':
                return np.flip(s.state, 0)
            else:
                temp = np.empty_like(s.state)
                for i in range(int(s.N / s.n)):
                    if self.pauli == 'X':
                        temp = s.opX(i, overwrite=overwrite)
                    elif self.pauli == 'Y':
                        s.opY(i, overwrite=overwrite)
                    elif self.pauli == 'Z':
                        s.opZ(i, overwrite=overwrite)

    def right_multiply(self, s: State, overwrite=True):
        s.state = s.state @ tools.tensor_product([self.operator]*int(s.N/s.n)).conj().T


class HamiltonianBookatzPenalty(object):
    def __init__(self):
        super().__init__()

    def evolve(self, s: State, penalty, overwrite=True):
        # Term for a single qubit
        projector = np.identity(2 ** s.n) - s.proj
        op = np.exp(-1j * penalty) * projector - projector + np.identity(2 ** s.n)
        for i in range(int(s.N / s.n)):
            s.single_qubit_operation(i, op, overwrite=True)

    def left_multiply(self, s: State, overwrite=True):
        projector = np.identity(2 ** s.n) - s.proj
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            if s.is_ket:
                out = out + s.single_qubit_operation(i, projector, overwrite=False)
            else:
                out = out + operations.left_multiply(s.state, i, projector, is_ket=s.is_ket)
        s.state = out

    def right_multiply(self, s: State, overwrite=True):
        projector = np.identity(2 ** s.n) - s.proj
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(int(s.N / s.n)):
            out = out + operations.right_multiply(s.state, i, projector, is_ket=s.is_ket)
        s.state = out


class HamiltonianMarvianPenalty(object):
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


class HamiltonianRydberg(object):
    def __init__(self, graph: nx.Graph, mis=True, code=State, penalty=1):
        # If MIS is true, create an MIS Hamiltonian. Otherwise, make a MaxCut Hamiltonian
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.
        """
        self.code = code
        self.mis = mis
        self.graph = graph
        self.N = self.graph.number_of_nodes()
        C = np.zeros([2 ** (self.code.n * self.N), 1]).T

        sigma_plus = np.array([1, 0])
        rr = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        myeye = lambda n: np.ones(np.asarray(sigma_plus.shape[0]) ** n)

        for i, j in graph.edges:
            temp = tools.tensor_product([rr, tools.identity(self.N - 2)])
            temp = np.reshape(temp, 2 * np.ones(2 * self.N, dtype=int))
            temp = np.moveaxis(temp, [0, 1, self.N, self.N + 1], [i, j, self.N + i, self.N + j])
            temp = np.reshape(temp, (2 ** self.N, 2 ** self.N))
            C = C + np.array([np.diagonal(temp)]) * penalty * -1
        for c in graph.nodes:
            C = C + tools.tensor_product([myeye(c), sigma_plus, myeye(self.N - c - 1)])
        C = C.T
        self.hamiltonian_diag = C
        super().__init__(hamiltonian=np.diag(np.squeeze(C)))

    def evolve(self, s: State, gamma, overwrite=True):
        if s.is_ket:
            if overwrite:
                s.state = np.exp(-1j * gamma * self.hamiltonian_diag) * s.state
            return np.exp(-1j * gamma * self.hamiltonian_diag) * s.state
        else:
            if overwrite:
                s.state = np.exp(-1j * gamma * self.hamiltonian_diag) * s.state * np.exp(
                    1j * gamma * self.hamiltonian_diag).T
            return np.exp(-1j * gamma * self.hamiltonian_diag) * s.state * np.exp(
                1j * gamma * self.hamiltonian_diag).T

    def left_multiply(self, s: State, overwrite=True):
        if overwrite:
            s.state = self.hamiltonian_diag * s.state
        return self.hamiltonian_diag * s.state

    def right_multiply(self, s: State, overwrite=True):
        # Already real, so you don't need to conjugate
        if overwrite:
            s.state = s.state * self.hamiltonian_diag.T
        return s.state * self.hamiltonian_diag.T
