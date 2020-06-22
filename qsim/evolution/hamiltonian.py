import numpy as np
import networkx as nx
from qsim.state import qubit, state_tools
from qsim import tools, operations
import math
from scipy.linalg import expm
from qsim.graph_algorithms import graph


class HamiltonianB(object):
    def __init__(self, pauli='X', code=None):
        super().__init__()
        self.pauli = pauli
        self.time_independent = True
        if code is None:
            code = qubit
        self.code = code

    def left_multiply(self, state, is_ket=True):
        out = np.zeros_like(state, dtype=np.complex128)
        for i in range(int(math.log(state.shape[0], self.code.d) / self.code.n)):
            # Use Pauli operations because it's a little bit faster
            if self.pauli == 'X':
                out = out + self.code.left_multiply(state, [i], ['X'], is_ket=is_ket, pauli=True)
            elif self.pauli == 'Y':
                out = out + self.code.left_multiply(state, [i], ['Y'], is_ket=is_ket, pauli=True)
            elif self.pauli == 'Z':
                out = out + self.code.left_multiply(state, [i], ['Z'], is_ket=is_ket, pauli=True)
        return out

    def right_multiply(self, state, is_ket=True):
        out = np.zeros_like(state, dtype=np.complex128)
        for i in range(state_tools.num_qubits(state, self.code)):
            if self.pauli == 'X':
                out = out + self.code.right_multiply(state, [i], ['X'], is_ket=is_ket, pauli=True)
            elif self.pauli == 'Y':
                out = out + self.code.right_multiply(state, [i], ['Y'], is_ket=is_ket, pauli=True)
            elif self.pauli == 'Z':
                out = out + self.code.right_multiply(state, [i], ['Z'], is_ket=is_ket, pauli=True)
        return out

    def evolve(self, state, beta, is_ket=True):
        r"""
        Use reshape to efficiently implement evolution under :math:`H_B=\\sum_i X_i`
        """
        for i in range(state_tools.num_qubits(state, self.code)):
            if self.pauli == 'X':
                state = self.code.rotation(state, [i], beta, self.code.X, is_ket=is_ket, is_involutary=True)
            elif self.pauli == 'Y':
                state = self.code.rotation(state, [i], beta, self.code.X, is_ket=is_ket, is_involutary=True)
            elif self.pauli == 'Z':
                state = self.code.rotation(state, [i], beta, self.code.X, is_ket=is_ket, is_involutary=True)
        return state


class HamiltonianC(object):
    def __init__(self, G: nx.Graph, mis=True, code=None):
        # If MIS is true, create an MIS Hamiltonian. Otherwise, make a MaxCut Hamiltonian
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.
        """
        if code is None:
            self.code = qubit
        else:
            self.code = code
        self.mis = mis
        self.time_independent = True
        self.graph = G
        self.optimization = 'max'
        self.N = self.graph.number_of_nodes()
        C = np.zeros([self.code.d ** (self.code.n * self.N), 1])
        if tools.is_diagonal(self.code.Z):
            self.is_diagonal = True
            Z = np.expand_dims(np.diagonal(self.code.Z), axis=0).T
            myeye = lambda n: np.ones(np.asarray(Z.shape) ** n)
        else:
            self.is_diagonal = False
            Z = self.code.Z
            myeye = lambda n: np.identity(np.asarray(Z.shape[0]) ** n)

        for a, b in self.graph.edges:
            if b < a:
                temp = a
                a = b
                b = temp
            C = C - self.graph[a][b]['weight'] * tools.tensor_product(
                [myeye(a), Z, myeye(b - a - 1), Z, myeye(self.N - b - 1)])
        if self.mis:
            for c in self.graph.nodes:
                C = C + tools.tensor_product([myeye(c), Z, myeye(self.N - c - 1)])
        self.hamiltonian = C

    def evolve(self, state, gamma, is_ket=True):
        if is_ket:
            if self.is_diagonal:
                return np.exp(-1j * gamma * self.hamiltonian) * state
            else:
                return expm(-1j * gamma * self.hamiltonian) @ state
        else:
            if self.is_diagonal:
                return np.exp(-1j * gamma * self.hamiltonian) * state * np.exp(
                    1j * gamma * self.hamiltonian).T
            else:
                temp = expm(-1j * gamma * self.hamiltonian)
                return temp @ state @ temp.conj().T

    def left_multiply(self, state, is_ket=True):
        if self.is_diagonal:
            return self.hamiltonian * state
        else:
            return self.hamiltonian @ state

    def right_multiply(self, state, is_ket=True):
        # Already real, so you don't need to conjugate
        if self.is_diagonal:
            return state * self.hamiltonian.T
        else:
            return state @ self.hamiltonian.T

    def cost_function(self, state, is_ket):
        # Returns <s|C|s>
        if is_ket:
            return np.real(np.vdot(state, self.hamiltonian * state))
        else:
            # Density matrix
            return np.real(np.squeeze(tools.trace(self.hamiltonian * state)))


class HamiltonianGlobalPauli(object):
    def __init__(self, pauli: str, code=None):
        super().__init__()
        if code is None:
            self.code = qubit
        else:
            self.code = code
        self.pauli = pauli
        if self.pauli == 'X':
            self.operator = self.code.X
        elif self.pauli == 'Y':
            self.operator = self.code.Y
        elif self.pauli == 'Z':
            self.operator = self.code.Z
        self.hamiltonian = None

    def evolve(self, state, alpha, is_ket=True):
        if self.hamiltonian is None:
            """Initialize the Hamiltonian only once, as it is costly."""
            self.hamiltonian = tools.tensor_product([self.operator] * int(math.log(state.shape[-1], self.code.d) /
                                                                          self.code.n))
        return state.multiply(np.cos(alpha) * np.identity(2 ** state.N) - 1j * np.sin(alpha) * self.hamiltonian)

    def left_multiply(self, state, is_ket=True):
        all_qubits = list(range(int(math.log(state.shape[0], self.code.d) / self.code.n)))
        return self.code.left_multiply(state, all_qubits,
                                       [self.pauli] * int(math.log(state.shape[0], self.code.d) / self.code.n),
                                       is_ket=is_ket, pauli=True)

    def right_multiply(self, state, is_ket=True):
        all_qubits = list(range(int(math.log(state.shape[0], self.code.d) / self.code.n)))
        return self.code.right_multiply(state, all_qubits,
                                        [self.pauli] * int(math.log(state.shape[0], self.code.d) / self.code.n),
                                        is_ket=is_ket, pauli=True)


class HamiltonianBookatzPenalty(object):
    def __init__(self, code=None):
        if code is None:
            self.code = qubit
        else:
            self.code = code
        self.projector = np.identity(self.code.d ** self.code.n) - self.code.codespace_projector

    def evolve(self, state, penalty, is_ket=True):
        # Term for a single qubit
        for i in range(state_tools.num_qubits(state, self.code)):
            state = self.code.rotation(state, [i], penalty, self.projector, is_ket=is_ket, is_idempotent=True)
        return state

    def left_multiply(self, state, is_ket=True):
        out = np.zeros_like(state, dtype=np.complex128)
        for i in range(state_tools.num_qubits(state, self.code)):
            if is_ket:
                out = out + self.code.left_multiply(state, [i], self.projector, is_ket=is_ket)
            else:
                out = out + self.code.left_multiply(state, [i], self.projector, is_ket=is_ket)
        return out

    def right_multiply(self, state, is_ket=True):
        out = np.zeros_like(state, dtype=np.complex128)
        for i in range(state_tools.num_qubits(state, self.code)):
            out = out + self.code.single_qubit_right_multiply(state, [i], self.projector, is_ket=is_ket)
        return out


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
    def __init__(self, G: nx.Graph, blockade_energy=1, detuning=0, code=None):
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.
        """
        if code is None:
            code = qubit
        self.code = code
        self.graph = G
        self.N = self.graph.number_of_nodes()
        self.blockade_energy = blockade_energy
        self.detuning = detuning
        self.optimization = 'max'
        C = np.zeros([self.code.d ** self.N, 1]).T

        r = np.zeros(self.code.d)
        r[0] = 1
        myeye = lambda n: np.ones(np.asarray(self.code.d) ** n)

        for i, j in G.edges:
            if j < i:
                temp = i
                i = j
                j = temp
            C = C + self.blockade_energy * self.graph[i][j]['weight'] * tools.tensor_product(
                [myeye(i), r, myeye(j - i - 1), r, myeye(self.N - j - 1)])
        for c in G.nodes:
            C = C + self.detuning * tools.tensor_product([myeye(c), r, myeye(self.N - c - 1)])
        C = C.T
        self.hamiltonian = C
        self.IS_projector = graph.IS_projector(self.graph, self.code)

    def evolve(self, state, is_ket=True):
        if is_ket:
            return np.exp(-1j * self.hamiltonian) * state
        else:
            return np.exp(-1j * self.hamiltonian) * state * np.exp(
                1j * self.hamiltonian).T

    def left_multiply(self, state, is_ket=True):
        return self.hamiltonian * state

    def right_multiply(self, state, is_ket=True):
        # Already real, so you don't need to conjugate
        return state * self.hamiltonian.T

    def cost_function(self, state, is_ket=True):
        # Need to project into the IS subspace
        # Returns <s|C|s>
        if is_ket:
            return np.real(np.vdot(state, self.IS_projector*self.hamiltonian * state))
        else:
            # Density matrix
            return np.real(np.squeeze(tools.trace(self.IS_projector*self.hamiltonian * state)))



class HamiltonianRydbergHeisenberg(object):
    def __init__(self, G: nx.Graph, blockade_energy=1, code=None):
        if code is None:
            code = qubit
        self.code = code
        self.graph = G
        self.N = self.graph.number_of_nodes()
        self.blockade_energy = blockade_energy
        self.hamiltonian = None
        self.IS_projector = graph.IS_projector(self.graph, self.code)


    def left_multiply(self, s, is_ket=True):
        temp = np.zeros(s.shape)
        for edge in self.graph.edges:
            term = self.code.left_multiply(s, [edge[0], edge[1]], ['X', 'X'], is_ket=is_ket, pauli=True)
            temp = temp + term
            term = self.code.left_multiply(s, [edge[0], edge[1]], ['Y', 'Y'], is_ket=is_ket, pauli=True)
            temp = temp + term
            term = self.code.left_multiply(s, [edge[0], edge[1]], tools.tensor_product([self.code.U, self.code.U]),
                                           is_ket=is_ket, pauli=False)
            temp = temp + self.blockade_energy * term
        return temp

    def right_multiply(self, s, is_ket=True):
        temp = np.zeros(s.shape)
        for edge in self.graph.edges:
            term = self.code.right_multiply(s, [edge[0], edge[1]], ['X', 'X'], is_ket=is_ket, pauli=True)
            temp = temp + term
            term = self.code.right_multiply(s, [edge[0], edge[1]], ['Y', 'Y'], is_ket=is_ket, pauli=True)
            temp = temp + term
            term = self.code.right_multiply(s, [edge[0], edge[1]], tools.tensor_product([self.code.U, self.code.U]),
                                            is_ket=is_ket, pauli=False)
            temp = temp + self.blockade_energy * term
        return temp

    def evolve(self, state, is_ket=True):
        if self.hamiltonian is None:
            """Initialize the Hamiltonian."""
            hamiltonian = np.zeros((state.shape[0], state.shape[0]))
            for (i, j) in self.graph.edges:
                hamiltonian = hamiltonian + \
                              tools.tensor_product([tools.identity(self.code.n * i, d=self.code.d), self.code.X,
                                                    tools.identity(self.code.n * (j - i - 1), d=self.code.d),
                                                    self.code.X,
                                                    tools.identity(self.code.n * (self.N - j - 1), d=self.code.d)])
                hamiltonian = hamiltonian + \
                              tools.tensor_product([tools.identity(self.code.n * i, d=self.code.d), self.code.Y,
                                                    tools.identity(self.code.n * (j - i - 1), d=self.code.d),
                                                    self.code.Y,
                                                    tools.identity(self.code.n * (self.N - j - 1), d=self.code.d)])
                hamiltonian = hamiltonian + self.blockade_energy * \
                              tools.tensor_product([tools.identity(self.code.n * i, d=self.code.d), self.code.U,
                                                    tools.identity(self.code.n * (j - i - 1), d=self.code.d),
                                                    self.code.U,
                                                    tools.identity(self.code.n * (self.N - j - 1), d=self.code.d)])
            self.hamiltonian = hamiltonian
        if not is_ket:
            return expm(-1j * self.hamiltonian) @ state @ expm(1j * self.hamiltonian)
        if is_ket:
            return expm(-1j * self.hamiltonian) @ state

    def cost_function(self, state, is_ket=True):
        # Need to project into the IS subspace
        # Returns <s|C|s>
        if is_ket:
            return np.real(np.vdot(state, self.IS_projector * self.hamiltonian * state))
        else:
            # Density matrix
            return np.real(np.squeeze(tools.trace(self.IS_projector * self.hamiltonian * state)))
