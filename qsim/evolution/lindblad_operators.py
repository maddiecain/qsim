from qsim.tools import operations, tools
import numpy as np
import networkx as nx
from qsim.state import qubit, state_tools
import math


class SpontaneousEmission(object):
    def __init__(self, transition: tuple, rate, code=None):
        # jump_operators and weights are numpy arrays
        if code is None:
            code = qubit
        self.code = code
        self.rate = rate
        self.transition = sorted(transition)
        jump_operator = np.zeros((self.code.d, self.code.d))
        jump_operator[self.transition[1], self.transition[0]] = 1
        self.jump_operator = jump_operator

    def liouvillian(self, state, apply_to=None, is_ket=True):
        if apply_to is None:
            # Apply to all physical qubits
            apply_to = list(range(int(math.log(state.shape[0], self.code.d))))
        a = np.zeros(state.shape)
        for i in range(len(apply_to)):
            a = a + self.rate * (self.code.multiply(state, [apply_to[i]], self.jump_operator, is_ket=is_ket) -
                                 1 / 2 * self.code.left_multiply(state, [apply_to[i]],
                                                                 self.jump_operator.T @ self.jump_operator, is_ket=is_ket) -
                                 1 / 2 * self.code.right_multiply(state, [apply_to[i]],
                                                                  self.jump_operator.T @ self.jump_operator, is_ket=is_ket))
        return a


class MISLoweringJumpOperator(object):
    """Jump operators which enforce the independent set constraint."""

    def __init__(self, graph: nx.Graph, rate):
        super().__init__(jump_operators=[np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])],
                         weights=rate)
        # Construct the right jump_operators operators
        self.code = qubit
        self.graph = graph
        self.N = self.graph.number_of_nodes()
        jump_operators = []

        for (i, j) in graph.edges:
            for p in range(len(self.jump_operators)):
                temp = tools.tensor_product([self.jump_operators[p], tools.identity(self.N - 2)])
                temp = np.reshape(temp, 2 * np.ones(2 * self.N, dtype=int))
                temp = np.moveaxis(temp, [0, 1, self.N, self.N + 1], [i, j, self.N + i, self.N + j])
                temp = np.reshape(temp, (2 ** self.N, 2 ** self.N))
                jump_operators.append(temp)
        self.jump_operators = jump_operators

    def liouvillian(self, s, i):
        a = np.zeros(s.shape)
        a = a + self.weights * (self.jump_operators[i] @ s @ self.jump_operators[i].T -
                                1 / 2 * s @ self.jump_operators[i].conj().T @ self.jump_operators[i] -
                                1 / 2 * self.jump_operators[i].conj().T @ self.jump_operators[i] @ s)
        return a

    def all_qubit_liouvillian(self, s):
        a = np.zeros(s.shape)
        for i in range(len(self.jump_operators)):
            a = a + self.liouvillian(s, i)
        return a


class MISRaisingTwoLocal(object):
    def __init__(self, graph: nx.Graph, rate=1):
        self.graph = graph
        self.rate = rate

    def edge_jump(self, s, i, j, status, node):
        """Left multiply by c_i"""
        assert status == 0 or status == 1
        assert node == 0 or node == 1

        temp = s.copy()
        if status == 0:
            # Lower
            # Decide which node to lower
            temp = operations.single_qubit_operation(temp, j, up, is_ket=True)
            temp = operations.single_qubit_operation(temp, i, up, is_ket=True)
            if node == 0:
                # Lower i
                temp = operations.single_qubit_operation(temp, i, lower_spin, is_ket=True)
            else:
                temp = operations.single_qubit_operation(temp, j, lower_spin, is_ket=True)
        else:
            # Raise
            # Decide which node to raise
            temp = operations.single_qubit_operation(temp, j, down, is_ket=True)
            temp = operations.single_qubit_operation(temp, i, down, is_ket=True)
            if node == 0:
                # Lower i
                temp = operations.single_qubit_operation(temp, j, raise_spin, is_ket=True)
            else:
                temp = operations.single_qubit_operation(temp, i, raise_spin, is_ket=True)
        return temp

    def edge_probability(self, s, i, j, status):
        """Compute probability for jump by c_i"""
        assert status == 0 or status == 1
        if status == 0:
            # Lower a node
            term = s.copy()
            term = operations.single_qubit_operation(term, i, up, is_ket=True)
            term = operations.single_qubit_operation(term, j, up, is_ket=True)
            return np.real(np.squeeze(s.conj().T @ term * self.rate))
        else:
            # Raise a node
            term = s.copy()
            term = operations.single_qubit_operation(term, i, down, is_ket=True)
            term = operations.single_qubit_operation(term, j, down, is_ket=True)
            return np.real(np.squeeze(s.conj().T @ term * self.rate))

    def random_jump(self, s):
        # Compute all the probabilities
        probabilities = np.zeros((self.graph.m, 4))
        for (i, edge) in zip(range(self.graph.m), self.graph.edges):
            probabilities[i, 0] = self.edge_probability(s, edge[0], edge[1], 0)
            probabilities[i, 1] = probabilities[i, 0]
            probabilities[i, 2] = self.edge_probability(s, edge[0], edge[1], 1)
            probabilities[i, 3] = probabilities[i, 2]
        total_probability = np.sum(probabilities)
        index = np.random.choice(self.graph.m * 4, size=1, p=probabilities.flatten() / total_probability)[0]
        row = (index - index % 4) // 4
        if index % 4 == 0 or index % 4 == 1:
            # Lower the node
            return self.edge_jump(s, self.graph.edges[row][0], self.graph.edges[row][1], 0, index % 4)
        else:
            # Raise the node
            return self.edge_jump(s, self.graph.edges[row][0], self.graph.edges[row][1], 1, index % 4 - 2)

    def left_multiply(self, s):
        """Left multiply by the effective Hamiltonian"""
        temp = np.zeros(s.shape)
        # Two terms for each edge
        for edge in self.graph.edges:
            # Term for each node
            term = s.copy()
            term = operations.single_qubit_pauli(term, edge[0], 'Z', is_ket=True)
            term = operations.single_qubit_pauli(term, edge[1], 'Z', is_ket=True)
            temp = temp + s + term
        return -1j * self.rate * temp

    def jump_rate(self, s):
        """Compute the probability that a quantum jump happens on any node"""
        return np.squeeze(1j * s.conj().T @ self.left_multiply(s))


class MISDissipation(object):
    def __init__(self, graph: nx.Graph, IS_penalty = 1, code=None):
        self.graph = graph
        if code is None:
            code = qubit
        self.code = code
        self.N = self.graph.number_of_nodes()
        self.IS_penalty = IS_penalty
        # Construct jump operators for edges and nodes
        #node =  (np.identity(self.code.d * self.code.n) - self.code.Z) / 2
        #edge = self.IS_penalty * tools.tensor_product([self.code.U, self.code.U])
        node = self.code.X @ (np.identity(self.code.d*self.code.n)-self.code.Z)/2
        edge = self.IS_penalty * (tools.tensor_product([self.code.X, np.identity(self.code.d*self.code.n)]) +tools.tensor_product([np.identity(self.code.d*self.code.n), self.code.X]))@ tools.tensor_product([self.code.U, self.code.U])/2
        self.jump_operators = [node, edge]

    def liouvillian(self, state, apply_to=None, is_ket=True):
        if apply_to is None:
            # Apply to all physical qubits
            apply_to_nodes = list(self.graph.nodes)
            apply_to_edges = list(self.graph.edges)
            apply_to = (apply_to_nodes, apply_to_edges)
        a = np.zeros(state.shape)
        for j in range(2):
            for i in range(len(apply_to[j])):
                a = a + self.code.multiply(state, apply_to[j][i], self.jump_operators[j], is_ket=is_ket) -\
                                        1 / 2 * self.code.left_multiply(state, apply_to[j][i],
                                        self.jump_operators[j].T @ self.jump_operators[j], is_ket=is_ket) -\
                                        1 / 2 * self.code.right_multiply(state, apply_to[j][i],
                                        self.jump_operators[j].T @ self.jump_operators[j], is_ket=is_ket)
        return a

class Hadamard_Dissipation(object):
    def __init__(self, code=None):
        if code is None:
            code = qubit
        self.code = code
        #self.jump_operator=(np.identity(self.code.d*self.code.n)-self.code.X)/2

        self.jump_operator=self.code.Z @ (np.identity(self.code.d*self.code.n)-self.code.X)/2

    def liouvillian(self, state, apply_to=None, is_ket=True):
        if apply_to is None:
            # Apply to all physical qubits
            apply_to = list(range(int(math.log(state.shape[0], self.code.d))))
        a = np.zeros(state.shape)
        for i in range(len(apply_to)):
            a = a + self.code.multiply(state, [apply_to[i]], self.jump_operator, is_ket=is_ket) - \
                                 1 / 2 * self.code.left_multiply(state, [apply_to[i]],
                                                                 self.jump_operator.T @ self.jump_operator, is_ket=is_ket) - \
                                 1 / 2 * self.code.right_multiply(state, [apply_to[i]],
                                                                  self.jump_operator.T @ self.jump_operator, is_ket=is_ket)
        return a

