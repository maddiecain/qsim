from qsim.tools import operations, tools
import numpy as np
import networkx as nx


class LindbladNoise(object):
    def __init__(self, jump_operators=None, weights=None, n=1):
        # POVM and weights are numpy arrays
        self.n = n
        if jump_operators is None:
            self.jump_operators = []
        else:
            self.jump_operators = jump_operators
        self.weights = weights
        if weights is None:
            # Is this a good default?
            self.weights = np.array([1] * len(jump_operators))

    def liouvillian(self, s, i):
        a = np.zeros(s.shape)
        for j in range(len(self.jump_operators)):
            a = a + self.weights[j] * (operations.single_qubit_operation(s, i, self.jump_operators[j]) -
                                       1 / 2 * operations.left_multiply(s, i, self.jump_operators[j].conj().T @ self.jump_operators[j]) -
                                       1 / 2 * operations.right_multiply(s, i, self.jump_operators[j].conj().T @ self.jump_operators[j]))
        return a

    def all_qubit_liouvillian(self, s):
        a = np.zeros(s.shape)
        for i in range(int(np.log2(s.shape[0]) / np.log2(2 ** self.n))):
            a = a + self.liouvillian(s, i)
        return a


class SpontaneousEmission(LindbladNoise):
    def __init__(self, rate):
        super().__init__(jump_operators=[np.array([[0, 0], [1, 0]])], weights=[rate])


class RydbergNoise(LindbladNoise):
    def __init__(self, N, rate, graph: nx.Graph):
        super().__init__(jump_operators=[np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                               np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),
                               np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])], weights=rate, n=4)
        # Construct the right jump_operators operators
        jump_operators = []
        np.set_printoptions(threshold=np.inf)

        for (i, j) in graph.edges:
            for p in range(len(self.jump_operators)):
                temp = tools.tensor_product([self.jump_operators[p], tools.identity(N - 2)])
                temp = np.reshape(temp, 2 * np.ones(2 * N, dtype=int))
                temp = np.moveaxis(temp, [0, 1, N, N + 1], [i, j, N + i, N + j])
                temp = np.reshape(temp, (2 ** N, 2 ** N))
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
