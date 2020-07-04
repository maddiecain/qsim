import numpy as np
from qsim.tools import operations, tools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import approximation

"""Class for storing basic graph operations."""


class Graph(object):
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        # Nodes are assumed to be integers from zero to # nodes - 1
        self.n = len([n for n in self.graph])
        self.nodes = np.arange(self.n, dtype=int)
        self.edges = [m for m in self.graph.edges]
        self.m = len(self.edges)
        self.configuration = np.zeros(self.graph.number_of_nodes())

    def random_node(self):
        """Return a random node"""
        return np.random.randint(0, self.n)

    def random_edge(self):
        """Return a random edge"""
        return self.edges[np.random.randint(0, self.m)]

    def raised_neighbors(self, i):
        """Returns the number of raised neighbors of node i."""
        raised = []
        for neighbor in self.graph.neighbors(i):
            if self.configuration[neighbor] == 1:
                raised.append(neighbor)
        return raised

    def free_node(self, i, ignore=None):
        """Checks if node i is free, ignoring some neighbors if indicated. If free, return True; otherwise,
        return False. ignore is assumed to be a list of nodes."""
        if self.configuration[i] == 0:
            if ignore is None:
                for neighbor in list(self.graph.neighbors(i)):
                    if self.configuration[neighbor] == 1:
                        return False
            else:
                for neighbor in list(self.graph.neighbors(i)):
                    if neighbor not in ignore:
                        if self.configuration[neighbor] == 1:
                            return False
            return True
        return False

    def free_nodes(self):
        """Count the number of free nodes"""
        free = []
        for j in range(self.n):
            if self.free_node(j): free.append(j)
        return free

    def raise_node(self, i):
        """Check if node i is free. If it is, change its value to 1. Return True if the node has been raised."""
        if self.configuration[i] == 1:
            return True  # Already raised
        else:
            if self.free_node(i):
                self.configuration[i] = 1
                return True
            return False

    def flip_flop(self, i):
        """If i is spin up, pick a neighbor of i at random, subject to the condition that it satisfies the condition
        that the node is free if i is ignored. If i is spin down, and there is exactly one neighbor of i that is spin up,
        swap the two. Return True if a flip flop has taken place, and False otherwise."""
        if self.configuration[i] == 1:
            shuffed_neighbors = list(self.graph.neighbors(i))
            np.random.shuffle(shuffed_neighbors)
            for neighbor in shuffed_neighbors:
                if self.free_node(neighbor, ignore=[i]):
                    self.configuration[i] = 0
                    self.configuration[neighbor] = 1
                    return True
        elif self.configuration[i] == 0:
            raised_neighbors = self.raised_neighbors(i)
            if len(raised_neighbors) == 1:
                self.configuration[i] = 1
                self.configuration[raised_neighbors[0]] = 0
                return True
        return False

    def spin_exchange(self, i):
        """Returns True if and only if a node i can spin exchange."""
        if self.configuration[i] == 1:
            for neighbor in self.graph.neighbors(i):
                if self.free_node(neighbor, ignore=[i]):
                    return True
        elif self.configuration[i] == 0:
            raised_neighbors = self.raised_neighbors(i)
            if len(raised_neighbors) == 1:
                return True
        return False

    def spin_exchanges(self):
        """Returns a list of pairs of nodes which can spin exchange."""
        exchanges = []
        for i in self.nodes:
            if self.configuration[i] == 1:
                for neighbor in self.graph.neighbors(i):
                    if self.free_node(neighbor, ignore=[i]):
                        exchanges.append((i, neighbor))
        return exchanges

    def random_spin_exchange(self):
        """Spin exchange on a random spin pair eligible to spin exchange."""
        exchanges = self.spin_exchanges()
        to_exchange = exchanges[np.random.randint(0, len(exchanges))]
        self.configuration[to_exchange[0]] = self.configuration[to_exchange[1]]
        self.configuration[to_exchange[1]] = self.configuration[to_exchange[0]]
        return self.configuration

    def random_raise(self):
        """Raise a random free node"""
        free_nodes = self.free_nodes()
        to_raise = free_nodes[np.random.randint(0, len(free_nodes))]
        self.raise_node(to_raise)
        return self.configuration

    def run_monte_carlo(self, t0, tf, dt, rates=None, config=None):
        """Run a Monte Carlo algorithm for time tf-t0 with given spin exchange and greedy spin raise rates."""
        if config is None:
            self.configuration = np.zeros(self.n)
        else:
            self.configuration = config
        if rates is None:
            rates = [1, 1]
        times = np.arange(t0, tf, dt)
        outputs = np.zeros((times.shape[0], self.configuration.shape[0], 1), dtype=np.complex128)
        for (j, time) in zip(range(times.shape[0]), times):
            if j == 0:
                outputs[0, :] = np.array([self.configuration.copy()]).T
            if j != 0:
                # Compute the probability that something happens
                # Probability of spin exchange
                probability_exchange = dt * rates[0] * len(self.spin_exchanges())  # TODO: figure out if this is over 2
                # Probability of raising
                probability_raise = dt * rates[1] * len(self.free_nodes())
                probability = probability_exchange + probability_raise
                if np.random.uniform() < probability:
                    if np.random.uniform(0, probability) < probability_exchange:
                        # Do a spin exchange
                        self.random_spin_exchange()
                    else:
                        # Raise a node
                        self.random_raise()
                outputs[j, :] = np.array([self.configuration.copy()]).T
        return outputs

    def draw_configuration(self):
        """Spin up is white, spin down is black."""
        # First, color the nodes
        color_map = []
        for node in self.graph:
            if self.configuration[node] == 1:
                color_map.append('teal')
            if self.configuration[node] == 0:
                color_map.append('black')
        nx.draw_circular(self.graph, node_color=color_map)
        plt.show()

    def configuration_weight(self):
        weight = np.sum(self.configuration)
        print('Configuration weight: ' + str(weight))
        return weight

    def nx_mis(self):
        return nx.algorithms.approximation.maximum_independent_set(self.graph)


def chain_graph(n):
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, n), weight=1)
    for i in range(n - 1):
        g.add_edge(i, i + 1, weight=-1)
    return g


def IS_projector(graph: nx.Graph, code):
    """Returns a projector (represented as a column vector) into the space of independent sets for general codes."""
    n = graph.number_of_nodes()
    # Check if U is diagonal
    if tools.is_diagonal(code.U):
        U = np.diag(code.U)
        proj = np.ones(code.d ** n)
        for i, j in graph.edges:
            if i > j:
                # Requires i < j
                temp = i
                i = j
                j = temp
            temp = tools.tensor_product(
                [np.ones(code.d ** i), U, np.ones(code.d ** (j - i - 1)), U,
                 np.ones(code.d ** (n - j - 1))])
            proj = proj * (np.ones(code.d ** n) - temp)
        return np.array([proj]).T
    else:
        proj = np.identity(code.d ** n)
        for i, j in graph.edges:
            if i > j:
                # Requires i < j
                temp = i
                i = j
                j = temp
            temp = tools.tensor_product([tools.identity(i, d=code.d), code.U, tools.identity(j - i - 1, d=code.d), code.U,
                                         tools.identity(n - j - 1, d=code.d)])
            proj = proj @ (np.identity(code.d ** n) - temp)
        return np.array([np.diagonal(proj)]).T
