import numpy as np
from qsim.tools import operations
from qsim import schrodinger_equation
from qsim import hamiltonian
from qsim.state import *
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import approximation

up = np.array([[1,0], [0,0]])
down = np.array([[0,0], [0,1]])
raise_spin = np.array([[0,1],[0,0]])
lower_spin = np.array([[0,0],[1,0]])


class StochasticWavefunction(object):
    def __init__(self, hamiltonians, jumps):
        self.hamiltonians = hamiltonians
        self.jumps = jumps

    def run(self, s, t0, tf, dt):
        # Compute probability that we have a jump
        times = np.arange(t0, tf, dt)
        outputs = np.zeros((times.shape[0], s.shape[0], s.shape[1]), dtype=np.complex64)
        se = schrodinger_equation.SchrodingerEquation(hamiltonians=self.hamiltonians, is_ket=True)
        for (j, time) in zip(range(times.shape[0]), times):
            output = se.run_ode_solver(s, time, time+2*dt, dt)[-1]
            jump_probability_integrated = 1-np.linalg.norm(np.squeeze(output.T))**2
            jump = self.jumps[0]
            jump_probability = jump.jump_rate(s)*dt
            if np.random.uniform() < jump_probability:
                s = jump.random_jump(s)
                # Renormalize state
                s = s / np.linalg.norm(s)
            else:
                s = output/(1-jump_probability_integrated)**.5
            outputs[j,...] = s
        return outputs




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

    def raise_node(self, i):
        """Check if node i is free. If it is, change its value to 1. Return True if the node has been raised."""
        if self.configuration[i] == 1:
            return True # Already raised
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
        print('Configuration weight: '+str(weight))
        return weight


class HamiltonianHeisenberg(object):
    def __init__(self, graph: Graph, k = 1):
        self.graph = graph
        self.k = k

    def left_multiply(self, s):
        temp = s.copy()
        for edge in self.graph.edges:
            term = s.copy()
            term = operations.single_qubit_pauli(term, edge[0], 'X', is_ket=True)
            term = operations.single_qubit_pauli(term, edge[1], 'X', is_ket=True)
            temp = temp + term
            term = s.copy()
            term = operations.single_qubit_pauli(term, edge[0], 'Y', is_ket=True)
            term = operations.single_qubit_pauli(term, edge[1], 'Y', is_ket=True)
            temp = temp + term
            term = s.copy()
            term = operations.single_qubit_operation(term, edge[0], up, is_ket=True)
            term = operations.single_qubit_operation(term, edge[1], up,  is_ket=True)
            temp = temp + self.k * term
        return temp


class GreedyNoiseTwoLocal(object):
    def __init__(self, graph: Graph, rate=1):
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
        index = np.random.choice(self.graph.m * 4, size=1, p=probabilities.flatten()/total_probability)[0]
        row = (index - index % 4) // 4
        if index % 4 == 0 or index % 4 == 1:
            # Lower the node
            return self.edge_jump(s, self.graph.edges[row][0], self.graph.edges[row][1], 0, index % 4)
        else:
            # Raise the node
            return self.edge_jump(s, self.graph.edges[row][0], self.graph.edges[row][1], 1, index % 4-2)

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


class GreedyNoise(object):
    def __init__(self, graph: Graph, rate=1):
        self.graph = graph
        self.rate = rate

    def node_jump(self, s, i):
        """Left multiply by c_i"""
        temp = s.copy()
        for neighbor in self.graph.graph.neighbors(i):
            temp = operations.single_qubit_operation(temp, neighbor, down, is_ket=True)
        temp = operations.single_qubit_operation(temp, i, down, is_ket=True)
        temp = operations.single_qubit_pauli(temp, i, 'X', is_ket=True)
        return temp

    def node_probability(self, s, i):
        """Compute probability for jump by c_i"""
        jumped = self.node_jump(s, i)
        return np.real(np.squeeze(jumped.conj().T @ jumped * self.rate))

    def random_jump(self, s):
        # Compute all the probabilities
        probabilities = np.zeros(self.graph.n)
        for (i, node) in zip(range(self.graph.n), self.graph.nodes):
            probabilities[i] = self.node_probability(s, i)
        total_probability = np.sum(probabilities)
        index = np.random.choice(self.graph.n, size=1, p=probabilities/total_probability)[0]
        return self.node_jump(s, index)

    def left_multiply(self, s):
        """Left multiply by the effective Hamiltonian"""
        term = np.zeros(s.shape)
        # Two terms for each edge
        for node in self.graph.nodes:
            # Term for each node
            temp = s.copy()
            for neighbor in self.graph.graph.neighbors(node):
                temp = operations.single_qubit_operation(temp, neighbor, down, is_ket=True)
            temp = operations.single_qubit_operation(temp, node, down, is_ket=True)
            term = temp + term
        return -1j * self.rate * term

    def jump_rate(self, s):
        """Compute the probability that a quantum jump happens on any node"""
        return np.squeeze(1j * s.conj().T @ self.left_multiply(s))


def simple_monte_carlo(d, n, iters = 10):
    """Simple Monte Carlo algorithm on a random d-regular graph with n nodes."""
    g = Graph(nx.random_regular_graph(d, n))
    # Randomly raise nodes:
    shuffled_nodes = g.nodes.copy()
    np.random.shuffle(shuffled_nodes)
    for node in shuffled_nodes:
        g.raise_node(node)
    g.configuration_weight()
    np.random.shuffle(shuffled_nodes)
    for k in range(iters):
        for node in shuffled_nodes:
            g.flip_flop(node)
        g.configuration_weight()
        for node in shuffled_nodes:
            g.raise_node(node)
        g.configuration_weight()
    print(nx.algorithms.approximation.maximum_independent_set(g.graph))


def simple_graph(dt = .001):
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edges_from([(0, 1), (1, 2)])
    psi0 = np.array([[0,0,1,0,0,0,0,0]]).T
    s = State(psi0, 3, is_ket=True)
    times = np.arange(0, 3, dt)
    hamiltonian = HamiltonianHeisenberg(graph)
    se = schrodinger_equation.SchrodingerEquation(hamiltonians=[hamiltonian], is_ket=True)
    output = np.squeeze(se.run_ode_solver(s.state, 0,  3, dt), axis=-1)
    overlap = np.abs(output@psi0.conj())
    plt.plot(times, overlap)
    plt.show()

def simple_stochastic(dt=0.001):
    """Simple stochastic integrator with no Hamiltonian evolution"""
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edges_from([(0, 1), (1, 2)])
    graph = Graph(graph)
    psi0 = np.array([[0, 1, 0, 0, 0, 0, 0, 0]]).T
    s = State(psi0, 3, is_ket=True)
    hamiltonian = GreedyNoiseTwoLocal(graph, rate = 1)
    sw = StochasticWavefunction(hamiltonians=[hamiltonian], jumps=[hamiltonian])
    output = sw.run(s.state, 0, 3, dt)
    print(output)


def simple_stochastic_heisenberg(dt=0.001):
    """Simple stochastic integrator with no Hamiltonian evolution"""
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edges_from([(0, 1), (1, 2)])
    graph = Graph(graph)
    tf = 3
    times = np.arange(0, tf, dt)
    psi0 = np.array([[0, 1, 0, 0, 0, 0, 0, 0]]).T
    s = State(psi0, 3, is_ket=True)
    greedy = GreedyNoiseTwoLocal(graph, rate = 1)
    heisenberg = HamiltonianHeisenberg(graph, k=10)
    sw = StochasticWavefunction(hamiltonians=[heisenberg, greedy], jumps=[greedy])
    output = sw.run(s.state, 0, tf, dt)
    overlap_mis = np.abs(np.squeeze(output, axis=-1)@np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).T)
    overlap_violators = np.abs(np.squeeze(output, axis=-1)@(np.array([[1, 1, 0, 1, 0, 0, 0, 0]]).T))
    plt.plot(times, overlap_mis)
    plt.plot(times, overlap_violators)
    plt.show()

def simple_greedy_heisenberg(dt=0.001):
    """Simple stochastic integrator with no Hamiltonian evolution"""
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4], weight = 1)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)], weight = -1)
    graph = Graph(g)
    tf = 3
    times = np.arange(0, tf, dt)
    psi0 = np.zeros((2**graph.n, 1))
    psi0[-1] = 1
    s = State(psi0, graph.n, is_ket=True)
    greedy = GreedyNoise(graph, rate = 10)
    heisenberg = HamiltonianHeisenberg(graph, k=100)
    mis_hamiltonian = hamiltonian.HamiltonianC(g)
    sw = StochasticWavefunction(hamiltonians=[heisenberg, greedy], jumps=[greedy])
    output = sw.run(s.state, 0, tf, dt)
    mis = np.zeros((2**graph.n, 1))
    mis[10] = 1
    overlap_mis = np.abs(np.squeeze(output, axis=-1)@mis)
    """overlap_violators = np.abs(np.squeeze(output, axis=-1)@(np.array([[1, 1, 0, 1, 0, 0, 0,0]]).T))
    overlap_start = np.abs(np.squeeze(output, axis=-1)@(np.array([[0,0, 0, 0, 0, 0, 0,1]]).T))
    overlap_1 = np.abs(np.squeeze(output, axis=-1)@np.array([[0, 0, 0, 1, 0, 0, 0, 0]]).T)
    overlap_2 = np.abs(np.squeeze(output, axis=-1)@np.array([[0, 0, 0, 0, 0, 0, 1, 0]]).T)
    overlap_3 = np.abs(np.squeeze(output, axis=-1)@np.array([[0, 0, 0, 0, 0, 1, 0, 0]]).T)"""
    #m = np.array([[3, 2, 2, 1, 2, 1, 1, 0]]).T
    spin = np.abs(np.squeeze(output, axis=-1))**2 @ mis_hamiltonian.hamiltonian_diag
    plt.plot(times, overlap_mis, label = 'mis')
    plt.plot(times, spin, label='spin')
    """plt.plot(times, overlap_2, label = '011')
    plt.plot(times, overlap_1, label = '110')
    plt.plot(times, overlap_3, label = '101')
    plt.plot(times, overlap_violators, label = 'violators')
    plt.plot(times, overlap_start, label = 'start')"""
    plt.legend(loc='upper left')
    plt.show()


simple_greedy_heisenberg()