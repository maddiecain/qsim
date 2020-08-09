import numpy as np
from qsim.tools import tools
from qsim.codes import qubit
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import approximation
from itertools import tee, product

"""Class for performing basic graph Monte Carlo operations on networkx Graphs. Future versions of this code
may subclass nx.Graph."""


class Graph(object):
    def __init__(self, graph: nx.Graph):
        # Set default weights to one
        for edge in graph.edges:
            if not ('weight' in graph.edges[edge]):
                graph.edges[edge]['weight'] = 1
        for node in graph.nodes:
            if not ('weight' in graph.nodes[node]):
                graph.nodes[node]['weight'] = 1
        self.graph = graph
        # Nodes are assumed to be integers from zero to # nodes - 1
        self.nodes = np.array([n for n in self.graph], dtype=int)
        self.n = self.nodes.size
        self.edges = np.array([m for m in self.graph.edges])
        self.m = self.edges.size
        # TODO: figure out if you really want to store the binary data, or that of a designated code
        # Initialize attributes to be set in self.generate_independent_sets()
        self.num_independent_sets = None
        self.independent_sets = None
        self.binary_to_index = None
        self.mis_size = None
        # Populate initialized attributes
        self.generate_independent_sets()

    def generate_independent_sets(self):
        # Construct generator containing independent sets
        # Don't generate anything that depends on the entire Hilbert space as to save space
        # Generate complement graph
        complement = nx.complement(self.graph)
        # These are your independent sets of the original graphs, ordered by node and size
        independent_sets, backup = tee(nx.algorithms.clique.enumerate_all_cliques(complement))
        self.num_independent_sets = sum(1 for _ in backup) + 1  # We add one to include the empty set
        # Generate a list of integers corresponding to the independent sets in binary
        indices = np.zeros(self.num_independent_sets, dtype=int)
        indices[-1] = 2 ** self.n - 1
        k = self.num_independent_sets - 2
        self.mis_size = 0
        IS = dict.fromkeys(np.arange(self.num_independent_sets))
        # All spins down should be at the end
        IS[self.num_independent_sets - 1] = (2 ** self.n - 1, 0, np.ones(self.n, dtype=int))
        for i in independent_sets:
            indices[k] = 2 ** self.n - sum(2 ** j for j in i) - 1
            IS[k] = (indices[k], len(i), tools.int_to_nary(indices[k], size=self.n))
            if len(i) > self.mis_size:
                self.mis_size = len(i)
            k -= 1
        binary_to_index = dict.fromkeys(indices)
        for j in range(self.num_independent_sets):
            binary_to_index[indices[j]] = j
        self.binary_to_index = binary_to_index
        self.independent_sets = IS
        return IS, binary_to_index, self.num_independent_sets

    def independent_sets_code(self, code):
        assert not code is qubit
        # You do NOT need to count the number in ground, this is just n-# excited
        # Count the number of elements in the ground space and map to their representation in ternary
        # Determine how large to make the array
        num_IS = 0
        for i in self.independent_sets:
            # Count the number of ones
            num_ground = self.n - self.independent_sets[i][1]
            num_IS += (code.d - 1) ** num_ground
        # Now, we need to generate a way to map between the restricted indices and the original
        # indices
        IS = dict.fromkeys(np.arange(num_IS))
        indices = np.zeros(num_IS, dtype=int)
        nary_reprs = np.zeros((num_IS, self.n))
        IS_sizes = np.zeros(num_IS, dtype=int)
        counter = 0
        # TODO: deal with the all one's string
        for i in self.independent_sets:
            # Generate binary representation of IS
            IS_index, IS_size, IS_binary = self.independent_sets[i]
            # Count the number of ones and generate all combinations of 1's and 2's
            num_ground = self.n - IS_size
            where_excited = np.where(IS_binary == 0)[0]
            # noinspection PyTypeChecker
            states_generator = product(range(1, code.d), repeat=num_ground)
            for j in states_generator:
                ind = 0
                bit_string = np.zeros(self.n)
                # Construct the bit string itself
                for k in range(self.n):
                    if np.any(where_excited == k):
                        pass
                    else:
                        bit_string[k] = j[ind]
                        ind += 1
                nary_reprs[counter, :] = bit_string
                IS_sizes[counter] = IS_size
                indices[counter] = tools.nary_to_int(bit_string, base=code.d)
                counter += 1
        # Get the sorted order for the indices
        order = np.argsort(indices)
        # Finally, populate the dictionary
        for i in IS:
            IS[i] = (indices[order[i]], IS_sizes[order[i]], nary_reprs[order[i]])
        # Generate conversion dictionary
        indices = indices[order]
        nary_to_index = dict.fromkeys(indices)
        counter = 0
        for i in nary_to_index:
            nary_to_index[i] = counter
            counter += 1
        return IS, nary_to_index, num_IS


class GraphMonteCarlo(object):
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        # Nodes are assumed to be integers from zero to # nodes - 1
        self.nodes = np.array([n for n in self.graph], dtype=int)
        self.n = self.nodes.size
        self.edges = np.array([m for m in self.graph.edges])
        self.m = self.edges.size

    def random_node(self):
        """Return a random node"""
        return np.random.randint(0, self.n)

    def random_edge(self):
        """Return a random edge"""
        return self.edges[np.random.randint(0, self.m)]

    def raised_neighbors(self, configuration: np.array, i):
        """Returns the raised neighbors of node i."""
        raised = []
        for neighbor in self.graph.neighbors(i):
            if configuration[neighbor] == 1:
                raised.append(neighbor)
        return raised

    def free_node(self, configuration: np.array, i, ignore=None):
        """Checks if node i is free, ignoring some neighbors if indicated. If free, return True; otherwise,
        return False. ignore is assumed to be a list of nodes."""
        if ignore is None:
            ignore = []
        if configuration[i] == 0:
            for neighbor in list(self.graph.neighbors(i)):
                if neighbor not in ignore:
                    if self.configuration[neighbor] == 1:
                        return False
            return True
        return False

    def free_nodes(self, configuration: np.array, ignore=None):
        """Return the free nodes in the graph."""
        free = []
        for j in range(self.n):
            if self.free_node(configuration, j, ignore=ignore):
                free.append(j)
        return free

    def raise_node(self, configuration: np.array, i, ignore=None):
        """Check if node i is free. If it is, change its value to 1 with probability p. Return True if the node has
        been raised."""
        if configuration[i] == 1:
            # Already raised
            return True
        else:
            if self.free_node(configuration, i, ignore=ignore):
                configuration[i] = 1
                return True
            return False

    def flip_flop(self, configuration: np.array, i):
        """If i is spin up, flip it if there exists a neighbor of i chosen at random satisfying the condition that
        the node is free if i is ignored. If i is spin down, and there is exactly one neighbor of i that is spin up,
        swap the two. Return True if a flip flop has taken place, and False otherwise."""
        if self.configuration[i] == 1:
            shuffed_neighbors = np.array(self.graph.neighbors(i))
            np.random.shuffle(shuffed_neighbors)
            for neighbor in shuffed_neighbors:
                if self.free_node(configuration, neighbor, ignore=[i]):
                    configuration[i] = 0
                    configuration[neighbor] = 1
                    return True
        elif configuration[i] == 0:
            raised_neighbors = self.raised_neighbors(configuration, i)
            if len(raised_neighbors) == 1:
                configuration[i] = 1
                configuration[raised_neighbors[0]] = 0
                return True
        return False

    def spin_exchange(self, configuration: np.array, i):
        """Returns True if and only if a node i can spin exchange."""
        if configuration[i] == 1:
            for neighbor in self.graph.neighbors(i):
                if self.free_node(configuration, neighbor, ignore=[i]):
                    return True
        elif configuration[i] == 0:
            raised_neighbors = self.raised_neighbors(configuration, i)
            if len(raised_neighbors) == 1:
                return True
        return False

    def spin_exchanges(self, configuration: np.array):
        """Returns a list of pairs of nodes which can spin exchange."""
        exchanges = []
        for i in self.nodes:
            if configuration[i] == 1:
                for neighbor in self.graph.neighbors(i):
                    if self.free_node(configuration, neighbor, ignore=[i]):
                        exchanges.append((i, neighbor))
        return exchanges

    def random_spin_exchange(self, configuration: np.array):
        """Spin exchange on a random spin pair eligible to spin exchange."""
        exchanges = self.spin_exchanges(configuration)
        to_exchange = exchanges[np.random.randint(0, len(exchanges))]
        # Perform the spin exchange
        configuration[[to_exchange[0], to_exchange[1]]] = configuration[[to_exchange[1], to_exchange[0]]]
        return configuration

    def random_raise(self, configuration: np.array):
        """Raise a random free node"""
        free_nodes = self.free_nodes(configuration)
        to_raise = free_nodes[np.random.randint(0, len(free_nodes))]
        configuration[to_raise] = 1
        return configuration

    def spin_exchange_monte_carlo(self, t0, tf, num=50, rates=None, configuration=None):
        """Run a Monte Carlo algorithm for time tf-t0 with given spin exchange and greedy spin raise rates."""
        # TODO: generalize this for general Monte Carlo operations
        if configuration is None:
            configuration = np.zeros(self.n, dtype=int)
        if rates is None:
            rates = [1, 1]
        times = np.linspace(t0, tf, num=num)
        dt = (tf - t0) / num
        outputs = np.zeros((times.shape[0], configuration.shape[0], 1), dtype=np.complex128)
        for (j, time) in zip(range(times.shape[0]), times):
            if j == 0:
                outputs[0, :] = np.array([configuration.copy()]).T
            if j != 0:
                # Compute the probability that something happens
                # Probability of spin exchange
                probability_exchange = dt * rates[0] * len(
                    self.spin_exchanges(configuration))  # TODO: figure out if this is over 2
                # Probability of raising
                probability_raise = dt * rates[1] * len(self.free_nodes(configuration))
                probability = probability_exchange + probability_raise
                if np.random.uniform() < probability:
                    if np.random.uniform(0, probability) < probability_exchange:
                        # Do a spin exchange
                        configuration = self.random_spin_exchange(configuration)
                    else:
                        # Raise a node
                        configuration = self.random_raise(configuration)
                outputs[j, :] = np.array([configuration.copy()]).T
        return outputs

    def draw_configuration(self, configuration):
        """Spin up is white, spin down is black."""
        # First, color the nodes
        color_map = []
        for node in self.graph:
            if configuration[node] == 1:
                color_map.append('teal')
            if configuration[node] == 0:
                color_map.append('black')
        nx.draw_circular(self.graph, node_color=color_map)
        plt.show()

    def configuration_weight(self, configuration):
        weight = np.sum(configuration)
        print('Configuration weight: ' + str(weight))
        return weight

    def nx_mis(self):
        return nx.algorithms.approximation.maximum_independent_set(self.graph)


def line_graph(n, return_mis=False):
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, n), weight=1)
    if n == 1:
        if return_mis:
            return nx.trivial_graph(), 1
        else:
            return nx.trivial_graph()
    else:
        for i in range(n - 1):
            g.add_edge(i, i + 1, weight=-1)
    if return_mis:
        return g, np.ceil(n/2)
    return g


def ring_graph(n, node_weight=1, edge_weight=1, return_mis=False):
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, n), weight=node_weight)
    if n == 1:
        return nx.trivial_graph()
    else:
        for i in range(n - 1):
            g.add_edge(i, i + 1, weight=edge_weight)
        g.add_edge(0, n - 1, weight=edge_weight)
    if return_mis:
        return g, np.floor(n/2)
    return g


def degree_fails_graph(return_mis = False):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 4, 1), (0, 5, 1), (4, 5, 1), (1, 4, 1), (1, 3, 1), (2, 4, 1)])
    if return_mis:
        return graph, 3
    return graph


def IS_projector(graph, code):
    """Returns a projector (represented as a column vector or matrix) into the space of independent sets for
    general codes."""
    n = graph.n
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
        # TODO: make this output a sparse matrix
        proj = np.identity(code.d ** n)
        for i, j in graph.edges:
            if i > j:
                # Requires i < j
                temp = i
                i = j
                j = temp
            temp = tools.tensor_product(
                [tools.identity(i, d=code.d), code.U, tools.identity(j - i - 1, d=code.d), code.U,
                 tools.identity(n - j - 1, d=code.d)])
            proj = proj @ (np.identity(code.d ** n) - temp)
        return np.array([np.diagonal(proj)]).T
