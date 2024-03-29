import numpy as np
from qsim.tools import tools
from qsim.codes import qubit
import matplotlib.pyplot as plt
import networkx as nx
from itertools import tee, product
from collections import deque
from itertools import chain, islice


def enumerate_independent_sets(graph):
    graph = nx.complement(graph)
    index = {}
    nbrs = {}
    for u in graph:
        index[u] = len(index)
        # Neighbors of u that appear after u in the iteration order of G.
        nbrs[u] = {v for v in graph[u] if v not in index}
    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in graph)
    # Loop invariants:
    # 1. len(base) is nondecreasing.
    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
    # 3. cnbrs is a set of common neighbors of nodes in base.
    while queue:
        base, cnbrs = map(list, queue.popleft())
        yield base
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]), filter(nbrs[u].__contains__, islice(cnbrs, i + 1, None))))


class Graph(object):
    def __init__(self, graph: nx.Graph):
        # Initialize edge and node weights to one
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
        # Initialize attributes to be set in self.generate_independent_sets() or when generating mixer and
        # driver Hamiltonians.

        self._num_independent_sets = None
        self._degeneracy = None
        self._independent_sets = None

    @property
    def degeneracy(self):
        num_independent_sets = 1
        mis_size = 0
        degeneracy = 0
        if self._degeneracy is None:
            for independent_set in enumerate_independent_sets(self.graph):
                num_independent_sets += 1
                length = len(independent_set)
                if length > mis_size:
                    mis_size = length
                    degeneracy = 0
                if length == mis_size:
                    degeneracy += 1
            self._degeneracy = degeneracy
            self._mis_size = mis_size
            self.num_independent_sets = num_independent_sets
            return self._degeneracy
        else:
            return self._degeneracy

    @property
    def mis_size(self):
        num_independent_sets = 1
        mis_size = 0
        degeneracy = 0
        if self._mis_size is None:
            for independent_set in enumerate_independent_sets(self.graph):
                num_independent_sets += 1
                length = len(independent_set)
                if length > mis_size:
                    mis_size = length
                    degeneracy = 0
                if length == mis_size:
                    degeneracy += 1
            self._degeneracy = degeneracy
            self._mis_size = mis_size
            self.num_independent_sets = num_independent_sets
            return self._mis_size
        else:
            return self._mis_size

    @property
    def num_independent_sets(self):
        num_independent_sets = 1
        mis_size = 0
        degeneracy = 0
        if self._num_independent_sets is None:
            for independent_set in enumerate_independent_sets(self.graph):
                num_independent_sets += 1
                length = len(independent_set)
                if length > mis_size:
                    mis_size = length
                    degeneracy = 0
                if length == mis_size:
                    degeneracy += 1
            self._degeneracy = degeneracy
            self._mis_size = mis_size
            self._num_independent_sets = num_independent_sets
            return self._num_independent_sets
        else:
            return self._num_independent_sets

    @property
    def independent_sets(self):
        if self._independent_sets is None:
            # Generate a list of integers corresponding to the independent sets in binary
            # Construct generator containing independent sets
            # Don't generate anything that depends on the entire Hilbert space as to save space
            # Generate complement graph
            # These are your independent sets of the original graphs, ordered by node and size
            independent_sets, backup = tee(enumerate_independent_sets(self.graph))
            # Generate a list of integers corresponding to the independent sets in binary
            indices = np.zeros((self.num_independent_sets, self.n), dtype=int)
            # All ones
            indices[-1, ...] = np.ones(self.n)
            k = self.num_independent_sets - 2
            for i in independent_sets:
                nary = np.ones(self.n)
                for node in i:
                    nary[node] = 0
                indices[k, ...] = nary
                k -= 1
            self._independent_sets = indices
            return self._independent_sets
        else:
            return self._independent_sets

    def generate_independent_sets_qudit(self, code):
        assert code is not qubit
        # You do NOT need to count the number in ground, this is just n-# excited
        # Count the number of elements in the ground space and map to their representation in ternary
        # Determine how large to make the array
        num_IS = np.sum((code.d - 1) ** np.sum(self.independent_sets, axis=1))
        # Now, we need to generate a way to map between the restricted indices and the original
        # indices
        indices = np.zeros((num_IS, self.n))
        indices[-1, ...] = np.ones(self.n)
        counter = 0
        for i in self.independent_sets:
            # Generate binary representation of IS
            # Count the number of ones and generate all combinations of 1's and 2's
            where_excited = np.where(i == 0)[0]
            # noinspection PyTypeChecker
            states_generator = product(range(1, code.d), repeat=np.sum(i))
            for j in states_generator:
                ind = 0
                bit_string = np.zeros(self.n)
                # Construct the bit string itself
                for k in range(self.n):
                    if not np.any(where_excited == k):
                        bit_string[k] = j[ind]
                        ind += 1
                indices[counter, :] = bit_string
                counter += 1
        return indices, num_IS


def line_graph(n):
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, n), weight=1)
    if n == 1:
        return Graph(graph)
    else:
        for i in range(n - 1):
            graph.add_edge(i, i + 1, weight=1)
    return Graph(graph)


def ring_graph(n):
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, n))
    if n == 1:
        return Graph(graph)
    else:
        for i in range(n - 1):
            graph.add_edge(i, i + 1)
        graph.add_edge(0, n - 1)
    return Graph(graph)


def degree_fails_graph():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 4, 1), (0, 5, 1), (4, 5, 1), (1, 4, 1), (1, 3, 1), (2, 4, 1)])
    return Graph(graph)


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


def unit_disk_grid_graph(grid, radius=np.sqrt(2) + 1e-5, periodic=False, visualize=False):
    x = grid.shape[1]
    y = grid.shape[0]

    def neighbors_from_geometry(n):
        """Identify the neighbors within a unit distance of the atom at index (i, j) (zero-indexed).
        Returns a numpy array listing both the geometric graph of the neighbors, and the indices of the
        neighbors of the form [[first indices], [second indices]]"""
        # Assert that we actually have an atom at this location
        assert grid[n[0], n[1]] != 0
        grid_x, grid_y = np.meshgrid(np.arange(x), np.arange(y))
        # a is 1 if the location is within a unit distance of (i, j), and zero otherwise
        a = np.sqrt((grid_x - n[1]) ** 2 + (grid_y - n[0]) ** 2) <= radius
        if periodic:
            b = np.sqrt((np.abs(grid_x - n[1]) - x) ** 2 + (grid_y - n[0]) ** 2) <= radius
            a = a + b
            b = np.sqrt((grid_x - n[1]) ** 2 + (np.abs(grid_y - n[0]) - y) ** 2) <= radius
            a = a + b
            b = np.sqrt((np.abs(grid_x - n[1]) - x) ** 2 + (np.abs(grid_y - n[0]) - y) ** 2) <= radius
            a = a + b
        # Remove the node itself
        a[n[0], n[1]] = 0
        # a is 1 if  within a unit distance of (i, j) and a node is at that location, and zero otherwise
        a = a * grid
        return np.argwhere(a != 0)

    nodes_geometric = np.argwhere(grid != 0)
    nodes = list(range(len(nodes_geometric)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    j = 0
    for node in nodes_geometric:
        neighbors = neighbors_from_geometry(node)
        neighbors = [np.argwhere(np.all(nodes_geometric == i, axis=1))[0, 0] for i in neighbors]
        for neighbor in neighbors:
            graph.add_edge(j, neighbor)
        j += 1

    if visualize:
        pos = {nodes[i]: nodes_geometric[i] for i in range(len(nodes))}
        nx.draw_networkx_nodes(graph, pos=pos, node_color='cornflowerblue', node_size=40,
                               edgecolors='black')  # edgecolors='black',
        nx.draw_networkx_edges(graph, pos=pos, edge_color='black')
        plt.axis('off')
        plt.show()
    graph = Graph(graph)
    graph.positions = grid
    graph.radius = radius
    graph.periodic = periodic
    return graph


def unit_disk_grid_graph_rydberg(grid, radius=np.sqrt(2) + 1e-5, B=863300 / 4.47 ** 6, visualize=False):
    y, x = grid.shape

    def neighbors_from_geometry(n):
        """Identify the neighbors within a unit distance of the atom at index (i, j) (zero-indexed).
        Returns a numpy array listing both the geometric graph of the neighbors, and the indices of the
        neighbors of the form [[first indices], [second indices]]"""
        # Assert that we actually have an atom at this location
        assert grid[n[0], n[1]] != 0
        grid_x, grid_y = np.meshgrid(np.arange(x), np.arange(y))
        # a is 1 if the location is within a unit distance of (i, j), and zero otherwise
        a = np.sqrt((grid_x - n[1]) ** 2 + (grid_y - n[0]) ** 2) <= radius
        # TODO: add an option for periodic boundary conditions
        # Remove the node itself
        a[n[0], n[1]] = 0
        # a is 1 if  within a unit distance of (i, j) and a node is at that location, and zero otherwise
        a = a * grid
        return np.argwhere(a != 0)

    nodes_geometric = np.argwhere(grid != 0)
    nodes = list(range(len(nodes_geometric)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    j = 0
    for node in nodes_geometric:
        neighbors = neighbors_from_geometry(node)
        neighbors_geometric = neighbors.copy()
        neighbors = [np.argwhere(np.all(nodes_geometric == i, axis=1))[0, 0] for i in neighbors]
        i = 0
        for neighbor in neighbors:
            graph.add_edge(j, neighbor, weight=B / (
                np.sqrt((node[0] - neighbors_geometric[i][0]) ** 2 + (node[1] - neighbors_geometric[i][1]) ** 2)) ** 6)
            i += 1
        j += 1

    if visualize:
        pos = {nodes[i]: nodes_geometric[i] for i in range(len(nodes))}
        nx.draw_networkx_nodes(graph, pos=pos, node_color='cornflowerblue', node_size=40,
                               edgecolors='black')  # edgecolors='black',
        nx.draw_networkx_edges(graph, pos=pos, edge_color='black')
        plt.axis('off')
        plt.show()
    graph = Graph(graph)
    graph.positions = nodes
    graph.radius = radius
    graph.B = B
    return graph


def rydberg_graph(points, B=863300 / 4.47 ** 6, alpha=6, threshold=1e-8, label_node_by_coords=False, periodic=False,
                  visualize=False):
    """ Create a Graph object from xy-coordinates
    where the edge weights correspond to
            B / ||r_i - r_j||^alpha

    Arguments:
        xy = an n-by-2 matrix of xy-coordinates
        B = Blockade interaction at unit distance
        alpha = power parameter in interaction strength.
                Reduces to unit step function if alpha = float('inf')
        threshold = the minimum value of a edge
                weight to be counted as an edge (default: 1e-8)
        label_node_by_coords = True if the nodes are labelled by their (x,y)
                coordinates, or False if nodes are indexed by integers

    Returns:
        graph = a networkx.Graph
        pos = a position dictionary {node: xy(node)}
    """
    n, d = points.shape
    if d != 2:  # convert to 2D coordinates
        points = np.argwhere(points != 0)
        d = 2
        n = points.shape[0]
    assert d == 2

    def interaction(displacement):
        # If alpha is 'inf', does a hard constraint with weight B
        distnorm = np.linalg.norm(displacement)
        if alpha < float('inf'):
            return B / distnorm ** alpha
        elif distnorm <= 1:
            return B
        else:
            return 0

    if label_node_by_coords:
        nodelist = [(points[i, 0], points[i, 1]) for i in range(n)]
        pos = {v: v for v in nodelist}
    else:
        nodelist = np.arange(n)
        pos = {i: (points[i, 0], points[i, 1]) for i in nodelist}
    graph = nx.Graph()
    graph.add_nodes_from(nodelist)
    for i in range(n - 1):
        for j in range(i + 1, n):
            temp = interaction(points[i] - points[j])
            if temp >= threshold:
                graph.add_edge(nodelist[i], nodelist[j], weight=temp)
    if visualize:
        plt.figure(figsize=(7, 7))
        nx.draw_networkx(graph, pos=pos,
                         labels={}, node_size=20, node_color='r')
        plt.show()
    graph = Graph(graph)
    graph.positions = points
    graph.B = B
    graph.periodic = periodic
    return graph


def unit_disk_graph(points, radius=1 + 1e-5, visualize=False):
    nodes = np.arange(points.shape[0])
    adjacency_matrix = np.zeros((points.shape[0], points.shape[0]))
    for n1 in nodes:
        for n2 in nodes:
            if n2 > n1:
                if np.sqrt((points[n1][0] - points[n2][0]) ** 2 + (points[n1][1] - points[n2][1]) ** 2) <= radius:
                    adjacency_matrix[n1, n2] = 1
                    adjacency_matrix[n2, n1] = 1
    graph = nx.from_numpy_array(adjacency_matrix)
    if visualize:
        pos = {nodes[i]: points[i] for i in range(len(nodes))}
        nx.draw(graph, pos=pos)
        plt.show()
    graph = Graph(graph)
    graph.positions = nodes
    return graph


def branching_tree_from_edge(n_branches, visualize=True):
    graph = nx.Graph()
    graph.add_edge(0, 1)
    last_layer = [0, 1]
    count = 2
    for i in range(len(n_branches)):
        added = []
        for k in range(len(last_layer)):
            for j in range(n_branches[i]):
                graph.add_edge(last_layer[k], count)
                added.append(count)
                count += 1
        last_layer = added
    if visualize:
        nx.draw(graph, with_labels=True)
        plt.show()
    return Graph(graph)
