import networkx as nx
from qsim.evolution import hamiltonian
from qsim.graph_algorithms import qaoa
import numpy as np
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph
from qsim import tools
from qsim.codes.quantum_state import State


# Construct a known graph
def make_ring_graph_multiring(n):
    assert n % 2 == 0 and n % 3 == 0
    graph_dict = {x: {(x + i) % n: {'weight': 1} for i in [-1, 1]} for x in range(n)}

    for i in range(int(n)):
        if i % 3 == 0:
            graph_dict[i][(i + int(n / 2) + 1) % n] = {'weight': 1}
            graph_dict[(i + int(n / 2) + 1) % n][i] = {'weight': 1}
            graph_dict[i][(i + int(n / 2) - 1) % n] = {'weight': 1}
            graph_dict[(i + int(n / 2) - 1) % n][i] = {'weight': 1}
        else:
            graph_dict[i][(i + int(n / 2)) % n] = {'weight': 1}
            graph_dict[(i + int(n / 2)) % n][i] = {'weight': 1}
    graph = nx.to_networkx_graph(graph_dict)
    return Graph(graph)

def lattice(n=2):
    graph = nx.Graph()
    if n == 1:
        graph.add_node(0)
    else:
        for i in range(n):
            for j in range(n):
                graph.add_edge((j + 1) * i, (j + 1) * ((i + 1) % n), weight=1)
                graph.add_edge((j + 1) * i, (j + 2) * (i % n), weight=1)


def defect_ring(n=2, uniform=True):
    # Assert n is even and greater than 2
    assert n % 4 == 0 and n > 2
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(i, (i + 1) % n, 2) for i in range(n)])
    graph.add_weighted_edges_from(
        [(2 * i, (2 * i + int(n / 2)) % (n), 2) for i in range(1, int(n / 4))])
    return [graph, np.floor(n / 2)]


def node_defect_torus(y, x, return_mis=False):
    graph = nx.grid_2d_graph(x, y, periodic=True)
    # Remove 3 of four corners
    for node in graph.nodes:
        graph.nodes[node]['weight'] = 1
    graph.nodes[(0, 0)]['weight'] = 2
    #print(graph.nodes)
    # graph.remove_node((0, 0))
    nodes = graph.nodes
    new_nodes = list(range(len(nodes)))
    mapping = dict(zip(nodes, new_nodes))
    nx.relabel_nodes(graph, mapping, copy=False)
    if return_mis:
        return Graph(graph), x * y / 2 - 1
    else:
        return Graph(graph)


def special_defect_ring(n=2, uniform=True):
    # Assert n is even and greater than 2
    assert n % 4 == 0 and n > 2
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(i, (i + 1) % n, 2) for i in range(n)])
    graph.add_weighted_edges_from(
        [(2 * i, (2 * i + int(n / 2)) % (n), 2) for i in range(int(n / 4))])

    return [graph, np.floor(n / 2)]


def defect_ring_imperfect(n=2, uniform=True):
    # Assert n is even and greater than 2
    assert n % 4 == 0 and n > 2
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(i, (i + 1) % n, 2) for i in range(n)])
    graph.add_weighted_edges_from(
        [(2 * i, (2 * i + int(n / 2)) % (n), 2) for i in range(1, int(n / 4))])
    return [graph, np.floor(n / 2)]


def defect_ring_nearby(n=2, uniform=True):
    # Assert n is even and greater than 2
    assert n % 2 == 0 and n > 2
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(i, (i + 1) % n, 1) for i in range(n)])
    graph.add_weighted_edges_from(
        [(2 * i, (2 * i + 2) % (n), 1) for i in range(int(n / 2))])
    return [graph, np.floor(n / 2)]


def defect_ring_nearby_uniform_degree(n=5, uniform=True):
    # Assert n is even and greater than 2
    assert (n - 1) % 2 == 0 and n > 5
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(i, (i + 1) % (n - 1), 2) for i in range(n - 1)])
    graph.add_weighted_edges_from(
        [(2 * i, (2 * i + 2) % (n - 1), 2) for i in range(int((n - 1) / 2))])
    # Add big boi node in the middle
    graph.add_node(n - 1)
    # Connect odd nodes to it
    graph.add_weighted_edges_from(
        [(2 * i + 1, n - 1, 2) for i in range(int((n - 1) / 2))])
    return [graph, np.floor(n / 2)]


def ring(n=2):
    graph = nx.Graph()
    if n == 1:
        graph.add_node(0)
    else:
        graph.add_weighted_edges_from(
            [(i, (i + 1) % n, 1) for i in range(n)])
    return [graph, np.floor(n / 2)]


def double_ring(n=2):
    graph = nx.Graph()
    # First, make a ring
    if n == 1:
        graph.add_node(0)
    else:
        graph.add_weighted_edges_from(
            [(i, (i + 1) % n, 1) for i in range(n)])
    # Then, add a second ring
    if n == 1:
        graph.add_node(1)
        graph.add_edge(0, 1)
    else:
        graph.add_weighted_edges_from(
            [(i + n, (i + n + 1) % (2 * n), 1) for i in range(n)])
    # Connect the rings
    graph.add_edge(0, n - 1, weight=1)
    if n != 1:
        graph.add_weighted_edges_from(
            [(i, i + n, 1) for i in range(n)])
    return [graph, np.floor(n / 2)]


def counterintuitive_graph(n=6):
    graph = nx.complete_graph(n)
    # First, make a ring
    assert n % 2 == 0 and n >= 6
    graph.add_weighted_edges_from([(n, i, 2) for i in range(n)])
    graph.add_weighted_edges_from([(n + 1, i, 2) for i in range(n)])
    return [graph, 2]


def crossy_double_ring(n=2):
    graph = nx.Graph()
    # First, make a ring
    if n == 1:
        graph.add_node(0)
    else:
        graph.add_weighted_edges_from(
            [(i, (i + 1) % n, 1) for i in range(n)])
    # Then, add a second ring
    if n == 1:
        graph.add_node(1)
        graph.add_edge(0, 1)
    else:
        graph.add_weighted_edges_from(
            [(i + n, (i + n + 1) % (2 * n), 1) for i in range(n)])
        graph.add_edge(n, 2 * n - 1, weight=1)
    # Connect the rings
    graph.add_edge(0, n - 1, weight=1)
    if n != 1:
        graph.add_weighted_edges_from(
            [(i, i + n, 1) for i in range(n)])
    # Add crossy terms
    if n != 1:
        graph.add_weighted_edges_from(
            [(i, (i + 1) % n + n, 1) for i in range(n)])
        graph.add_weighted_edges_from(
            [(i, (i - 1) % n + n, 1) for i in range(n)])
    return [graph, np.floor(n / 2)]


def mis_qaoa(n, method='minimize', show=True, analytic_gradient=True):
    penalty = 1
    psi0 = tools.equal_superposition(n)
    psi0 = State(psi0)
    G = make_ring_graph_multiring(n)
    if show:
        nx.draw_networkx(G)
        plt.show()

    depths = [2 * i for i in range(1, 2 * n + 1)]
    mis = []
    # Find MIS optimum
    # Uncomment to visualize graph
    hc_qubit = hamiltonian.HamiltonianMIS(G, energies=[1, penalty])
    cost = hamiltonian.HamiltonianMIS(G, energies=[1, penalty])
    # Set the default variational operators
    hb_qubit = hamiltonian.HamiltonianDriver()
    # Create Hamiltonian list
    sim = qaoa.SimulateQAOA(G, cost_hamiltonian=cost, hamiltonian=[], noise_model=None)
    sim.hamiltonian = []
    for p in depths:
        sim.hamiltonian.append(hc_qubit)
        sim.hamiltonian.append(hb_qubit)
        sim.depth = p
        # You should get the same thing
        print(p)
        if method == 'minimize':
            results = sim.find_parameters_minimize(verbose=True, initial_state=psi0,
                                                   analytic_gradient=analytic_gradient)

            approximation_ratio = np.real(results['approximation_ratio'])
            mis.append(approximation_ratio)
        if method == 'brute':
            results = sim.find_parameters_brute(n=15, verbose=True, initial_state=psi0)
            approximation_ratio = np.real(results['approximation_ratio'])
            mis.append(approximation_ratio)
        if method == 'basinhopping':
            if p >= 10:
                results = sim.find_parameters_basinhopping(verbose=True, initial_state=psi0, n=250,
                                                           analytic_gradient=analytic_gradient)
                print(results)
                approximation_ratio = np.real(results['approximation_ratio'])
                mis.append(approximation_ratio)

    # plt.plot(list(range(n)), maxcut, c='y', label='maxcut')
    print(mis)
    plt.plot(depths, [(i + 1) / (i + 2) for i in depths])
    plt.scatter(depths, [i / (i + 1) for i in depths], label='maxcut')
    plt.scatter(depths, mis, label='mis with $n=$' + str(n))
    plt.plot(depths, mis)

    plt.legend()
    if show:
        plt.show()


# mis_qaoa(8, show=True, method='basinhopping', analytic_gradient=False)
mis_qaoa(12, show=False, method='basinhopping', analytic_gradient=False)

# mis_qaoa(16, show=False, method='basinhopping', analytic_gradient=False)
