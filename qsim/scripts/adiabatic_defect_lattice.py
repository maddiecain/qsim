import sys
import networkx as nx
from qsim.evolution import hamiltonian
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
import numpy as np
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph


def node_removed_torus(y, x, return_mis=False):
    graph = nx.grid_2d_graph(x, y, periodic=True)
    # Remove 3 of four corners
    graph.remove_node((0, 0))
    graph.add_edge((1, 0), (y - 1, 0), weight=1)
    graph.add_edge((1, 0), (1, x - 1), weight=1)
    graph.add_edge((0, 1), (0, x - 1), weight=1)
    graph.add_edge((0, 1), (y - 1, 1), weight=1)
    nodes = graph.nodes
    new_nodes = list(range(len(nodes)))
    mapping = dict(zip(nodes, new_nodes))
    nx.relabel_nodes(graph, mapping, copy=False)
    if return_mis:
        return Graph(graph), x * y / 2 - 1
    else:
        return Graph(graph)


def node_defect_torus(y, x, return_mis=False):
    graph = nx.grid_2d_graph(x, y, periodic=True)
    # Remove 3 of four corners
    graph.nodes[(0, 0)]['weight'] = 2
    nodes = graph.nodes
    new_nodes = list(range(len(nodes)))
    mapping = dict(zip(nodes, new_nodes))
    nx.relabel_nodes(graph, mapping, copy=False)
    if return_mis:
        return Graph(graph), x * y / 2 - 1
    else:
        return Graph(graph)


def defect_crossy_lattice(y, x):
    assert y % 2 == 1 and x % 2 == 1
    graph = nx.grid_2d_graph(x, y, periodic=True)
    # Remove 3 of four corners
    for i in range(y - 1):
        for j in range(x - 1):
            graph.add_edge((i, j), (i + 1, j + 1), weight=1)
            graph.add_edge((i + 1, j), (i, j + 1), weight=1)
    graph.remove_node((0, 0))
    nodes = graph.nodes
    new_nodes = list(range(len(nodes)))
    mapping = dict(zip(nodes, new_nodes))
    nx.relabel_nodes(graph, mapping, copy=False)
    return graph, (x + 1) * (y + 1) / 4 - 1


def adiabatic_simulation(graph, show_graph=False):
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    laser = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    graph.mis_size = int(np.max(rydberg_hamiltonian_cost.hamiltonian))
    print('degeneracy',
          len(np.argwhere(rydberg_hamiltonian_cost.hamiltonian == np.max(rydberg_hamiltonian_cost.hamiltonian)).T[0]))
    # Initialize adiabatic algorithm
    simulation = SimulateAdiabatic(graph, hamiltonian=[laser, detuning], cost_hamiltonian=rydberg_hamiltonian_cost,
                                   IS_subspace=True)
    return simulation


def experiment_rydberg_MIS_schedule(t, tf, simulation, coefficients=None):
    if coefficients is None:
        coefficients = [1, 1]
    for i in range(len(simulation.hamiltonian)):
        if i == 0:
            # We don't want to update normal detunings
            simulation.hamiltonian[i].energies = [coefficients[0] * np.sin(np.pi * t / tf) ** 2]
        if i == 1:
            simulation.hamiltonian[i].energies = [coefficients[1] * (2 * t / tf - 1)]
    return True


if __name__ == "__main__":
    i, j = 2, 2
    graph = node_defect_torus(i, j)
    simulation = adiabatic_simulation(graph)
    res = simulation.performance_vs_total_time(np.arange(5, 10, 1), metric='optimum_overlap',
                                               schedule=lambda t, tf: experiment_rydberg_MIS_schedule(t, tf, simulation,
                                                                                                      coefficients=[10,
                                                                                                                    10]),
                                               plot=True, verbose=True, method='odeint')
    print('results: ', res, flush=True)
