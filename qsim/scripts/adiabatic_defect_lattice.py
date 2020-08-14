import networkx as nx
from qsim.evolution import hamiltonian
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
import numpy as np
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph
from qsim.codes.quantum_state import State


def defect_torus(y, x):
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
    return graph, x * y / 2 - 1


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
    print(np.argwhere(rydberg_hamiltonian_cost.hamiltonian == np.max(rydberg_hamiltonian_cost.hamiltonian)),
          np.max(rydberg_hamiltonian_cost.hamiltonian))
    # Initialize adiabatic algorithm
    simulation = SimulateAdiabatic(graph, hamiltonian=[laser, detuning], cost_hamiltonian=rydberg_hamiltonian_cost,
                                   IS_subspace=True)
    return simulation


def experiment_rydberg_MIS_schedule(t, tf, simulation, coefficients=None):
    if coefficients is None:
        coefficients = [1, 1, 1]
    for i in range(len(simulation.hamiltonian)):
        if i == 0:
            # We don't want to update normal detunings
            simulation.hamiltonian[i].energies = [coefficients[0] * np.sin(np.pi * t / tf) ** 2]
        if i == 1:
            simulation.hamiltonian[i].energies = [coefficients[1] * (2 * t / tf - 1)]
    return True


graph, mis = defect_torus(4, 6)
graph = Graph(graph)
graph.mis_size = mis
simulation = adiabatic_simulation(graph)
res = simulation.ratio_vs_total_time(np.arange(15, 55),
                                     schedule=lambda t, tf: experiment_rydberg_MIS_schedule(t, tf, simulation,
                                                                                            coefficients=[1, 1]),
                                     plot=True)
print('results: ', res)
