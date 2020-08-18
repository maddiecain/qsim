from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
import time
import networkx as nx
from qsim.evolution import hamiltonian
from qsim.graph_algorithms import qaoa
import numpy as np
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph
from qsim import tools
from qsim.codes.quantum_state import State
from qsim.tools.tools import equal_superposition


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
    for node in graph.nodes:
        graph.nodes[node]['weight'] = 1
    graph.nodes[(0, 0)]['weight'] = 2
    print(graph.nodes)
    # graph.remove_node((0, 0))
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


def adiabatic_simulation(graph, show_graph=False, IS_subspace=True):
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    laser = hamiltonian.HamiltonianDriver(IS_subspace=IS_subspace, graph=graph)
    detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=IS_subspace)
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=IS_subspace)
    graph.mis_size = int(np.max(rydberg_hamiltonian_cost.hamiltonian))
    print('Degeneracy',
          len(np.argwhere(rydberg_hamiltonian_cost.hamiltonian == np.max(rydberg_hamiltonian_cost.hamiltonian)).T[0]))
    # Initialize adiabatic algorithm
    simulation = SimulateAdiabatic(graph, hamiltonian=[laser, detuning], cost_hamiltonian=rydberg_hamiltonian_cost,
                                   IS_subspace=IS_subspace)
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


def mis_qaoa(n, method='basinhopping', show=False, analytic_gradient=True, IS_subspace=True):
    penalty = 1
    if not IS_subspace:
        psi0 = tools.equal_superposition(n * n)
        psi0 = State(psi0)
    else:
        psi0 = None
    G = node_defect_torus(n, n)
    if show:
        nx.draw_networkx(G)
        plt.show()

    depths = [2 * i for i in range(1, n * n + 1)]
    mis = []
    # Find MIS optimum
    # Uncomment to visualize graph
    hc_qubit = hamiltonian.HamiltonianMIS(G, energies=[1, penalty], IS_subspace=IS_subspace)
    cost = hamiltonian.HamiltonianMIS(G, energies=[1, penalty], IS_subspace=IS_subspace)
    # Set the default variational operators
    hb_qubit = hamiltonian.HamiltonianDriver(graph=G, IS_subspace=IS_subspace)
    # Create Hamiltonian list
    sim = qaoa.SimulateQAOA(G, cost_hamiltonian=cost, hamiltonian=[], noise_model=None)
    sim.hamiltonian = []
    for p in depths:
        sim.hamiltonian.append(hc_qubit)
        sim.hamiltonian.append(hb_qubit)
        sim.depth = p
        # You should get the same thing
        print(p)
        if method == 'basinhopping':
            results = sim.find_parameters_basinhopping(verbose=True, initial_state=psi0, n=50,
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


if __name__ == "__main__":
    i, j = 4, 4
    graph = node_defect_torus(i, j)
    #simulation = adiabatic_simulation(graph, IS_subspace=True)
    #t0 = time.time()

    #res = simulation.performance_vs_total_time(np.arange(0, 0), metric='optimum_overlap',
    #                                           schedule=lambda t, tf: experiment_rydberg_MIS_schedule(t, tf, simulation,
    #                                                                                                  coefficients=[10,
    #                                                                                                                10]),
    #                                           plot=True, verbose=True, method='trotterize')

    #print('results: ', res, flush=True)
    #simulation = mis_qaoa(4, IS_subspace=True, method='basinhopping')
    simulation = adiabatic_simulation(graph, IS_subspace=False)
    res = simulation.performance_vs_total_time(np.arange(40, 95, 5), metric='optimum_overlap',
                                               initial_state=State(equal_superposition(i * j)),
                                               schedule=lambda t, tf: simulation.linear_schedule(t, tf,
                                                                                                 coefficients=[1, 1]),
                                               plot=True, verbose=True, method='trotterize')
    print('results: ', res, flush=True)
