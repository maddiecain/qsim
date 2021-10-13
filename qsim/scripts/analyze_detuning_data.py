import matplotlib.pyplot as plt
import numpy as np
from qsim.evolution import hamiltonian
from qsim.graph_algorithms.graph import unit_disk_grid_graph, rydberg_graph
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim import schrodinger_equation
import matplotlib.gridspec as gridspec
import scipy.sparse
import scipy.optimize
import pandas as pd
import sys
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.interpolate
from qsim.codes.quantum_state import State

def max_degree(graph):
    """
    :param graph: non-null and non-oriented networkx type graph to analyze.
    :return: list of nodes of this graph with maximum degree; 0 if the graph has no nodes; -1 if the graph is disconnected;
    """
    if len(graph.nodes) == 0:
        return 0
    # graph.degree -> [(<node>, <grade>), ...]
    max_grade = max(graph.degree, key=lambda x: x[1])[1]

    return [node[0] for node in graph.degree if node[1] == max_grade]


def loadfile(graph_index, size):
    graph_mask = np.reshape(np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 3:],
                            (size, size), order='F')[::-1, ::-1].T.astype(bool)
    MIS_size = np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 0].astype(int)
    degeneracy = np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 1].astype(int)
    graph_data = {'graph_index': graph_index, 'MIS_size': MIS_size, 'degeneracy': degeneracy,
                  'side_length': size,
                  'graph_mask': graph_mask, 'number_of_nodes': int(np.sum(graph_mask))}
    return graph_data


def greedy_remove(graph, IScandidate):
    """ Greedily remove IS violations (by descending order of nodes involved
    in most number of violations, breaking ties randomly)

    :param graph: Graph for which the MIS problem is defined
    :param IScandidate: a list of nodes in the graph as a candidate for IS
    :return: a set of nodes corresponding to a valid independent set
    """
    input_graph = graph.subgraph(IScandidate).copy()
    n_removed = 0
    while input_graph.number_of_edges() > 0:
        maxdegnodes = max_degree(input_graph)
        v = random.choice(maxdegnodes)
        input_graph.remove_node(v)
        n_removed += 1

    IS = set(input_graph.nodes)

    return IS


def MIS_probability_finite(state, graph, graph_finite):
    prob = 0
    for i in range(len(state)):
        if np.sum(1-graph_finite.independent_sets[i]) >= graph.mis_size:
            bitstring = greedy_remove(graph.graph, set(list(np.argwhere(graph_finite.independent_sets[i] == 0).T[0])))
            if len(bitstring) == graph.mis_size:
                prob += np.abs(state[i])**2
        else:
            break
    return prob

def plot_7():
    size = 7
    critical_detuning = -8.495#-6.019449429835163###-9.604213726908476#
    critical_detuning = -7.001
    critical_detunings = np.concatenate([-np.linspace(2, 10, 10), [critical_detuning]])
    #graph_index = 667
    #graph_index = 807
    graph_index = 173
    graph_index = 336
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    print('Initializing graph')
    graph = unit_disk_grid_graph(grid, periodic=False, radius=1.6)
    tails_graph = rydberg_graph(grid, visualize=False)
    n_points = 7
    times = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    times = np.concatenate([times, [2.5]])#np.array([2.5])#
    times = times + .312 * 2
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)

    performances = np.zeros((len(critical_detunings), len(times)))*np.nan
    for (d, detuning) in enumerate(critical_detunings):
        for (t, tf) in enumerate(times):
            cost.energies = (1,)
            try:
                state = State(np.load('{}x{}_{}_{}_{}_trotterize.npz'.format(size, size, graph_index, np.round(np.abs(detuning), 2), np.round(np.abs(tf), 2)))['state'], is_ket=True, IS_subspace=True,
                              graph=graph)
                performances[d, t] = cost.optimum_overlap(state)
                print(tf, detuning, performances[d, t])
            except:
                pass
    colors = ['blue', 'green', 'navy', 'orange', 'firebrick', 'purple', 'magenta', 'cornflowerblue', 'teal',
              'grey', 'cyan', 'limegreen', 'red', 'yellow', 'pink', 'orangered', 'salmon', 'violet']

    for (d, detuning) in enumerate(critical_detunings):
        plt.scatter(times-2*.312, performances[d], color = colors[d], label='Detuning $={}$'.format(np.round(detuning, 2)))
        plt.plot(times-2*.312, performances[d], color = colors[d])
    plt.xlabel('Total time ($\mu s$)')
    plt.ylabel('MIS probability')
    plt.semilogx()
    plt.legend()
    plt.show()

def plot_6():
    size = 6
    critical_detuning = -9.604213726908476#-6.019449429835163
    critical_detunings = np.concatenate([-np.linspace(2, 10, 10), [critical_detuning]])
    graph_index = 807
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    print('Initializing graph')
    graph = unit_disk_grid_graph(grid, periodic=False, radius=1.6)
    graph_finite = unit_disk_grid_graph(grid, periodic=False, radius=1.1)

    graph_finite.generate_independent_sets()

    n_points = 7
    times = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    times = np.concatenate([times, [2.5]])  # np.array([2.5])#
    times = times + .312 * 2
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)

    performances = np.zeros((len(critical_detunings), len(times))) * np.nan
    for (d, detuning) in enumerate(critical_detunings):
        for (t, tf) in enumerate(times):
            try:
                cost.energies = (1,)
                state = State(np.load(
                    '{}x{}_{}_{}_{}_trotterize.npz'.format(size, size, graph_index, np.round(np.abs(detuning), 2),
                                                           np.round(np.abs(tf), 2)))['state'], is_ket=True,
                              IS_subspace=True,
                              graph=graph)
                performances[d, t] = MIS_probability_finite(state, graph, graph_finite)
                print(tf, detuning, performances[d, t])
            except:
                pass

    colors = ['blue', 'green', 'navy', 'orange', 'firebrick', 'purple', 'magenta', 'cornflowerblue', 'teal',
              'grey', 'cyan', 'limegreen', 'red', 'yellow', 'pink', 'orangered', 'salmon', 'violet']

    for (d, detuning) in enumerate(critical_detunings):
        plt.scatter(times - 2 * .312, performances[d], color=colors[d],
                    label='Detuning $={}$'.format(np.round(detuning, 2)))
        plt.plot(times - 2 * .312, performances[d], color=colors[d])
    print(repr(performances))
    plt.xlabel('Total time ($\mu s$)')
    plt.ylabel('MIS probability')
    plt.semilogx()
    plt.legend()
    plt.show()


plot_6()