import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qsim.tools.tools import *
from qsim.evolution.hamiltonian import HamiltonianMaxCut
from qsim.graph_algorithms.graph import Graph
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix

while True:
    d = 5
    n = 24
    graph = nx.random_regular_graph(d, n)
    hamz = HamiltonianMaxCut(Graph(graph))
    maxcut_size = np.max(hamz._diagonal_hamiltonian).astype(int)
    mincut_size = np.min(hamz._diagonal_hamiltonian).astype(int)
    #plt.hist(np.array(csr_matrix.diagonal(hamz.hamiltonian)), bins=maxcut_size-mincut_size, range=(mincut_size, maxcut_size))
    #plt.show()
    independence_polynomial = []
    for i in range(maxcut_size-7, maxcut_size):
        independence_polynomial.append(len(np.argwhere(hamz._diagonal_hamiltonian==i).T[0]))
    ratios = []
    for i in range(len(independence_polynomial)-1):
        try:
            ratios.append(independence_polynomial[i]/independence_polynomial[i+1])
        except ZeroDivisionError:
            ratios.append(0)
    print(independence_polynomial, ratios)
    if True:#ratios[-1] > 3:
        maxcut = np.argwhere(hamz._diagonal_hamiltonian == maxcut_size).T[0]
        maxcutminusone = np.argwhere(hamz._diagonal_hamiltonian == maxcut_size-1).T[0]
        maxcut_configs = np.zeros((len(maxcut), n), dtype=bool)
        maxcutminusone_configs = np.zeros((len(maxcutminusone), n), dtype=bool)
        maxcutminusone_degrees = np.zeros(len(maxcutminusone), dtype=int)

        for (_, i) in enumerate(maxcut):
            maxcut_configs[_] = int_to_nary(i, size=n).astype(bool)

        for (_, i) in enumerate(maxcutminusone):
            maxcutminusone_configs[_] = int_to_nary(i, size=n).astype(bool)

        configuration_hds = (n*pairwise_distances(maxcutminusone_configs, maxcutminusone_configs, metric='hamming')).astype(int) == 2
        degrees = np.sum(configuration_hds, axis=1)
        distances = (n*pairwise_distances(maxcut_configs, maxcutminusone_configs, metric='hamming')).astype(int)
        print(np.max(distances, axis=0))
        print(np.min(distances, axis=0))
        distances = np.min(distances, axis=0)
        configuration_graph = nx.from_numpy_array(configuration_hds)
        colors = []
        for node in configuration_graph:
            if distances[node] == 1:
                colors.append('red')
            else:
                colors.append('navy')
        nx.draw(configuration_graph, node_color=colors)
        plt.show()
        from collections import Counter

        c = Counter(zip(distances, degrees))
        s = [10 * c[(xx, yy)] for xx, yy in zip(distances,degrees )]
        plt.scatter(distances, degrees, s=s)
        plt.show()
