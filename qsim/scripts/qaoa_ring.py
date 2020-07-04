import networkx as nx
from qsim.evolution import hamiltonian
from qsim.graph_algorithms import qaoa
import numpy as np
import matplotlib.pyplot as plt
# Construct a known graph

def ring(n=2):
    graph = nx.Graph()
    if n == 1:
        graph.add_node(0)
    else:
        graph.add_weighted_edges_from(
            [(i, i + 1, 1) for i in range(n - 1)])
        graph.add_edge(0, n-1, weight=1)
    return [graph, np.floor(n / 2)]




def mis_vs_maxcut(n, method='minimize', show=True):
    maxcut = [(2*p+1)/(2*p+2) for p in range(1, int(np.floor(n/2)))]
    G, opt = ring(n)
    # nx.draw_networkx(G)
    # plt.show()
    mis = []
    for p in range(1, int(np.floor(n/2))):
        # Find MIS optimum
        # Uncomment to visualize graph
        print(p)
        # For MaxCut
        if p == 3:
            # hc_qubit = hamiltonian.HamiltonianC(G, mis=False)"
            hc_qubit = hamiltonian.HamiltonianRydberg(G, energy=-1, detuning=1/2)
            sim = qaoa.SimulateQAOA(G, p, 2, is_ket=True, C=hc_qubit)
            # Set the default variational operators
            sim.hamiltonian = [hc_qubit, hamiltonian.HamiltonianB()]

            sim.noise = [None] * len(sim.hamiltonian)

            # You should get the same thing
            if method == 'minimize':
                results = sim.find_parameters_minimize(verbose=False)
                approximation_ratio = -1 * np.real(results.fun) / opt
                mis.append(approximation_ratio)
                print(approximation_ratio)
        else:
            mis.append(0)

    plt.plot(list(range(1, int(np.floor(n/2)))), maxcut, c='y', label='maxcut')
    plt.scatter(list(range(1, int(np.floor(n/2)))), mis, c='teal', label='mis')
    plt.legend()
    if show:
        plt.show()

mis_vs_maxcut(16, show=True)