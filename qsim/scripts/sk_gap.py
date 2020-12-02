import networkx as nx
import itertools
import numpy as np
from qsim.graph_algorithms.graph import Graph
from qsim.evolution.hamiltonian import HamiltonianMaxCut, HamiltonianDriver
import scipy.sparse as sparse
from qsim.schrodinger_equation import SchrodingerEquation
import matplotlib.pyplot as plt

def sk_integer(n, verbose=False):
    graph = nx.complete_graph(n)
    weights = [-1, 1]
    for (i, j) in itertools.combinations(range(n), 2):
        graph[i][j]['weight'] = weights[np.random.randint(2)]
    if verbose:
        print(graph.edges.data())
    return Graph(graph)


def ground_states(h):
    return np.argwhere(h==np.min(h))

def first_excited_states(h):
    gs = ground_states(h).T
    h[gs[0], gs[1]] = np.inf
    return ground_states(h)

def find_gap(graph, nondegenerate=False):
    # Compute the number of ground states and first excited states
    cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=True)
    # Generate a dummy graph with one fewer nodes
    # Cut cost and driver hamiltonian in half to account for Z2 symmetry
    driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n - 1)))
    driver.hamiltonian
    row = list(range(2 ** (graph.n - 1)))
    column = list(range(2 ** (graph.n - 1)))
    column.reverse()
    driver._hamiltonian = driver._hamiltonian + sparse.csr_matrix((np.ones(2 ** (graph.n - 1)), (row, column)))
    n_ground = len(ground_states(cost.hamiltonian))
    if nondegenerate:
        if n_ground > 1:
            return None
    times = np.arange(0, 1, .01)
    def schedule(t):
        cost.energies = (t,)
        driver.energies = (t-1,)
    min_gap = np.inf
    for i in range(len(times)):
        schedule(times[i])
        eigvals = SchrodingerEquation(hamiltonians=[cost, driver]).eig(k=n_ground+1, return_eigenvectors=False)
        eigvals = np.flip(eigvals)
        if eigvals[-1]-eigvals[0] < min_gap:
            #print(min_gap, eigvals, eigvals[-1]-eigvals[0], times[i], n_ground)
            min_gap = eigvals[-1]-eigvals[0]
    return min_gap

def collect_min_gap_statistics(n, iter=50, verbose=False):
    gaps = []
    for i in range(iter):
        if verbose:
            print(i)
        graph = sk_integer(n)
        gaps.append(find_gap(graph))
    return gaps

def min_gap_vs_n(ns, iters=None, verbose=False):
    for i in range(len(ns)):
        if verbose:
            print(ns[i])
        if iters is not None:
            gaps = collect_min_gap_statistics(ns[i], iters[i], verbose=verbose)
        else:
            gaps = collect_min_gap_statistics(ns[i], verbose=verbose)
        print(ns[i], np.mean(gaps), np.std(gaps))
        plt.errorbar(ns[i], np.mean(gaps), yerr=np.std(gaps), color='black')
        plt.scatter(ns[i], np.mean(gaps), color='blue')
    plt.show()


if __name__ == '__main__':
    import sys
    index = sys.argv[1]
    index = int(index)
    n = 15
    print(n, collect_min_gap_statistics(n, iter=1, verbose=False)[0], flush=True)

