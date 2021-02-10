import networkx as nx
import itertools
import numpy as np
from qsim.graph_algorithms.graph import Graph
from qsim.evolution.hamiltonian import HamiltonianMaxCut, HamiltonianDriver
import scipy.sparse as sparse
from qsim.schrodinger_equation import SchrodingerEquation
import matplotlib.pyplot as plt
import scipy
from qsim.codes import qubit
from qsim.tools import tools

def sk_integer(n, verbose=False):
    graph = nx.complete_graph(n)

    weights = [-1, 1]
    for (i, j) in itertools.combinations(range(n), 2):
        graph[i][j]['weight'] = weights[np.random.randint(2)]
    if verbose:
        print(graph.edges.data())
    return Graph(graph)


def sk_hamiltonian(n, p=3, use_Z2_symmetry=False, use_degenerate=True, verbose=False):
    w = []

    if use_Z2_symmetry:
        c = np.zeros([2**(n-1), 1])
    else:
        c = np.zeros([2**n, 1])

    z = np.expand_dims(np.diagonal(qubit.Z), axis=0).T

    def my_eye(n):
        return np.ones((np.asarray(2) ** n, 1))
    weights = [-1, 1]
    if n%2 == 1 and use_Z2_symmetry:
        raise Exception('n odd hamiltonians are not Z2 symmetric')
    for i in itertools.combinations(range(n), p):
        op = []
        i = sorted(i)
        for j in range(len(i)):
            if use_Z2_symmetry:
                if j == 0:
                    if i[j] != 0:
                        op.append(my_eye(i[j]-1))
                        op.append(z)
                elif j != len(i)-1:
                    op.append(my_eye(i[j]-i[j-1]-1))
                    op.append(z)
                else:
                    op.append(my_eye(i[j] - i[j - 1] - 1))
                    op.append(z)
                    op.append(my_eye(n-i[j]-1))
            else:
                if j == 0:
                    op.append(my_eye(i[j]))
                    op.append(z)
                elif j != len(i) - 1:
                    op.append(my_eye(i[j] - i[j - 1] - 1))
                    op.append(z)
                else:
                    op.append(my_eye(i[j] - i[j - 1] - 1))
                    op.append(z)
                    op.append(my_eye(n - i[j] - 1))
        if use_Z2_symmetry:
            weight = weights[np.random.randint(2)]
            w.append(weight)
            c = c - weight * (tools.tensor_product(op))
        else:
            weight = weights[np.random.randint(2)]
            w.append(weight)
            c = c - weight * (tools.tensor_product(op))
    if use_degenerate:
        return c
    else:
        # Check that the result is not degenerate
        if len(ground_states(c)) == 1:
            return c
        else:
            if verbose:
                print('degeneracy', len(ground_states(c)))
            return sk_hamiltonian(n, p=p, use_Z2_symmetry=use_Z2_symmetry, use_degenerate=use_degenerate, verbose=verbose)


def sk_p3_instance():
    n = 18
    use_Z2_symmetry = False
    w = [1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1]

    if use_Z2_symmetry:
        c = np.zeros([2**(n-1), 1])
    else:
        c = np.zeros([2**n, 1])

    z = np.expand_dims(np.diagonal(qubit.Z), axis=0).T

    def my_eye(n):
        return np.ones((np.asarray(2) ** n, 1))
    k = 0
    if n%2 == 1 and use_Z2_symmetry:
        raise Exception('n odd hamiltonians are not Z2 symmetric')
    for i in itertools.combinations(range(n), 3):
        op = []
        i = sorted(i)
        for j in range(len(i)):
            if use_Z2_symmetry:
                if j == 0:
                    if i[j] != 0:
                        op.append(my_eye(i[j]-1))
                        op.append(z)
                elif j != len(i)-1:
                    op.append(my_eye(i[j]-i[j-1]-1))
                    op.append(z)
                else:
                    op.append(my_eye(i[j] - i[j - 1] - 1))
                    op.append(z)
                    op.append(my_eye(n-i[j]-1))
            else:
                if j == 0:
                    op.append(my_eye(i[j]))
                    op.append(z)
                elif j != len(i) - 1:
                    op.append(my_eye(i[j] - i[j - 1] - 1))
                    op.append(z)
                else:
                    op.append(my_eye(i[j] - i[j - 1] - 1))
                    op.append(z)
                    op.append(my_eye(n - i[j] - 1))

        weight = w[k]
        c = c - weight * (tools.tensor_product(op))
        k += 1
    return c


def ground_states(h):
    return np.argwhere(h == np.min(h))


def first_excited_states(h):
    gs = ground_states(h).T
    h[gs[0], gs[1]] = np.inf
    return ground_states(h)


def find_gap(graph, use_Z2_symmetry=True):
    # Compute the number of ground states and first excited states
    cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
    # Generate a dummy graph with one fewer nodes
    # Cut cost and driver hamiltonian in half to account for Z2 symmetry
    if use_Z2_symmetry:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n - 1)))
        driver.hamiltonian
        row = list(range(2 ** (graph.n - 1)))
        column = list(range(2 ** (graph.n - 1)))
        column.reverse()
        driver._hamiltonian = driver._hamiltonian + (-1) ** graph.n * sparse.csr_matrix(
            (np.ones(2 ** (graph.n - 1)), (row, column)))
    else:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n)))
    n_ground = len(ground_states(cost.hamiltonian))
    times = np.arange(0, 1, .01)

    def schedule(t):
        cost.energies = (t,)
        driver.energies = (t - 1,)

    min_gap = np.inf
    for i in range(len(times)):
        schedule(times[i])
        eigvals = SchrodingerEquation(hamiltonians=[cost, driver]).eig(k=n_ground + 1, return_eigenvectors=False)
        eigvals = np.flip(eigvals)
        if eigvals[-1] - eigvals[0] < min_gap:
            min_gap = eigvals[-1] - eigvals[0]
    return min_gap, n_ground


def find_gap_fixed_n(n, use_degenerate=True, use_Z2_symmetry=True, verbose=False, p=2):
    # Compute the number of ground states and first excited states
    n_ground = np.inf
    if not use_degenerate:
        if use_Z2_symmetry:
            if p == 2:
                while n_ground != 1:
                    graph = sk_integer(n)
                    cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
                    n_ground = len(ground_states(cost.hamiltonian))
                    if verbose and n_ground != 1:
                        print('Initially failed to find graph, n_ground =', n_ground)
            else:
                graph = sk_hamiltonian(n, p=p, use_degenerate=use_degenerate, use_Z2_symmetry=use_Z2_symmetry)

        else:
            if p == 2:
                while n_ground != 2:
                    graph = sk_integer(n)
                    cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
                    n_ground = len(ground_states(cost.hamiltonian))
                    if verbose and n_ground != 2:
                        print('Initially failed to find graph, n_ground =', n_ground)
            else:
                graph = sk_hamiltonian(n, p=p, use_degenerate=use_degenerate, use_Z2_symmetry=use_Z2_symmetry)
    else:
        if p == 2:
            graph = sk_integer(n)
            cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
            n_ground = len(ground_states(cost.hamiltonian))
        else:
            graph = sk_hamiltonian(n, p=p, use_degenerate=use_degenerate, use_Z2_symmetry=use_Z2_symmetry)
    if p != 2:
        ham = graph
        graph = sk_integer(n)
        cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
        # Replace the cost Hamiltonian
        cost._diagonal_hamiltonian = ham
        n_ground = len(ground_states(ham))

        if use_Z2_symmetry:
            cost._hamiltonian = sparse.csr_matrix((cost._diagonal_hamiltonian.flatten(), (np.arange(2 ** (graph.n - 1)),
                                                                                          np.arange(
                                                                                              2 ** (graph.n - 1)))),
                                                  shape=(2 ** (graph.n - 1), 2 ** (graph.n - 1)))
        else:
            cost._hamiltonian = sparse.csr_matrix((cost._diagonal_hamiltonian.flatten(), (np.arange(2 ** (graph.n)),
                                                                                          np.arange(2 ** (graph.n)))),
                                                  shape=(2 ** (graph.n), 2 ** (graph.n)))
    # Generate a dummy graph with one fewer nodes
    # Cut cost and driver hamiltonian in half to account for Z2 symmetry
    if use_Z2_symmetry:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n - 1)))
        driver.hamiltonian
        row = list(range(2 ** (graph.n - 1)))
        column = list(range(2 ** (graph.n - 1)))
        column.reverse()
        driver._hamiltonian = driver._hamiltonian + (-1) ** graph.n * sparse.csr_matrix(
            (np.ones(2 ** (graph.n - 1)), (row, column)))
    else:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n)))

    def schedule(t):
        cost.energies = (np.sqrt(np.math.factorial(p)/(2 * graph.n**(p-1))) * t,)
        driver.energies = (1-t,)

    def gap(t):
        t = t[0]
        if verbose:
            print('time', t)
        schedule(t)
        eigvals = SchrodingerEquation(hamiltonians=[cost, driver]).eig(k=n_ground + 1, return_eigenvectors=False)
        eigvals = np.flip(eigvals)
        if verbose:
            print('gap', eigvals[-1]-eigvals[0])
        return eigvals[-1]-eigvals[0]

    # Now minimize the gap
    min_gap = scipy.optimize.minimize(gap, .85, bounds=[(0, 1)])
    return min_gap.fun


def gap_over_time(graph, verbose=False, use_Z2_symmetry=True):
    # Compute the number of ground states and first excited states
    cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
    print(np.min(cost.hamiltonian) / graph.n ** (3 / 2))
    # Generate a dummy graph with one fewer nodes
    # Cut cost and driver hamiltonian in half to account for Z2 symmetry
    if use_Z2_symmetry:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n - 1)))
        driver.hamiltonian
        row = list(range(2 ** (graph.n - 1)))
        column = list(range(2 ** (graph.n - 1)))
        column.reverse()
        driver._hamiltonian = driver._hamiltonian + (-1) ** graph.n * sparse.csr_matrix(
            (np.ones(2 ** (graph.n - 1)), (row, column)))
    else:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n)))
    if verbose:
        print(ground_states(cost.hamiltonian))
    n_ground = len(ground_states(cost.hamiltonian))
    # print(driver.hamiltonian.todense())
    # print(np.linalg.eigh(driver.hamiltonian.todense()))
    if verbose:
        print('degeneracy ', n_ground)
    times = np.arange(0, 1, .01)

    def schedule(t):
        cost.energies = (1 / np.sqrt(graph.n) * t,)
        driver.energies = (1 - t,)

    all_eigvals = np.zeros((len(times), n_ground + 1))
    for i in range(len(times)):
        if verbose:
            print(times[i])
        schedule(times[i])
        eigvals = SchrodingerEquation(hamiltonians=[cost, driver]).eig(k=n_ground + 1, return_eigenvectors=False)
        eigvals = np.flip(eigvals)
        all_eigvals[i] = eigvals
    for i in range(n_ground + 1):
        plt.scatter(times, all_eigvals[:, i], color='blue', s=2)
    if verbose:
        print(all_eigvals)
    plt.xlabel(r'normalized time $s$')
    plt.ylabel(r'energy')
    plt.show()


def low_energy_subspace_at_fixed_time(graph, s,  use_Z2_symmetry=True, n_ground=None, k=None, p=2):
    # Compute the number of ground states and first excited states
    if p == 2:
        cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
    if p != 2:
        ham = graph
        if use_Z2_symmetry:
            graph = sk_integer(int(np.log2(graph.shape[0]))+1)
        else:
            graph = sk_integer(int(np.log2(graph.shape[0])))

        cost = HamiltonianMaxCut(graph, cost_function=False, use_Z2_symmetry=use_Z2_symmetry)
        # Replace the cost Hamiltonian
        cost._diagonal_hamiltonian = ham
        if use_Z2_symmetry:
            cost._hamiltonian = sparse.csr_matrix((cost._diagonal_hamiltonian.flatten(), (np.arange(2 ** (graph.n-1)),
                                                 np.arange(2 ** (graph.n-1)))),shape=(2 ** (graph.n-1), 2 ** (graph.n-1)))
        else:
            cost._hamiltonian = sparse.csr_matrix((cost._diagonal_hamiltonian.flatten(), (np.arange(2 ** (graph.n)),
                                                 np.arange(2 ** (graph.n)))),shape=(2 ** (graph.n), 2 ** (graph.n)))
    # Generate a dummy graph with one fewer nodes
    # Cut cost and driver hamiltonian in half to account for Z2 symmetry
    if use_Z2_symmetry:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n - 1)))
        driver.hamiltonian
        row = list(range(2 ** (graph.n - 1)))
        column = list(range(2 ** (graph.n - 1)))
        column.reverse()
        driver._hamiltonian = driver._hamiltonian + (-1) ** graph.n * sparse.csr_matrix(
            (np.ones(2 ** (graph.n - 1)), (row, column)))
    else:
        driver = HamiltonianDriver(graph=Graph(nx.complete_graph(graph.n)))
    if k is None:
        if n_ground is None:
            n_ground = len(ground_states(cost.hamiltonian))
        k=n_ground+1

    def schedule(t):
        cost.energies = (np.sqrt(np.math.factorial(p)/(2 * graph.n**(p-1))) * t,)
        driver.energies = (1 - t,)
    schedule(s)
    eigvals = SchrodingerEquation(hamiltonians=[cost, driver]).eig(k=k, return_eigenvectors=False)
    eigvals = np.flip(eigvals)
    return s, eigvals


def collect_min_gap_statistics(n, iter=50, verbose=False):
    gaps = []
    degeneracies = []
    for i in range(iter):
        if verbose:
            print(i)
        graph = sk_integer(n)
        gap, degeneracy = find_gap(graph)
        gaps.append(gap)
        degeneracies.append(degeneracy)
    return gaps, degeneracies


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


"""gaps = []
n = 17
for i in range(100):
    gap = find_gap_fixed_n(n, use_degenerate=False, use_Z2_symmetry=True, verbose=False)
    print(i, gap)
    gaps.append(gap)

print(gaps)
print(np.median(gaps))
print(np.std(gaps))"""


#graph = sk_integer(5)
#times = np.arange(0, 1, .01)
#s, eigvals = low_energy_subspace_at_fixed_time(graph, times[0], use_Z2_symmetry=True, k=5)g
"""gaps = []
for i in range(400):
    print(i)
    n = 15
    gap = find_gap_fixed_n(n, use_degenerate=False, use_Z2_symmetry=True, verbose=False)
    gaps.append(gap)
print(gaps)"""
#print(s, eigvals, flush=True)
n = 15
times = np.arange(0, 1.0, .01)
k = 2
use_Z2_symmetry=False
eigvals = np.zeros((len(times), k))
p = 7
graph = sk_hamiltonian(n, use_degenerate=False, p=p, verbose=True)
#graph = sk_p3_instance()
for i in range(len(times)):
    print(i)
    s, eigval = low_energy_subspace_at_fixed_time(graph, times[i], use_Z2_symmetry=use_Z2_symmetry, k=k, p=p)
    eigvals[i, :] = eigval
for i in range(k):
    if i == 0:
        plt.plot(times, eigvals[:,i], color='red')
    else:
        plt.plot(times, eigvals[:,i], color='navy')
plt.xlabel(r'$t/T$')
plt.ylabel(r'Energy')
plt.show()
"""
n = 17
gaps = []
for i in range(100):
    print(i, flush=True)
    gap = find_gap_fixed_n(n, use_Z2_symmetry=False, use_degenerate=False, p=3, verbose=False)
    gaps.append(gap)
    print(gaps)
print(gaps)"""

if __name__ == '__main__':
    import sys
    index = sys.argv[1]
    index = int(index)
    n = 21
    #times = np.arange(0, 1.01, .01)
    #graph = sk_p3_instance()
    #s, eigval = low_energy_subspace_at_fixed_time(graph, times[index], use_Z2_symmetry=False, k=2, p=3)
    gap = find_gap_fixed_n(n, use_Z2_symmetry=False, use_degenerate=False, p=3, verbose=False)
    #gap = find_gap_fixed_n(n, use_degenerate=False, use_Z2_symmetry=True, verbose=False)
    print(n, gap, flush=True)
    #times = np.arange(0, 1, .01)
    #s, eigvals = low_energy_subspace_at_fixed_time(Graph(nx.from_numpy_array(graph)), times[index], use_Z2_symmetry=True, k=20)
    #rint(s, eigvals, flush=True)
