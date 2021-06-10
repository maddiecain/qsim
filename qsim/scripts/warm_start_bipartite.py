import networkx as nx
import cvxpy as cvx

from qsim.evolution import hamiltonian
from qsim.graph_algorithms import qaoa
import numpy as np
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph
from qsim import tools
from qsim.codes.quantum_state import State
from typing import Tuple
from scipy import sparse


def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    Returns:
        np.ndarray: Graph coloring (+/-1 for each node)
        float:      The GW score for this cut.
        float:      The GW bound from the SDP relaxation
    """
    # Kudos: Originally implementation by Nick Rubin, with refactoring and
    # cleanup by Jonathon Ward and Gavin E. Crooks
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # Setup and solve the GW semidefinite programming problem
    psd_mat = cvx.Variable(laplacian.shape, PSD=True)
    obj = cvx.Maximize(cvx.trace(laplacian * psd_mat))
    constraints = [cvx.diag(psd_mat) == 1]  # unit norm
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    evals, evects = np.linalg.eigh(psd_mat.value)
    sdp_vectors = evects.T[evals > float(1.0E-6)].T
    print(sdp_vectors.shape)
    # Bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_mat.value)
    print(bound)
    random_vector = np.random.randn(sdp_vectors.shape[1])
    random_vector = random_vector / np.linalg.norm(random_vector)
    # print(sdp_vectors)
    print([vec @ random_vector for vec in sdp_vectors])
    colors = np.sign([vec @ random_vector for vec in sdp_vectors])
    score = colors @ laplacian @ colors.T
    print(score)
    return bound, colors, score


def generate_SDP_graph(d, epsilon, visualize=False, le=False):
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(2 ** d))
    for i in range(2 ** d):
        for j in range(2 ** d):
            binary_i = 2 * (tools.int_to_nary(i, size=d) - 1 / 2) / np.sqrt(d)
            binary_j = 2 * (tools.int_to_nary(j, size=d) - 1 / 2) / np.sqrt(d)
            if le:
                if (1 - np.dot(binary_i, binary_j)) / 2 > 1 - epsilon + 1e-5:
                    graph.add_edge(i, j, weight=1)
            else:
                if np.isclose((1 - np.dot(binary_i, binary_j)) / 2, 1 - epsilon):
                    graph.add_edge(i, j, weight=1)
    if visualize:
        nx.draw(graph, with_labels=True)
        plt.show()
    return Graph(graph)


def degeneracy(maxcut):
    return sparse.find(maxcut.hamiltonian.real == np.max(maxcut.hamiltonian))[0]


def small_bipartite_graph(visualize=False):
    edges = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 5), (2, 6), (2, 7),
             (2, 8), (2, 9), (3, 5), (3, 7), (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 9)]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(10)))
    graph.add_edges_from(edges)
    if visualize:
        nx.draw(graph, pos=nx.bipartite_layout(graph, list(graph.nodes)[:5]))
        plt.show()
    return Graph(graph)


def large_bipartite_graph(visualize=False):
    edges = [(0, 11), (0, 13), (0, 14), (0, 15), (0, 17), (0, 18), (0, 19), (1, 10), (1, 14), (1, 15), (1, 16), (1, 18),
             (2, 10), (2, 11), (2, 13), (2, 14), (2, 19), (3, 12), (3, 16), (3, 18), (3, 19), (4, 11), (4, 13), (4, 14),
             (4, 15), (4, 16), (4, 18), (5, 10), (5, 12), (5, 14), (5, 15), (5, 17), (5, 18), (5, 19), (6, 10), (6, 12),
             (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14),
             (7, 16), (7, 17), (7, 18), (7, 19), (8, 10), (8, 12), (8, 13), (8, 14), (8, 16), (8, 17), (8, 18), (8, 19),
             (9, 11), (9, 12), (9, 13), (9, 14), (9, 16), (9, 17), (9, 19)]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(10)))
    graph.add_edges_from(edges)
    if visualize:
        nx.draw(graph, pos=nx.bipartite_layout(graph, list(graph.nodes)[:5]))
        plt.show()
    return Graph(graph)


def medium_bipartite_graph(visualize=False):
    edges = [(0, 7), (0, 8), (0, 9), (0, 12), (0, 13), (1, 7), (1, 8), (1, 10), (1, 11), (1, 12), (1, 13), (2, 7),
             (2, 8), (2, 9), (2, 10), (2, 12), (2, 13), (3, 7), (3, 8), (3, 9), (3, 10), (4, 8), (4, 11), (4, 12),
             (4, 13), (5, 8), (5, 10), (5, 11), (6, 7), (6, 8), (6, 10), (6, 11), (6, 13)]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(10)))
    graph.add_edges_from(edges)
    if visualize:
        nx.draw(graph, pos=nx.bipartite_layout(graph, list(graph.nodes)[:5]))
        plt.show()
    return Graph(graph)


def small_3regular_graph(visualize=False):
    edges = [(6, 9), (6, 4), (6, 0), (9, 3), (9, 0), (1, 3), (1, 5), (1, 7), (3, 8), (4, 7), (4, 2), (7, 0), (2, 8),
             (2, 5), (8, 5)]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(10)))
    graph.add_edges_from(edges)
    if visualize:
        nx.draw(graph, pos=nx.bipartite_layout(graph, list(graph.nodes)[:5]))
        plt.show()
    return Graph(graph)


def medium_3regular_graph(visualize=False):
    edges = [(4, 7), (4, 10), (4, 15), (7, 15), (7, 2), (10, 11), (10, 12), (11, 9), (11, 5), (2, 5), (2, 3), (5, 14),
             (3, 14), (3, 6), (14, 15), (12, 0), (12, 8), (9, 1), (9, 13), (6, 8), (6, 13), (1, 13), (1, 0), (0, 8)]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(10)))
    graph.add_edges_from(edges)
    if visualize:
        nx.draw(graph, pos=nx.bipartite_layout(graph, list(graph.nodes)[:5]))
    plt.show()
    return Graph(graph)


def generate_bipartite_graph(n, m, p, visualize=False):
    graph = nx.algorithms.bipartite.generators.random_graph(n, m, p)
    print(graph.edges)
    if visualize:
        nx.draw(graph, pos=nx.bipartite_layout(graph, list(graph.nodes)[:n]))
        plt.show()
    return Graph(graph)


def generate_regular_graph(d, n, visualize=False):
    graph = nx.random_regular_graph(d, n)
    print(graph.edges)
    if visualize:
        nx.draw(graph)
        plt.show()
    return Graph(graph)


def generate_initial_state_hd(graph, hd=1):
    n = len(graph.nodes) // 2
    state = np.zeros(2 ** graph.n, dtype=np.complex128)
    perfect = np.zeros(graph.n, dtype=int)
    perfect[:n] = 1
    for cut in range(2 ** graph.n):
        # Check if perfect with a single defect
        binary = tools.int_to_nary(cut, size=graph.n)
        if np.sum(np.abs(binary - perfect)) == hd or np.sum(np.abs(binary - perfect)) == graph.n - hd:
            state[cut] = 1
    return State(state[:, np.newaxis] / np.linalg.norm(state), is_ket=True, graph=graph)


def generate_initial_state_legal_gw(graph, verbose=False):
    state = np.zeros(2 ** graph.n)
    for i in range(1000):
        bound, colors, score = goemans_williamson(graph.graph)
        colors = (colors + 1) / 2
        if verbose:
            print(i)
            print(colors)
        j = tools.nary_to_int(colors)
        state[j] = 1
        state[2 ** graph.n - 1 - j] = 1
    if verbose:
        print(len(np.where(state != 0)))
    return State(state[:, np.newaxis] / np.linalg.norm(state), is_ket=True, graph=graph)


def generate_inital_state_cf(graph, diff=2, verbose=False):
    cost = hamiltonian.HamiltonianMaxCut(graph, cost_function=True)
    maxcut = np.max(cost.hamiltonian)
    target = maxcut - diff
    where_target = sparse.find(cost.hamiltonian.real == target)[0]
    state = np.zeros(2 ** graph.n)
    state[where_target] = 1
    if verbose:
        print('num nonzero', len(np.argwhere(state != 0)))
    return State(state[:, np.newaxis] / np.linalg.norm(state), is_ket=True, graph=graph)


from qsim.graph_algorithms.graph import ring_graph, line_graph

"""deg = 0
while deg!=2:
    graph = generate_regular_graph(3, 10, visualize=False)
    #state = generate_initial_state_hd(graph, hd=1)
    cost = hamiltonian.HamiltonianMaxCut(graph, cost_function=True)
    deg = len(degeneracy(cost))
    print(deg)"""
# graph = small_bipartite_graph()#(3, 1/3, visualize=False)
# state = generate_initial_state_hd(graph, hd=2) # np.load('d3_SDP_output.npy')#
graph = medium_bipartite_graph()  # generate_regular_graph(3, 16)

graph = ring_graph(12)
state = np.zeros((2 ** 12, 1), dtype=np.complex128)
state = State(state, IS_subspace=False)
for j in range(1):
    i = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    i = tools.nary_to_int(np.roll(i, j))
    state[i] = 1
    i = 1 - np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    i = tools.nary_to_int(np.roll(i, j))
    state[i] = 1
print(np.argwhere(state != 0))
state = state / np.linalg.norm(state)
"""where_optimal = degeneracy(cost)
state[where_optimal] = 10
state = state/np.linalg.norm(state)"""
# state = generate_inital_state_cf(graph, diff=2, verbose=True)
# state = generate_initial_state_legal_gw(graph)
cost = hamiltonian.HamiltonianMaxCut(graph, cost_function=True)

# state = State(state[:, np.newaxis], is_ket=True, graph=graph)
driver = hamiltonian.HamiltonianDriver()
qaoa = qaoa.SimulateQAOA(graph=graph, hamiltonian=[], cost_hamiltonian=cost)
# print('oo',  cost.optimum_overlap(state))
print('cf', cost.cost_function(state))
print('maxcut', np.max(cost.hamiltonian))
for p in range(2, 10):
    max_result = 0
    print(p)
    hamiltonian = [driver] + [cost, driver] * p
    qaoa.hamiltonian = hamiltonian
    """num = 20
    gammas = np.linspace(0, np.pi/2, num)
    betas = np.linspace(0, np.pi, num)
    results = np.zeros((num,num))
    for g in range(len(gammas)):
        print(g)
        for b in range(len(betas)):
            res = qaoa.run([1,.23,.17, 2.13, gammas[g], betas[b], 2.01, 1.53, 1.2, .92])
            results[g, b] = res
    results = np.fft.fft2(results)
    results[0, 0] = 0

    plt.plot(results[:, 2])
    plt.show()
    plt.imshow(np.log(np.abs(results)**2))
    plt.colorbar()
    plt.show()"""

    result = qaoa.find_parameters_basinhopping(n=20, verbose=False, initial_state=state)
    if result['f_val'] > max_result:
        max_result = result['f_val']
        print('better', max_result)
    else:
        pass
        # print('not better', result['f_val']
