import networkx as nx
import scipy.sparse.csr
import scipy.sparse.linalg
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random


def gen_hamx(n, d, num_connections=None, num_removals=0):
    nx_graph = nx.generators.random_regular_graph(d, n-1)
    if num_connections is None:
        num_connections = 10 * d
    nx_graph.add_node(n)
    # pick num_connections//2 edges at random
    edges = random.sample(tuple(nx_graph.edges), k=num_connections//2)
    if num_removals >= 1:
        nx_graph.remove_edges_from(edges)
    nonmaximal = np.unique(np.array(edges).flatten())
    if num_removals > 1:
        for nm in nonmaximal:
            remove = np.random.choice(list(nx_graph[nm]), size=num_removals-1)
            nx_graph.remove_edges_from([(nm, r) for r in remove])
    nx_graph.add_edges_from([(n, u) for u in nonmaximal])
    #nx.draw(nx_graph)
    #plt.show()
    adj_matrix = nx.to_scipy_sparse_matrix(nx_graph)
    #print(np.sum(adj_matrix.todense(), axis=1).flatten())
    #plt.plot(np.array(np.sum(adj_matrix.todense(), axis=1)).flatten())
    #plt.show()
    #plt.hist(np.sum(adj_matrix, axis=1))
    #plt.show()
    return adj_matrix


def gen_hamz(n, indices=(0,)):
    #indices = np.random.choice(np.arange(n), size=len(indices))
    hamz = scipy.sparse.csr_matrix((np.ones(len(indices)), (indices, indices)), shape=(n, n))
    return hamz


def find_gap(hamx, hamz, k=1):
    def gap(s):
        if not isinstance(s, float):
            s = s[0]
        hams = (1 - s) * hamx + s * hamz
        eigval, eigvec = scipy.sparse.linalg.eigsh(hams, which='LA', k=k + 1)
        return np.abs(eigval[-2] - eigval[-1])

    res = minimize(gap, [.5], bounds=[[.7, 1 - 1e-5]])
    s = res.x[0]
    hams = (1 - s) * hamx + s * hamz
    eigval, eigvec = scipy.sparse.linalg.eigsh(hams, which='LA', k=k + 1)
    #print(eigvec[:, 0] - eigvec[:, 1], eigvec[:, 0] + eigvec[:, 1])
    plt.plot((eigvec[:, 0] - eigvec[:, 1]))
    plt.plot((eigvec[:, 0] + eigvec[:, 1]))
    #plt.plot((eigvec[:, 0]))
    #plt.plot( eigvec[:, 1])
    #plt.plot(np.diag(hamz.todense()))
    #plt.plot(np.sum(hamx.todense(), axis=1))
    plt.show()
    return res


def plot_gap(hamx, hamz, k=1):
    def gap(s):
        if not isinstance(s, float):
            s = s[0]
        hams = (1 - s) * hamx + s * hamz
        eigval, eigvec = scipy.sparse.linalg.eigsh(hams, which='LA', k=k + 1)
        return eigval

    for s in np.linspace(.87, .999, 100):
        eigval = gap(s)
        plt.scatter(np.ones_like(eigval) * s, eigval, color='navy')
        #d = 50
        #plt.scatter(s, s+.5*4*d*(1-s)**2, color='magenta')
        #plt.scatter(s, d*(1-s), color='seagreen')
    plt.show()


def generate_hamiltonians(n, d, num_connections=None):
    if num_connections is None:
        num_connections = 2*d
    hamx = gen_hamx(n, d,  num_connections=num_connections)
    degrees = np.sum(hamx, axis=0)[0]
    difference = num_connections-d
    indices = tuple(np.argwhere(np.abs(degrees - np.max(degrees)) < difference // 2)[:, 1:].flatten())
    hamz = gen_hamz(n, indices=indices)

    return hamx, hamz

k=1
fig, ax = plt.subplots(1, 2)
x = []
gap = []
loc = []
er = False
for n in (10 ** np.linspace(2.5, 4.5, 30)).astype(int):
    d = int(np.log2(n)/3)*2
    d = 10
    print(n, d)
    # int(np.log2(n))*2
    # np.log2(n)/n*2
    #4.6*d

    hamx, hamz = generate_hamiltonians(n, d,  num_connections=int(2*d))
    #plot_gap(hamx, hamz)
    #degrees = np.sum(hamx, axis=1)
    #print(np.std(degrees[degrees <= d]))
    res = find_gap(hamx, hamz)
    ax[0].scatter(n, res.fun, color='navy')
    ax[1].scatter(n, 1 - res.x, color='navy')
    x.append(n)
    gap.append(res.fun)
    loc.append(1 - res.x[0])
#res, err = curve_fit(lambda x, a, b: a * x ** b, x, gap)
#ax[0].plot(x, res[0] * np.array(x) ** res[1], color='k', label=np.round(res[1], 3))
#res, err = curve_fit(lambda x, a, b: a * x ** b, x, loc)
ax[0].plot(x, .5*np.array(x, dtype=float)**-.5, color='k', linestyle='solid', label='$y\sim 1/\sqrt{x}$')
ax[0].plot(x, .5*np.array(x, dtype=float)**-1, color='k', linestyle='dashed', label='$y\sim 1/x$')
#ax[1].plot(x, res[0] * np.array(x) ** res[1], color='k', label=np.round(res[1], 3))
ax[0].loglog()
ax[0].set_xlabel('$n$')
ax[0].set_ylabel('gap')
ax[1].set_xlabel('$n$')
ax[1].set_ylabel('$1-s_{\\rm{crit}}$')
ax[1].loglog()
ax[0].legend(frameon=False)
plt.show()

