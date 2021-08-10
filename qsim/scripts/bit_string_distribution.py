import matplotlib.pyplot as plt
import numpy as np
import pickle
import dill

import scipy.sparse as sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from qsim.codes import qubit
from qsim.codes.quantum_state import State
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.graph_algorithms.graph import Graph, unit_disk_graph, line_graph, degree_fails_graph, unit_disk_grid_graph, rydberg_graph
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim.tools import tools
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim.evolution.lindblad_operators import SpontaneousEmission
from matplotlib import rc
from scipy.optimize import minimize, minimize_scalar, basinhopping, brentq
grid3 = np.array([[False, False,  True,  True,  True,  True],
       [False,  True, False,  True,  True,  True],
       [ True,  True,  True,  True,  True, False],
       [False,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True, False]])

graph = unit_disk_grid_graph(grid3, periodic=False, visualize=False)
tails_graph = rydberg_graph(grid3, visualize=False)

"""
Plan:
Toss 5x6 unit disk graphs with 24 nodes. Time evolve with different sweep times until you find the critical time where 
the MIS probability is 0.95. At this time, look at the final MIS probabilities.
"""


def find_critical_time(graph, critical_optimum_overlap):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)

    def schedule(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (3*2*(1/2-x),)
        driver.energies = (amplitude,)

    ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver], cost_hamiltonian=cost,
                           IS_subspace=True)
    def compute_final_state(T):
        final_state = ad.run(T, schedule, method='odeint')[0][-1]
        cost.energies = (1,)
        optimum_overlap = cost.optimum_overlap(final_state)
        return final_state, optimum_overlap

    def cost_function(T):
        final_state, optimum_overlap = compute_final_state(T)
        print(T, optimum_overlap)
        return optimum_overlap-critical_optimum_overlap

    #return basinhopping(cost_function, 35, niter=3, minimizer_kwargs={'bounds':[(20,50)]})['x']
    return brentq(cost_function, 20, 50, xtol=1e-5)

def find_fidelity(graph, critical_time):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True)

    def schedule(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (3*2*(1/2-x),)
        driver.energies = (amplitude,)

    ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver], cost_hamiltonian=cost,
                           IS_subspace=True)

    final_state = ad.run(critical_time, schedule, method='odeint')[0][-1]
    cost.energies = (1,)
    optimum_indices = np.argwhere(cost._diagonal_hamiltonian == cost.optimum).T[0]
    # Construct an operator that is zero everywhere except at the optimum
    optimum = np.zeros(cost._diagonal_hamiltonian.shape)
    optimum[optimum_indices] = 1
    optimum_overlap = cost.optimum_overlap(final_state)
    return final_state, optimum, optimum_overlap


"""
- Final state MIS distribution vs runtime
- Figure out what the Porter-Thomas distribution is
- 
"""


n = 29
dim1 = 6
dim2 = 6
arr = np.concatenate([np.ones(n), np.zeros(dim1*dim2-n)])

np.random.shuffle(arr)
graph_arr = np.reshape(arr, (dim1, dim2))
graph_arr = np.array([[1., 1., 0., 1., 0., 1.], [1., 1., 0., 1., 1., 1.], [1., 1., 1., 1., 1., 1.],
                     [1., 0., 1., 1., 1., 1.], [1., 1., 0., 1., 0., 1.], [1., 0., 1., 1., 1., 1.]])
arr = graph_arr.flatten()
print(graph_arr)
graph = unit_disk_grid_graph(graph_arr)
print('initialized graph')
graph.independent_sets = 1-graph.independent_sets
#print(graph.degeneracy)
results = np.zeros(graph.n)
arr = 2*arr-1
correlations = np.zeros((graph.n,graph.n))
for IS in graph.independent_sets:
    if np.sum(IS) == graph.mis_size:
        results = results + IS
        for k in range(graph.n):
            j = 0
            for i in range(graph.n):
                if IS[i] == IS[k] and IS[i] == 1:
                    correlations[i][k] = correlations[i][k] + 1
temp = arr.copy()
temp[temp == 1] = results/graph.degeneracy
temp[temp == -1] = np.nan
temp = np.reshape(temp, (dim1, dim2))
plt.imshow(temp*(1-temp), cmap='RdYlGn', vmin=-.4, vmax=.4)
plt.colorbar()
#plt.show()
#print(correlations/graph.degeneracy)
print((correlations/graph.degeneracy)[0,:])
correlator = correlations/graph.degeneracy - np.outer(results/graph.degeneracy, results/graph.degeneracy)
eigvals, eigvecs = np.linalg.eigh(correlator)
print(np.sum(eigvecs.imag))
print(eigvals)
raise Exception
fig, axs = plt.subplots(1,6)
k = 0
for ax in axs:
    temp = arr.copy()
    temp[temp == 1] = eigvecs[:, graph.n - 1 - k].real
    temp[temp == -1] = np.nan
    temp = np.reshape(temp, (6, 6))
    pcm = ax.imshow(temp, cmap='RdYlGn', vmin=-.4, vmax=.4)
    k += 1
#fig.colorbar(pcm, ax=ax)
#fig.subplots_adjust(right=1.2)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(pcm, cax=axs[5],aspect=20)
plt.show()
fig, axs = plt.subplots(1,3)
k = 0
special = [5, 10, 19]
for ax in axs:
    print(results.shape, correlations.shape)
    res = correlator[special[k],:]
    # print(results)
    temp = arr.copy()
    temp[temp == 1] = res
    print(res)
    temp[temp == -1] = np.nan
    # print(arr)
    temp = np.reshape(temp, (dim1, dim2))
    pcm = ax.imshow(temp, cmap='RdYlGn', vmin=-np.max(res), vmax=np.max(res))
    k += 1
    fig.colorbar(pcm, ax=ax)
plt.show()





for i in range(0):
    degeneracy = 0
    while degeneracy < 100:
        arr = np.concatenate([np.ones(29), np.zeros(7)])
        np.random.shuffle(arr)
        graph = unit_disk_grid_graph(np.reshape(arr, (6, 6)), visualize=True)
        print(degeneracy)
        degeneracy = graph.degeneracy
    print('degeneracy', graph.degeneracy, 'hilbert space size', graph.num_independent_sets)
    final_state, optimum, optimum_overlap = find_fidelity(graph, 35)
    #plt.scatter(graph.degeneracy, optimum_overlap, color='navy')
    final_state = final_state*optimum
    final_state = final_state/np.linalg.norm(final_state)
    final_state = (np.abs(final_state)**2).flatten()
    final_state = np.flip(np.sort(final_state))
    print(final_state[:100])
    where_nonzero = np.max(np.argwhere(final_state != 0))
    final_state = final_state[:where_nonzero+1]
    plt.hist(final_state, bins=20)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.semilogy()

    plt.show()
plt.ylabel('1-Fidelity with MIS')
plt.xlabel('Degeneracy/Hilbert space size')
plt.show()

