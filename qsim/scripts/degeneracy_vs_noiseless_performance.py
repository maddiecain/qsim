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
from qsim.graph_algorithms.graph import Graph, unit_disk_graph, line_graph, degree_fails_graph, unit_disk_grid_graph
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim.tools import tools
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim.evolution.lindblad_operators import SpontaneousEmission
from matplotlib import rc
from scipy.optimize import minimize, minimize_scalar, basinhopping, brentq


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
for i in range(50):
    arr = np.concatenate([np.ones(24), np.zeros(6)])
    np.random.shuffle(arr)
    graph = unit_disk_grid_graph(np.reshape(arr, (5, 6)))
    #print('degeneracy', graph.degeneracy, 'hilbert space size', graph.num_independent_sets)
    final_state, optimum, optimum_overlap = find_fidelity(graph, 35)
    plt.scatter(graph.degeneracy, 1-optimum_overlap, color='navy')
    #final_state = final_state*optimum
    #final_state = final_state/np.linalg.norm(final_state)
    #final_state = (np.abs(final_state)**2).flatten()
    #final_state = np.flip(np.sort(final_state))
    #print(final_state[:100])
    #where_nonzero = np.max(np.argwhere(final_state != 0))
    #final_state = final_state[:where_nonzero+1]
    #plt.bar(np.arange(len(final_state)), final_state)
    #plt.xlabel('MIS index')
    #plt.ylabel('Probability')
    #plt.show()
plt.loglog()
plt.ylabel('1-Fidelity with MIS')
plt.xlabel('Degeneracy/Hilbert space size')
plt.show()

