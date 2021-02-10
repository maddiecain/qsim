import itertools

import numpy as np
import scipy.integrate
import scipy.optimize
from odeintw import odeintw
import matplotlib.pyplot as plt
import networkx as nx

import scipy.sparse as sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from qsim.codes import qubit
from qsim.codes.quantum_state import State
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.graph_algorithms.graph import Graph
from qsim.graph_algorithms.graph import line_graph, degree_fails_graph, ring_graph
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.tools import tools



def rate_vs_eigenenergy(times, graph=line_graph(n=2), which='S'):
    """For REIT, compute the total leakage from the ground state to a given state. Plot the total leakage versus
    the final eigenenergy"""
    bad = np.arange(0, 2**graph.n, 1)
    if which == 'S':
        index = 0
    elif which == 'L':
        index = -1
    else:
        index = which
    bad = np.delete(bad, index)
    full_rates = np.zeros(len(bad))

    # Good is a list of good eigenvalues
    # Bad is a list of bad eigenvalues. If 'other', defaults to all remaining eigenvalues outside of 'good'
    def schedule(t, tf):
        phi = (tf - t) / tf * np.pi / 2
        x.energies = (np.sin(phi)**2,)

    x = hamiltonian.HamiltonianDriver(graph=graph, IS_subspace=False, energies=(1,))
    zz = hamiltonian.HamiltonianMaxCut(graph, cost_function=False, energies=(1/2,))
    #dissipation = lindblad_operators.SpontaneousEmission(graph=graph, rates=(1,))
    eq = SchrodingerEquation(hamiltonians=[x, zz])

    def compute_rate():
        # Construct the first order transition matrix
        energies, states = eq.eig(k='all')
        rates = np.zeros(len(bad))
        for j in range(graph.n):
            for i in range(len(bad)):
                rates[i] = rates[i] + (np.abs(states[i].conj() @ qubit.left_multiply(State(states[index].T), [j], qubit.Z))**2)[0,0]
        # Select the relevant rates from 'good' to 'bad'
        print(rates)
        return rates

    for i in range(len(times)):
        print(times[i])
        schedule(times[i], 1)
        full_rates = full_rates + compute_rate()
        eigval, eigvec = eq.eig(k='all')
    return full_rates, eigval



n = 9
graph = line_graph(n)
rates, eigvals = rate_vs_eigenenergy([.5], graph=graph, which='S')
print(rates)
eigvals = np.abs(eigvals-eigvals[0])
eigvals = np.delete(eigvals, 0)
#plt.hist(eigvals, bins=30, weights=rates)
#res = np.polyfit(eigvals, np.log(rates), 1)
#print(res)
# What is the energy spacing
plt.scatter(eigvals, rates)
#print(res[0]*np.mean(np.diff(eigvals)[0:6]))
#plt.plot(eigvals, np.e**(res[0]*eigvals + res[1]))
plt.semilogy()
#plt.ylabel(r'$\sum_i \int_0^1 ds |\langle j |c_i| 0 \rangle|^2$')
#plt.xlabel('final independendent set size')
plt.ylabel(r'$\sum_i|\langle j |c_i| k \rangle|^2$')
plt.xlabel(r'$|E_j-E_k|$')
plt.show()




