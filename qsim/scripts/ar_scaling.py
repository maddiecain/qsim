import itertools

import numpy as np
import scipy.integrate
import scipy.optimize
from odeintw import odeintw
import matplotlib.pyplot as plt
import networkx as nx
import sys

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


"""
What do we want to do here?
We want to understand how the non-adiabatic corrections and noise scale with the system size. For simplicity, let's
start with noise. We know that a very small fraction of the eigenstates actually matter. 

We want to keep things simple, so let's solve for the approximation ratio at a fixed point in time (t/T = .5) for a length
n line graph. First, we can store the number of eigenstates that were below the threshold for previous system sizes. 
We can then fit that to a low degree polynomial to get an estimate of the number of eigenstates that will matter for the 
next system size. We can add a few more to the results to be safe.

We can then compute the total leakage from the ground state, and the total leakage into the states that we solved for.
We can verify that it was sufficient for the threshold. 


"""


class EffectiveOperatorHamiltonian(object):
    def __init__(self, omega_g, omega_r, energies=(1,), graph: Graph = None, IS_subspace=True, code=qubit):
        # Just need to define self.hamiltonian
        assert IS_subspace
        self.energies = energies
        self.IS_subspace = IS_subspace
        self.graph = graph
        self.omega_r = omega_r
        self.omega_g = omega_g
        self.code = code
        assert self.code is qubit

        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            if code is not qubit:
                IS, num_IS = graph.independent_sets_qudit(self.code)
            else:
                # We have already solved for this information
                IS, num_IS = graph.independent_sets, graph.num_independent_sets
            self.transition = (0, 1)
            self._hamiltonian_rr = np.zeros((num_IS, num_IS))
            self._hamiltonian_gg = np.zeros((num_IS, num_IS))
            self._hamiltonian_cross_terms = np.zeros((num_IS, num_IS))
            for k in range(IS.shape[0]):
                self._hamiltonian_rr[k, k] = np.sum(IS[k][1] == self.transition[0])
                self._hamiltonian_gg[k, k] = np.sum(IS[k][1] == self.transition[1])
            self._csc_hamiltonian_rr = sparse.csc_matrix(self._hamiltonian_rr)
            self._csc_hamiltonian_gg = sparse.csc_matrix(self._hamiltonian_gg)
            # For each IS, look at spin flips generated by the laser
            # Over-allocate space
            rows = np.zeros(graph.n * num_IS, dtype=int)
            columns = np.zeros(graph.n * num_IS, dtype=int)
            entries = np.zeros(graph.n * num_IS, dtype=float)
            num_terms = 0
            for i in range(graph.num_independent_sets):
                for j in range(graph.n):
                    if IS[i,j] == self.transition[1]:
                        # Flip spin at this location
                        # Get binary representation
                        temp = IS[i].copy()
                        temp[j] = self.transition[0]
                        where_matched = (np.argwhere(np.sum(np.abs(IS - temp), axis=1) == 0).flatten())
                        if len(where_matched) > 0:
                            # This is a valid spin flip
                            rows[num_terms] = where_matched[0]
                            columns[num_terms] = i
                            entries[num_terms] = 1
                            num_terms += 1
            # Cut off the excess in the arrays
            columns = columns[:2 * num_terms]
            rows = rows[:2 * num_terms]
            entries = entries[:2 * num_terms]
            # Populate the second half of the entries according to self.pauli
            columns[num_terms:2 * num_terms] = rows[:num_terms]
            rows[num_terms:2 * num_terms] = columns[:num_terms]
            entries[num_terms:2 * num_terms] = entries[:num_terms]
            # Now, construct the Hamiltonian
            self._csc_hamiltonian_cross_terms = sparse.csc_matrix((entries, (rows, columns)), shape=(num_IS, num_IS))
            self._hamiltonian_cross_terms = self._csc_hamiltonian_cross_terms

        else:
            # We are not in the IS subspace
            pass

    @property
    def hamiltonian(self):
        return self.energies[0] * (self.omega_g * self.omega_r * self._hamiltonian_cross_terms +
                                   self.omega_g ** 2 * self._csc_hamiltonian_gg +
                                   self.omega_r ** 2 * self._csc_hamiltonian_rr)

    def left_multiply(self, state: State):
        return self.hamiltonian @ state

    def right_multiply(self, state: State):
        return state @ self.hamiltonian

    def evolve(self, state: State, time):
        if state.is_ket:
            return State(expm_multiply(-1j * time * self.hamiltonian, state),
                         is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)

        else:
            exp_hamiltonian = expm(-1j * time * self.hamiltonian)
            return State(exp_hamiltonian @ state @ exp_hamiltonian.conj().T,
                         is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)


class EffectiveOperatorDissipation(lindblad_operators.LindbladJumpOperator):
    def __init__(self, omega_g, omega_r, rates=(1,), graph: Graph = None, IS_subspace=True, code=qubit):
        self.omega_g = omega_g
        self.omega_r = omega_r

        self.IS_subspace = IS_subspace
        self.transition = (0, 1)
        self.graph = graph
        # Construct jump operators
        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            if code is not qubit:
                IS, num_IS = graph.independent_sets_qudit(self.code)
            else:
                # We have already solved for this information
                IS, num_IS = graph.independent_sets, graph.num_independent_sets
            self._jump_operators_rg = []
            self._jump_operators_gg = []
            # For each atom, consider the states spontaneous emission can generate transitions between
            # Over-allocate space
            for j in range(graph.n):
                rows_rg = np.zeros(num_IS, dtype=int)
                columns_rg = np.zeros(num_IS, dtype=int)
                entries_rg = np.zeros(num_IS, dtype=int)
                rows_gg = np.zeros(num_IS, dtype=int)
                columns_gg = np.zeros(num_IS, dtype=int)
                entries_gg = np.zeros(num_IS, dtype=int)
                num_terms_gg = 0
                num_terms_rg = 0
                for i in range(IS.shape[0]):
                    if IS[i,j] == self.transition[0]:
                        # Flip spin at this location
                        # Get binary representation
                        temp = IS[i].copy()
                        temp[j] = self.transition[1]
                        where_matched = (np.argwhere(np.sum(np.abs(IS - temp), axis=1) == 0).flatten())
                        if len(where_matched) > 0:
                            # This is a valid spin flip
                            rows_rg[num_terms_rg] = where_matched[0]
                            columns_rg[num_terms_rg] = i
                            entries_rg[num_terms_rg] = 1
                            num_terms_rg += 1
                    elif IS[i,j] == self.transition[1]:
                        rows_gg[num_terms_gg] = i
                        columns_gg[num_terms_gg] = i
                        entries_gg[num_terms_gg] = 1
                        num_terms_gg += 1

                # Cut off the excess in the arrays
                columns_rg = columns_rg[:num_terms_rg]
                rows_rg = rows_rg[:num_terms_rg]
                entries_rg = entries_rg[:num_terms_rg]
                columns_gg = columns_gg[:num_terms_gg]
                rows_gg = rows_gg[:num_terms_gg]
                entries_gg = entries_gg[:num_terms_gg]
                # Now, append the jump operator
                jump_operator_rg = sparse.csc_matrix((entries_rg, (rows_rg, columns_rg)), shape=(num_IS, num_IS))
                jump_operator_gg = sparse.csc_matrix((entries_gg, (rows_gg, columns_gg)), shape=(num_IS, num_IS))

                self._jump_operators_rg.append(jump_operator_rg)
                self._jump_operators_gg.append(jump_operator_gg)
            self._jump_operators_rg = np.asarray(self._jump_operators_rg)
            self._jump_operators_gg = np.asarray(self._jump_operators_gg)
        else:
            # self._jump_operators_rg = []
            # self._jump_operators_gg = []
            op_rg = np.array([[[0, 0], [1, 0]]])
            op_gg = np.array([[[0, 0], [0, 1]]])
            self._jump_operators_rg = op_rg
            self._jump_operators_gg = op_gg

        super().__init__(None, rates=rates, graph=graph, IS_subspace=IS_subspace, code=code)

    @property
    def jump_operators(self):
        return np.sqrt(self.rates[0]) * (self.omega_g * self._jump_operators_gg +
                                         self.omega_r * self._jump_operators_rg)

    @property
    def liouville_evolution_operator(self):
        if self._evolution_operator is None and self.IS_subspace:
            num_IS = self.graph.num_independent_sets
            self._evolution_operator = sparse.csr_matrix((num_IS ** 2, num_IS ** 2))
            for jump_operator in self.jump_operators:
                # Jump operator is real, so we don't need to conjugate
                self._evolution_operator = self._evolution_operator + sparse.kron(jump_operator,
                                                                                  jump_operator) - 1 / 2 * \
                                           sparse.kron(jump_operator.T @ jump_operator, sparse.identity(num_IS)) - \
                                           1 / 2 * sparse.kron(sparse.identity(num_IS), jump_operator.T @ jump_operator)

        elif self._evolution_operator is None:
            # TODO: generate the evolution operator for non-IS subspace states
            raise NotImplementedError
        return self.rates[0] * self._evolution_operator


def alpha_correction_ar(graph=line_graph(n=2), mode='hybrid', angle=np.pi/4, which='S', verbose=False):
    """For REIT, compute the total leakage from the ground state to a given state. Plot the total leakage versus
    the final eigenenergy"""
    if which == 'S':
        index = 0
    elif which == 'L':
        index = graph.num_independent_sets - 1
    else:
        index = which

    energies = []
    bad_indices = []
    target_energy = graph.independent_sets[index][1]
    degeneracy = 0
    # Compute maximum independent set size and "bad" energies
    for IS in graph.independent_sets:
        if graph.independent_sets[IS][1] != target_energy:
            energies.append(graph.independent_sets[IS][1])
            bad_indices.append(IS)
        else:
            degeneracy += 1
    energies = np.array(energies)
    energies = np.abs(target_energy-energies)

    # Good is a list of good eigenvalues
    # Bad is a list of bad eigenvalues. If 'other', defaults to all remaining eigenvalues outside of 'good'
    def schedule_hybrid(t, tf):
        phi = (tf - t) / tf * np.pi / 2
        energy_shift.energies = (np.sin(2*((tf - t) / tf-1/2) * np.pi),)
        laser.omega_g = np.cos(phi)
        laser.omega_r = np.sin(phi)
        dissipation.omega_g = np.cos(phi)
        dissipation.omega_r = np.sin(phi)

    def schedule_reit(t, tf):
        phi = (tf - t) / tf * np.pi / 2
        laser.omega_g = np.cos(phi)
        laser.omega_r = np.sin(phi)
        dissipation.omega_g = np.cos(phi)
        dissipation.omega_r = np.sin(phi)

    def schedule_adiabatic(t, tf):
        phi = (tf - t) / tf * np.pi / 2
        energy_shift.energies = (np.sin(2*((tf - t) / tf-1/2) * np.pi),)
        laser.omega_g = np.cos(angle) * np.sin(phi) * np.cos(phi)
        laser.omega_r = np.sin(angle) * np.sin(phi) * np.cos(phi)
        dissipation.omega_g = np.cos(angle) * np.cos(phi) * np.sin(phi)
        dissipation.omega_r = np.sin(angle) * np.cos(phi) * np.sin(phi)

    laser = EffectiveOperatorHamiltonian(graph=graph, IS_subspace=True,
                                         energies=(1,),
                                         omega_g=np.cos(np.pi / 4),
                                         omega_r=np.sin(np.pi / 4))
    energy_shift = hamiltonian.HamiltonianEnergyShift(IS_subspace=True, graph=graph,
                                                      energies=(2.5,), index=0)
    dissipation = EffectiveOperatorDissipation(graph=graph, omega_r=1, omega_g=1,
                                               rates=(1,))
    if mode == 'hybrid':
        schedule = schedule_hybrid
    elif mode == 'adiabatic':
        schedule = schedule_adiabatic
    elif mode == 'reit':
        schedule = schedule_reit
    if mode != 'reit':
        eq = SchrodingerEquation(hamiltonians=[laser, energy_shift])
    else:
        eq = SchrodingerEquation(hamiltonians=[laser])

    def compute_rate(t):
        if verbose:
            print(t)
        schedule(t, 1)
        # Construct the first order transition matrix
        eigval, eigvec = eq.eig(k='all')
        eigvec = eigvec.T
        rates = np.zeros(eigval.shape[0] ** 2)
        for op in dissipation.jump_operators:
            rates = rates + ((eigvec.conj().T @ op @ eigvec)**2).flatten()
        # Select the relevant rates from 'good' to 'bad'

        rates = np.reshape(rates.real, (graph.num_independent_sets, graph.num_independent_sets))
        bad_rates = np.zeros(len(bad_indices))
        for i in range(len(bad_indices)):
            bad_rates[i] = bad_rates[i] + rates[bad_indices[i], index]
        return np.dot(bad_rates, energies)

    # Integrate to find the correction
    res, error = scipy.integrate.quad_vec(compute_rate, 1e-5, .9999)
    return res/target_energy, error

odd_rates = [0.01885195077253434, 0.02927606703168038, .03588440678455766, 0.04056175101661809, 0.0440182976521066, 0.04671109220510976]
even_rates = [0.005499572794943576, 0.005870499486106632, 0.00613223651331744, 0.007199171064989809, 0.0074339667580066125, 0.008037112891670875]
odd = [3, 5, 7, 9, 11, 13]
even = [4, 6, 8, 10, 12, 14]
plt.plot(odd, odd_rates)
plt.scatter(odd, odd_rates)
plt.plot(even, even_rates)
plt.scatter(even, even_rates)
plt.show()
"""n = 3
odd_n = np.arange(3, )
odd_rates = [0.01885195077253434, 0.02927606703168038, .03588440678455766, 0.04056175101661809, 0.0440182976521066]
even_rates = [0.005499572794943576, 0.005870499486106632, 0.00613223651331744, 0.007199171064989809, 0.0074]
for n in range(3, 15):
    graph = line_graph(n)

    #print(graph.num_independent_sets)
    res, error = alpha_correction_ar(graph=graph, mode='reit', which='S')
    print(res, error)"""

rates = [0.010662442042852313, 0.016957472702683322, 0.02374938250983679, 0.03498899357236483, 0.043702391822540175,
         0.054663814324193705, 0.06593339301684226]
diss = [0.2927448927120103, 0.40019282741113876, 0.5126483512648607, 0.6198496428799345, 0.7245172743824079, 0.8360234218829589,
        0.9456842157184615]
print(np.diff(rates))
plt.plot(np.arange(4, 2*len(diss)+4, 2), diss)
plt.scatter(np.arange(4, 2*len(diss)+4, 2), diss)

plt.plot(np.arange(4, 2*len(rates)+4, 2), rates)
plt.scatter(np.arange(4, 2*len(rates)+4, 2), rates, c='black')
plt.show()

if __name__ == "__main__":
    n = int(sys.argv[1])
    res, error = alpha_correction_ar(graph=line_graph(n), mode='reit', which='S')
    print(res, error, flush=True)


"""
Compute <E>:
- function which at a given time, computes all the "bad" rates and dots them with the difference between the final 
eigenvalues and the final ground state energy.
"""
