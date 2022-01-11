import networkx as nx
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import networkx
import scipy.optimize
import scipy.stats
import scipy.stats.distributions
import pandas as pd
from scipy.optimize import minimize
from qsim.codes.quantum_state import State
from qsim.evolution.hamiltonian import HamiltonianDriver, HamiltonianMIS
from qsim.graph_algorithms.graph import Graph, unit_disk_grid_graph, enumerate_independent_sets
from scipy.linalg import expm
import scipy.sparse as sparse
from scipy.sparse.linalg import expm_multiply
import sys


class HamiltonianSpinExchange(object):
    def __init__(self, graph, energies=(1,)):
        """Default is that the first element in transition is the higher energy s."""
        self.energies = energies
        self.IS_subspace = True
        self.graph = graph
        # Generate sparse mixing Hamiltonian
        assert isinstance(graph, Graph)
        independent_sets = enumerate_independent_sets(graph.graph)

        # Generate a list of integers corresponding to the independent sets in binary

        def free_node(n, n_init, sol):
            free = True
            for neighbor in self.graph.graph[n]:
                if neighbor != n_init and (neighbor in sol):
                    free = False
            return free

        def neighbors(sol):
            sol = list(sol)
            candidate_sols = []
            for i in sol:
                for n in self.graph.graph[i]:
                    if free_node(n, i, sol):
                        candidate_sol = sol.copy()
                        candidate_sol.append(n)
                        candidate_sol.remove(i)
                        candidate_sols.append(tuple(candidate_sol))
            return candidate_sols

        subspace_matrices = []
        configuration_graph = nx.Graph()
        current_size = 0
        nodelist = []
        for i in independent_sets:
            if current_size != len(i):
                # We are incrementing in independent set size
                # First, we need to convert our current configuration graph to an adjacency matrix and update the
                # relevant Hamiltonian block
                if current_size != 0:
                    subspace_matrix = nx.to_scipy_sparse_matrix(configuration_graph, format='csc', nodelist=nodelist)
                else:
                    subspace_matrix = sparse.csr_matrix((1, 1))
                subspace_matrices.append(subspace_matrix)

                # Now, we need to reset our configuration graph
                configuration_graph = nx.Graph()
                current_size = len(i)
                nodelist = []
            configuration_graph.add_node(tuple(i))
            nodelist.append(tuple(i))
            # Now we need to add its neighbors

            neighbors_sol = neighbors(i)
            for neighbor_config in neighbors_sol:
                configuration_graph.add_edge(tuple(i), neighbor_config)
        subspace_matrix = nx.to_scipy_sparse_matrix(configuration_graph, format='csc', nodelist=nodelist)
        subspace_matrices.append(subspace_matrix)
        self.mis_size = current_size
        # Now, construct the Hamiltonian
        self._csc_hamiltonian = sparse.block_diag(subspace_matrices, format='csc')
        self._hamiltonian = self._csc_hamiltonian

    @property
    def hamiltonian(self):
        return self.energies[0] * self._hamiltonian

    @property
    def evolution_operator(self):
        return -1j * self.hamiltonian

    def left_multiply(self, state: State):
        return State(self.energies[0] * self._csc_hamiltonian @ state, is_ket=state.is_ket,
                     IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)

    def right_multiply(self, state: State):
        return State(state @ self.hamiltonian.T.conj(), is_ket=state.is_ket, IS_subspace=state.IS_subspace,
                     code=state.code, graph=self.graph)

    def evolve(self, state: State, time):
        r"""
        Use reshape to efficiently implement evolution under :math:`H_B=\\sum_i X_i`
        """
        if state.is_ket:
            # Handle dimensions
            if self.hamiltonian.shape[1] == 1:
                return State(np.exp(-1j * time * self.hamiltonian) * state, is_ket=state.is_ket,
                             IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
            else:
                return State(expm_multiply(-1j * time * self.hamiltonian, state),
                             is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
        else:
            if self.hamiltonian.shape[1] == 1:
                exp_hamiltonian = np.exp(-1j * time * self.hamiltonian)
                return State(exp_hamiltonian * state * exp_hamiltonian.conj().T,
                             is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
            else:
                exp_hamiltonian = expm(-1j * time * self.hamiltonian)
                return State(exp_hamiltonian @ state @ exp_hamiltonian.conj().T,
                             is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)


class HamiltonianOnsiteTerm(object):
    def __init__(self, spin_exchange: HamiltonianSpinExchange, energies=(1,)):
        """Default is that the first element in transition is the higher energy s."""
        self.energies = energies
        self.IS_subspace = True
        self._csc_hamiltonian = sparse.diags(np.asarray(np.sum(spin_exchange._hamiltonian, axis=1).flatten()), offsets=[0])
        self._hamiltonian = self._csc_hamiltonian

    @property
    def hamiltonian(self):
        return self.energies[0] * self._hamiltonian

    @property
    def evolution_operator(self):
        return -1j * self.hamiltonian

    def left_multiply(self, state: State):
        return State(self.energies[0] * self._csc_hamiltonian @ state, is_ket=state.is_ket,
                     IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)

    def right_multiply(self, state: State):
        return State(state @ self.hamiltonian.T.conj(), is_ket=state.is_ket, IS_subspace=state.IS_subspace,
                     code=state.code, graph=self.graph)

    def evolve(self, state: State, time):
        r"""
        Use reshape to efficiently implement evolution under :math:`H_B=\\sum_i X_i`
        """
        if state.is_ket:
            # Handle dimensions
            if self.hamiltonian.shape[1] == 1:
                return State(np.exp(-1j * time * self.hamiltonian) * state, is_ket=state.is_ket,
                             IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
            else:
                return State(expm_multiply(-1j * time * self.hamiltonian, state),
                             is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
        else:
            if self.hamiltonian.shape[1] == 1:
                exp_hamiltonian = np.exp(-1j * time * self.hamiltonian)
                return State(exp_hamiltonian * state * exp_hamiltonian.conj().T,
                             is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
            else:
                exp_hamiltonian = expm(-1j * time * self.hamiltonian)
                return State(exp_hamiltonian @ state @ exp_hamiltonian.conj().T,
                             is_ket=state.is_ket, IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)


def generate_hamiltonians(n, index):
    graph_mask = np.reshape(np.loadtxt('configurations/mis_degeneracy_L%d.dat' % n)[index, 3:],
                            (n, n), order='F')[::-1, ::-1].T.astype(bool)
    graph = unit_disk_grid_graph(graph_mask, visualize=False)
    spin_flip = HamiltonianDriver(IS_subspace=True, graph=graph)
    detuning = HamiltonianMIS(IS_subspace=True, graph=graph)
    spin_exchange = HamiltonianSpinExchange(graph)
    onsite_term = HamiltonianOnsiteTerm(spin_exchange)

    return detuning._hamiltonian, spin_flip._hamiltonian, spin_exchange._hamiltonian, onsite_term._hamiltonian


def find_gap(n, index, guess=None):
    detuning, spin_flip, spin_exchange, onsite_term = generate_hamiltonians(n, index)
    def gap(s):
        if not isinstance(s, float):
            s = s[0]
        o = 1 - 4 * (s - 1 / 2) ** 2
        d = 2 * (s - 1 / 2)
        h = o * (spin_flip + spin_exchange + onsite_term) + d * detuning
        eigval, eigvec = scipy.sparse.linalg.eigsh(h, which='LA', k=2)
        print(s, np.abs(eigval[-2] - eigval[-1]))
        return np.abs(eigval[-2] - eigval[-1])
    if guess is None:
        ubs = np.linspace(.94, .99, 20)
        lb = .90
    else:
        ubs = np.linspace(guess-.01, .99, 20)
        lb = guess-.03
    res = None
    for ub in ubs:
        res = minimize(gap, [lb], bounds=[[lb, ub]])
        if res.x[0] != ub and res.x[0] != lb:
            break
        else:
            lb = ub
    return res


def plot_gap(n, index, k=10):
    detuning, spin_flip, spin_exchange, onsite_term = generate_hamiltonians(n, index)

    def gap(s):
        if not isinstance(s, float):
            s = s[0]
        o = 1 - 4 * (s - 1 / 2) ** 2
        d = 2 * (s - 1 / 2)
        h = o * (spin_flip + spin_exchange + onsite_term) + d * detuning
        eigval, eigvec = scipy.sparse.linalg.eigsh(h, which='LA', k=k)
        return eigval

    for s in np.linspace(.94, .98, 20):
        print(s)
        eigval = gap(s)
        print(eigval[-1]-eigval)
        plt.scatter(np.ones_like(eigval) * s, eigval[-1]-eigval, color='navy')
    plt.show()


if __name__ == '__main__':
    indices_7 = np.array([189, 623, 354, 40, 323, 173, 661, 345, 813, 35, 162, 965, 336,
                          667, 870, 1, 156, 901, 576, 346])
    indices_8 = np.array(
        [188, 970, 91, 100, 72, 316, 747, 216, 168, 852, 7, 743, 32, 573, 991, 957, 555, 936, 342, 950])
    locs_7 = [0.9525957098164666, 0.9509759165213144, 0.9564276559574972, 0.9454866622519833, 0.9538848947761098, 0.9417274305959268, 0.9460157264398306, 0.9552535321757318, 0.9534812943172747, 0.952867728335307, 0.9483063448432979, 0.9553106649059776, 0.9579838153045512, 0.9610932489055783, 0.953116825169182, 0.9582257599778088, 0.9520453028729342, 0.9548518809658713, 0.9506983347525046, 0.957572423186429]
    locs_8 = []
    n = 7
    i = int(sys.argv[1])
    index = indices_7[i]
    degeneracy = np.loadtxt('configurations/mis_degeneracy_L%d.dat' % n)[index, 1].astype(int)
    res = find_gap(n, index, guess=locs_7[i])
    print(res.fun, res.x)