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

#locs = [-0.5206518294234064, -0.5595664226707892, -0.5868622898625512, -0.6082182166127748, -0.6253040187679393, -0.6394700731418002, -0.6513727029209592, -0.6615474835798988, -0.6703349691017637, -0.6780090377333616, -0.6847658203555493, -0.6907593783108079, -0.6961156175091332]
#gaps = [0.42027554189651806, 0.3688630074733279, 0.3280879175009712, 0.29518856916642733, 0.26820854085939416, 0.24569931827476488, 0.22665459114195308, 0.21033504635970068, 0.19619938943156612, 0.18383800598136268, 0.1729379455542741, 0.16325512712449708, 0.15459695414522656]

gaps = [0.10155676970089367, 0.05242872169532298, 0.0237620069354314, 0.009489537663579739, 0.003386274411392165, 0.0010949758428111522, 0.00032436690155179804, 8.876367147436781e-05, 2.25876825119542e-05, 5.37418613077989e-06]
locs = [-0.9453655006884185, -1.0658844153857516, -1.1782880050303366, -1.2817038837605843, -1.3770147539298574, -1.4657175647659646, -1.549077597576822, -1.6280156057824582, -1.7032036975793363, -1.7751506491474562]

gaps_se = [0.6507464785112731, 0.5306244826950106, 0.4294148029021505, 0.3443606458462565, 0.27362005112989785, 0.2155139293920456, 0.16837756685341887, 0.13058329565889082, 0.1005986312624394, 0.07703439603222861, 0.05867169698884567, 0.04446951120908338]
locs_se = [0.9280423128042129, 1.1551422471924153, 1.3233209086120739, 1.4567083664672107, 1.5680540063065838, 1.6647197406825602, 1.7512090135531067, 1.8303812123700365, 1.9041023913851918, 1.973613843204396, 2.039763480418017, 2.10314030423091]
#gaps_se = [0.6319285540626689, 0.4964715914196285, 0.3855665296880364, 0.2936109574674646, 0.2184588559627425, 0.158368938205788, 0.11167060723376032, 0.07655932039720525, 0.05106986374146416, 0.03319656200540422]

ns = np.array(range(11, 37, 2))[:len(gaps)]
ns_se = np.array(range(11, 37, 2))[:len(gaps_se)]
fig, ax = plt.subplots(1, 2)
ax[0].scatter(ns, gaps, edgecolor='navy', color='cornflowerblue', label='Bad guy')
ax[0].scatter(ns_se, gaps_se, edgecolor='navy', color='pink', label='Good guy')
#ax[0].plot(ns_se, 100*2.**(-ns_se/2), color='k')
#ax[0].plot(ns_se, 100*2.**(-ns_se/4), color='k')
#ax[0].scatter(35, 0.15459695414557828)
ax[1].scatter(ns, np.abs(locs), color='red', edgecolor='firebrick')
ax[1].scatter(ns_se, np.abs(locs_se), color='pink', edgecolor='firebrick')
#ax[1].scatter(ns[1:], np.diff(np.abs(locs)), color='red', edgecolor='firebrick')
#ax[0].semilogy()
ax[0].semilogy()
ax[1].loglog()
ax[0].set_xlabel('System length $L$')
#ax[1].set_xlabel('System length $L$')
ax[0].set_ylabel('Minimum adiabatic gap')
#ax[1].set_ylabel(r'$\|\delta/\Omega\|$ of minimum adiabatic gap')
res, err = scipy.optimize.curve_fit(lambda x, a, b: a*x**b, ns, gaps)
print(res)
#ax[0].plot(ns, res[0]*ns**res[1], color='k', label='$y={} x^{{{}}}$'.format(np.round(res[0], 2), np.round(res[1], 2)))
res, err = scipy.optimize.curve_fit(lambda x, a, b: a*np.exp(x*b), ns, gaps, p0=[1, -.05])
#ax[0].plot(ns, res[0]*np.exp(ns*res[1]), color='k', linestyle='dashed', label='$y={} e^{{{}x}}$'.format(np.round(res[0], 2), np.round(res[1], 2)))
print(res)

#res, err = scipy.optimize.curve_fit(lambda x, a, b, c: a*x**b+c, ns, np.abs(locs), p0=[-1, -3, .7])
#ax[1].plot(ns, res[0]*ns**res[1]+res[2], color='k', linestyle='dashed', label='$y=2.24 x^{-.88}+.8$')
ax[0].legend(frameon=False)
#ax[1].legend(frameon=False)
#ax[0].loglog()
#ax[1].semilogx()
print(res)
plt.show()
#raise Exception

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
        order = 1
        rows = []
        columns = []
        entries = []
        independent_sets_dict = {}
        for i in independent_sets:
            if current_size != len(i):
                # We are incrementing in independent set size
                # First, we need to convert our current configuration graph to an adjacency matrix and update the
                # relevant Hamiltonian block
                if current_size != 0:
                    subspace_matrix = sparse.csc_matrix((entries, (rows, columns)), shape=(order, order))
                else:
                    subspace_matrix = sparse.csr_matrix((1, 1))
                subspace_matrices.append(subspace_matrix)

                # Now, we need to reset our configuration graph
                current_size = len(i)
                independent_sets_dict = {}
                rows = []
                columns = []
                entries = []
                order = 0

            independent_sets_dict[tuple(i)] = order
            # Now we need to add its neighbors

            neighbors_sol = neighbors(i)
            for neighbor_config in neighbors_sol:
                if neighbor_config in independent_sets_dict:
                    entries.append(1)
                    entries.append(1)
                    rows.append(independent_sets_dict[neighbor_config])
                    columns.append(order)
                    columns.append(independent_sets_dict[neighbor_config])
                    rows.append(order)
            order += 1
        subspace_matrix = sparse.csc_matrix((entries, (rows, columns)), shape=(order, order))
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


def generate_detuning(n, index):
    graph_mask = np.reshape(np.loadtxt('configurations/mis_degeneracy_L%d.dat' % n)[index, 3:],
                            (n, n), order='F')[::-1, ::-1].T.astype(bool)
    graph = unit_disk_grid_graph(graph_mask, visualize=False)
    print('beginning detuning')
    detuning = HamiltonianMIS(IS_subspace=True, graph=graph)
    print('finished detuning')
    sparse.save_npz('detuning_{}x{}_{}.npz'.format(n, n, index), detuning._hamiltonian)
    return detuning._hamiltonian


def generate_spin_exchange(n, index):
    graph_mask = np.reshape(np.loadtxt('configurations/mis_degeneracy_L%d.dat' % n)[index, 3:],
                            (n, n), order='F')[::-1, ::-1].T.astype(bool)
    graph = unit_disk_grid_graph(graph_mask, visualize=False)
    print('beginning spin exchange')
    spin_exchange = HamiltonianSpinExchange(graph)
    print('finished spin exchange')
    sparse.save_npz('spin_exchange_{}x{}_{}.npz'.format(n, n, index), spin_exchange._hamiltonian)
    print('beginning onsite term')
    onsite_term = HamiltonianOnsiteTerm(spin_exchange)
    print('finished onsite term')
    sparse.save_npz('onsite_term_{}x{}_{}.npz'.format(n, n, index), onsite_term._hamiltonian)
    return spin_exchange._hamiltonian, onsite_term._hamiltonian


def generate_spin_flip(n, index):
    graph_mask = np.reshape(np.loadtxt('configurations/mis_degeneracy_L%d.dat' % n)[index, 3:],
                            (n, n), order='F')[::-1, ::-1].T.astype(bool)
    graph = unit_disk_grid_graph(graph_mask, visualize=False)
    print('beginning spin flip')
    spin_flip = HamiltonianDriver(IS_subspace=True, graph=graph)
    print('finished spin flip')
    sparse.save_npz('spin_flip_{}x{}_{}.npz'.format(n, n, index), spin_flip._hamiltonian)
    return spin_flip._hamiltonian


def find_gap(n, index, guess=None):
    try:
        detuning = sparse.load_npz('detuning_{}x{}_{}.npz'.format(n, n, index))
    except:
        detuning = generate_detuning(n, index)
    try:
        spin_exchange = sparse.load_npz('spin_exchange_{}x{}_{}.npz'.format(n, n, index))
        onsite_term = sparse.load_npz('onsite_term_{}x{}_{}.npz'.format(n, n, index))
    except:
        spin_exchange, onsite_term = generate_spin_exchange(n, index)
    try:
        spin_flip = sparse.load_npz('spin_flip_{}x{}_{}.npz'.format(n, n, index))
    except:
        spin_flip = generate_spin_flip(n, index)

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
        ubs = np.linspace(.94, .999, 20)
        lb = .90
    else:
        ubs = np.linspace(guess-0.01, .99, 15)
        lb = guess-.05
    res = None
    for ub in ubs:
        res = minimize(gap, [lb], bounds=[[lb, ub]])
        if res.x[0] != ub and res.x[0] != lb:
            break
        else:
            lb = ub
    return res


def plot_gap(n, index, k=10):
    try:
        detuning = sparse.load_npz('detuning_{}x{}_{}.npz'.format(n, n, index))
    except:
        detuning = generate_detuning(n, index)
    try:
        spin_exchange = sparse.load_npz('spin_exchange_{}x{}_{}.npz'.format(n, n, index))
        onsite_term = sparse.load_npz('onsite_term_{}x{}_{}.npz'.format(n, n, index))
    except:
        spin_exchange, onsite_term = generate_spin_exchange(n, index)
    try:
        spin_flip = sparse.load_npz('spin_flip_{}x{}_{}.npz'.format(n, n, index))
    except:
        spin_flip = generate_spin_flip(n, index)
    def gap(s):
        if not isinstance(s, float):
            s = s[0]
        o = 1 - 4 * (s - 1 / 2) ** 2
        d = 2 * (s - 1 / 2)
        h = o * (spin_flip + spin_exchange + onsite_term) + d * detuning
        eigval, eigvec = scipy.sparse.linalg.eigsh(h, which='LA', k=k)
        return eigval

    for s in np.linspace(.8, .98, 20):
        print(s)
        eigval = gap(s)
        print(eigval[-1]-eigval)
        plt.scatter(np.ones_like(eigval) * s, eigval[-1]-eigval, color='navy')
    plt.show()

def benchmark_delocalization(n, index, k=2):
    try:
        detuning = sparse.load_npz('detuning_{}x{}_{}.npz'.format(n, n, index))
    except:
        detuning = generate_detuning(n, index)
    try:
        #assert False
        onsite_term = sparse.load_npz('onsite_term_{}x{}_{}.npz'.format(n, n, index))
        spin_exchange = sparse.load_npz('spin_exchange_{}x{}_{}.npz'.format(n, n, index))
    except:
        spin_exchange, onsite_term = generate_spin_exchange(n, index)
    try:
        spin_flip = sparse.load_npz('spin_flip_{}x{}_{}.npz'.format(n, n, index))
    except:
        spin_flip = generate_spin_flip(n, index)
    #plt.plot(onsite_term.diagonal())
    #plt.show()
    def gap(s):
        if not isinstance(s, float):
            s = s[0]
        h = s * spin_flip + detuning - s**2*onsite_term#s * (-onsite_term + spin_exchange)
        eigval, eigvec = scipy.sparse.linalg.eigsh(h, which='LA', k=k+1)
        print(eigval)
        return eigvec, eigval

    def gap_flipped(s):
        if not isinstance(s, float):
            s = s[0]
        h = s * spin_flip + detuning + 10*(-onsite_term + spin_exchange) #+ s**2*onsite_term#s *
        eigval, eigvec = scipy.sparse.linalg.eigsh(h, which='LA', k=2)
        print(eigval)
        return eigvec, eigval

    def gap_exchange(s):
        if not isinstance(s, float):
            s = s[0]
        h = s * spin_flip + detuning + s * (-onsite_term + spin_exchange)
        eigval, eigvec = scipy.sparse.linalg.eigsh(h, which='LA', k=k+1)
        print(eigval)
        return eigvec, eigval
    energies = detuning.diagonal().astype(int)
    MIS_size = np.max(energies)
    mis = (energies == MIS_size)
    misminusone = (energies == MIS_size - 1)
    mis = mis / np.linalg.norm(mis)
    misminusone = misminusone / np.linalg.norm(misminusone)
    #print('hi')
    #plt.plot(np.abs((-onsite_term + spin_exchange)@misminusone))
    #plt.show()
    #print(np.dot(misminusone, (-onsite_term + spin_exchange)@misminusone))
    for (i, t) in enumerate(np.linspace(.1, .5, 10)):
        print(i, t)
        eigvec, eigval = gap(t)
        ground = eigvec[:, -1].flatten()
        excited = eigvec[:, 0].flatten()
        ground_overlap = np.dot(mis, ground)**2
        excited_overlap = np.dot(misminusone, excited)**2
        plt.scatter(t, ground_overlap, color='navy')
        plt.scatter(t, excited_overlap, color='cornflowerblue')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].plot(mis)
        #ax[0].plot(ground)
        #ax[1].plot(misminusone)
        #ax[1].plot(excited)
        #ax[1].plot(onsite_term.diagonal())
        #plt.show()
        eigvec, eigval = gap_flipped(t)
        ground = eigvec[:, -1].flatten()
        excited = eigvec[:, 0].flatten()
        ground_overlap = np.dot(mis, ground) ** 2
        excited_overlap = np.dot(misminusone, excited) ** 2
        plt.scatter(t, ground_overlap, color='navy', marker='*')
        plt.scatter(t, excited_overlap, color='cornflowerblue', marker='*')
        eigvec, eigval = gap_exchange(t)
        ground = eigvec[:, -1].flatten()
        excited = eigvec[:, 0].flatten()
        ground_overlap = np.dot(mis, ground) ** 2
        excited_overlap = np.dot(misminusone, excited) ** 2
        plt.scatter(t, ground_overlap, color='navy', marker='s')
        plt.scatter(t, excited_overlap, color='cornflowerblue', marker='s')

    plt.show()

gaps_7_se = np.array([0.31614461264071103, 0.22126365060525188, 0.2801061546574424, 0.2669887992187636, 0.3895843417678453,
             0.2847023034034617, 0.329333665341208, 0.37488571589901554])
locs_7_se = np.array([0.4637427170202207, 0.4711786111436284, 0.46105008671776415, 0.4591331567510377, 0.471543056697364,
             0.5238552416088684, 0.48154636450408705, 0.5189031535176792])
gaps_8_se = np.array([0.05803955258869209, 0.08743735203890779, 0.08239795153133045])
locs_8_se = np.array([0.3666666766666667, 0.38599393459868875 , 0.43362819690546534])

gaps_8_se_strong = np.array([np.nan, np.nan, 0.2673259118459903])
locs_8_se_strong = np.array([np.nan, np.nan, 0.3666666666666667])
gaps_7_se_strong = np.array([0.3893625239607097, 0.4348765400111603, 0.41929632442612785, 0.38936252396080917, 0.434876540010416,
                             0.41929632442603904, 0.4021218105700637, 0.41603480157175987])
locs_7_se_strong = np.array([0.393106697929194, 0.3533494432030669, 0.37094140145718096, 0.3931066915653211, 0.353349504891713,
                             0.3709414694605734, 0.3451051635510616, 0.385069579649524])
ratios_7 = np.array([126.0, 125.33333333333333, 116.66666666666667, 105.16666666666667, 101.5, 100.0, 87.33333333333333, 78.5,
            77.75, 77.0, 77.0, 77.0, 76.0, 76.0, 75.0, 74.0, 74.0, 74.0, 73.66666666666667, 73.0])
ratios_8 = np.array([814.0, 787.0, 303.0, 232.8409090909091, 166.7, 154.9, 153.27777777777777, 138.53658536585365,
                   136.88194444444446, 134.26363636363635, 132.8909090909091, 131.50515463917526,
                   130.13095238095238, 128.75850340136054, 125.97849462365592, 121.625, 119.125, 116.51219512195122,
                   116.23333333333333, 113.69506726457399])
plt.scatter(ratios_7[:len(gaps_7_se_strong)], gaps_7_se_strong/locs_7_se_strong, color='navy')
plt.scatter(ratios_8[:len(gaps_8_se_strong)], gaps_8_se_strong/locs_8_se_strong, color='navy')
plt.scatter(ratios_7[:len(gaps_7_se)], gaps_7_se/locs_7_se, color='cornflowerblue')
plt.scatter(ratios_8[:len(gaps_8_se)], gaps_8_se/locs_8_se, color='cornflowerblue')

#plt.scatter(ratios_7[:len(gaps_7_se_strong)], gaps_7_se_strong, color='navy')
#plt.scatter(ratios_8[:len(gaps_8_se_strong)], gaps_8_se_strong, color='navy')
#plt.scatter(ratios_7[:len(gaps_7_se)], gaps_7_se, color='cornflowerblue')
#plt.scatter(ratios_8[:len(gaps_8_se)], gaps_8_se, color='cornflowerblue')
ratios = np.concatenate([ratios_7, ratios_8])
plt.plot(ratios, 70*ratios**-1, color='k')
plt.plot(ratios, 7*ratios**-.5, color='red')
print(np.linspace(0.3447141378327797, 0.366666676666666, 2))
plt.loglog()

plt.show()


ratios_7 = [126.0, 125.33333333333333, 116.66666666666667, 105.16666666666667, 101.5, 100.0, 87.33333333333333, 78.5,
            77.75, 77.0, 77.0, 77.0, 76.0, 76.0, 75.0, 74.0, 74.0, 74.0, 73.66666666666667, 73.0]
gaps_7 = [0.21448831139861113, 0.11787331939737555, 0.1643879599626885, 0.16458804508436486, np.nan, 0.16046917600502297, 0.17556791209388223, 0.2511652744516226, 0.24485586745304389, 0.23005455540907604, 0.19859785811932085, 0.21876993563299862, 0.25904544467772084,
          0.28032094117290995, 0.24485586745304389, 0.23005455540907604, 0.19859785811932085, 0.21876993563299862, np.nan, np.nan]
locs_7 = [0.5642576949236335, 0.5345547326675147, 0.5134013528012862, 0.5581102362395576, np.nan, 0.584150828986268, 0.5514058183186268, 0.5894441343409658, 0.5423098432376741, 0.5739099000068317, 0.610322929635107, 0.5960664116893826, 0.5766223897624312,
          0.5999025339592823, 0.5423098432376741, 0.5739099000068317, 0.610322929635107, 0.5960664116893826, np.nan, np.nan]
gaps_8 = [ 0.011456102830845083, 0.02070286772759289, 0.01632697249234738, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
locs_8 = [0.40018875097395956, 0.3951513986274328, 0.4655306740936042, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
ratios_8 = np.array([814.0, 787.0, 303.0, 232.8409090909091, 166.7, 154.9, 153.27777777777777, 138.53658536585365,
                   136.88194444444446, 134.26363636363635, 132.8909090909091, 131.50515463917526,
                   130.13095238095238, 128.75850340136054, 125.97849462365592, 121.625, 119.125, 116.51219512195122,
                   116.23333333333333, 113.69506726457399])
plt.scatter(ratios_7, np.array(gaps_7)/np.array(locs_7), color='purple')
plt.scatter(ratios_8, np.array(gaps_8)/np.array(locs_8), color='navy')
ratios = np.concatenate([ratios_8, ratios_7])
plt.plot(ratios, 50*ratios**-1)
plt.plot(ratios, 5*ratios**-.5)
plt.loglog()
"""plt.show()

plt.clf()
for i in range(20):
    generate_spin_exchange(5, i)

raise Exception"""
if __name__ == '__main__':
    indices_7 = np.array([189, 623, 354, 40, 323, 173, 661, 345, 813, 35, 162, 965, 336,
                          667, 870, 1, 156, 901, 576, 346])
    indices_8 = np.array(
        [188, 970, 91, 100, 72, 316, 747, 216, 168, 852, 7, 743, 32, 573, 991, 957, 555, 936, 342, 950])
    locs_7 = [0.9525957098164666, 0.9509759165213144, 0.9564276559574972, 0.9454866622519833, 0.9538848947761098, 0.9417274305959268, 0.9460157264398306, 0.9552535321757318, 0.9534812943172747, 0.952867728335307, 0.9483063448432979, 0.9553106649059776, 0.9579838153045512, 0.9610932489055783, 0.953116825169182, 0.9582257599778088, 0.9520453028729342, 0.9548518809658713, 0.9506983347525046, 0.957572423186429]
    locs_8 = [0.9610962746983454, 0.9613753356500784, 0.949221638885922, 0.9568356505932953, 0.9346016143241139, 0.9559102729411983, 0.9470132021311632, 0.9445416785265718]
    n = 5
    #i = int(sys.argv[1])
    i = 2
    index = i
    #index = indices_8[i]
    degeneracy = np.loadtxt('configurations/mis_degeneracy_L%d.dat' % n)[index, 1].astype(int)
    print(degeneracy)
    #plot_gap(n, index)
    benchmark_delocalization(n, index, degeneracy)
    res = find_gap(n, index, guess=locs_8[i])
    print(res.fun, res.x[0])