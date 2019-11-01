"""
This is a collection of useful functions that help simulate QAOA efficiently
Has been used on a standard laptop up to system size N=24

Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA

Quick start:
    1) generate graph as a networkx.Graph object
    2) qaoa_fun = minimizable_qaoa_fun(graph)
    3) calculate objective function and gradient as (F, Fgrad) = qaoa_fun(parameters)

"""

import networkx as nx
import numpy as np

from qsim.simulate import Simulate
from qsim import tools, operations

"""
What do we actually want out of a simulation class? What are the heavy uses of this code?
Important functions:
    - Given parameters, start state, Hamiltonian, and noise model, simulate the performance on the full density
    matrix
    - Given the Hamiltonian and depth, find the optimal parameters
I'd like this to be as general as possible - Take in Hamiltonians, associate a variational parameter with
each. 
"""


class SimulateQAOA(Simulate):
    def __init__(self, graph: nx.Graph, unitaries=None, p: int, noise_model=None, flag_z2_sym=False,
                 node_to_index_map=None, noisy = False):
        super().__init__(noise_model)
        r"""Unitaries is a dictionary formatted in the following way:
        For each unitary to apply, associate a zero-indexed number that represents the order to apply in a given
        cycle. That should be the dictionary key. Then there should be sub-keys:
            - '
            - 'evolve': function that takes in variational parameter and state and acts on that state like
            e^{-i*v*H}, where v is the variational parameter and H is the Hamiltonian
        """
        self.graph = graph
        self.unitaries = unitaries
        self.node_to_index_map = node_to_index_map
        if self.node_to_index_map is None:
            self.node_to_index_map = {q: i for i, q in enumerate(self.graph.nodes)}
        self.N = self.graph.number_of_nodes()
        self.noisy = noisy
        self.noise_model = noise_model
        # Depth of circuit
        self.p = p
        self.flag_z2_sym = flag_z2_sym

    def create_ZZ_HamC(self):
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.

        Setting flag_z2_sym to True means we're only working in the X^\otimes N = +1
        symmetric sector which reduces the Hilbert space dimension by a factor of 2.
        (default: True to save memory)
        """
        HamC = np.zeros([2 ** N, 1])
        SZ = np.asarray([[1], [-1]])

        for a, b in self.graph.edges:
            HamC += self.graph[a][b]['weight'] * operations.ham_two_local_term(SZ, SZ, node_to_index_map[a],
                                                                            node_to_index_map[b], self.N)
        if self.flag_z2_sym:
            return HamC[range(2 ** (self.N - 1)), 0]  # restrict to first half of Hilbert space
        else:
            return HamC[:, 0]

    def qaoa_expectation_and_variance_fun(self):
        r"""
        Given a graph, return a function f(param) that outputs the expectation,
        variance, and evolved wavefunction for the QAOA evaluated at the parameters.

        The convention here is that C = \sum_{<ij> \in graph} w_{ij} Z_i Z_j
        """
        HamC = create_ZZ_HamC(self.graph, self.flag_z2_sym, self.node_to_index_map)
        return lambda param: ising_qaoa_expectation_and_variance(
            self.N, HamC, param, self.flag_z2_sym)

    def evolve_B(self, state, beta):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        if not self.flag_z2_sym:
            state.dmatrix = operations.rotate_all_qubit(state.dmatrix, self.N, beta, tools.SIGMA_X_IND)
        else:
            state.dmatrix = operations.rotate_all_qubit(state.dmatrix, self.N - 1, beta, 1)
            state.dmatrix = np.cos(beta) * state.dmatrix - 1j * np.sin(beta) * np.flipud(state.dmatrix)

    def minimizable_qaoa_fun(graph: nx.Graph, flag_z2_sym=True, node_to_index_map=None):
        r"""
        Given a graph, return a function f(param) that outputs a tuple (F, Fgrad), where
            F = <C> is the expectation value of objective function with respect
                to the QAOA wavefunction
            Fgrad = the analytically calculated gradient of F with respect to
                the QAOA parameters

        This can then be passed to scipy.optimize.minimize, with jac=True

        The convention here is that C = \sum_{<ij> \in graph} w_{ij} Z_i Z_j
        """
        HamC = create_ZZ_HamC(graph, flag_z2_sym, node_to_index_map)
        N = graph.number_of_nodes()
        return lambda param: ising_qaoa_grad(N, HamC, param, flag_z2_sym)

    def ising_qaoa_grad(N, HamC, param, flag_z2_sym=False):
        """ For QAOA on Ising problems, calculate the objective function F and its
            gradient exactly

            Input:
                N = number of spins
                HamC = a vector of diagonal of objective Hamiltonian in Z basis
                param = parameters of QAOA, should be 2*p in length
                flag_z2_sym = if True, we're only working in the Z2-symmetric sector
                        (saves Hilbert space dimension by a factor of 2)
                        default set to False because we can take more general HamC
                        that is not Z2-symmetric

            Output: (F, Fgrad)
               F = <HamC> for minimization
               Fgrad = gradient of F with respect to param
        """
        p = len(param) // 2
        self.gamma = param[:p]
        self.beta = param[p:]

        evolve_by_HamB_local = lambda beta, psi: evolve_by_HamB(N, beta, psi, flag_z2_sym, False)

        # Preallocate space for storing 2p+2 copies of wavefunction - necessary for efficient computation of analytic gradient
        if flag_z2_sym:
            psi_p = np.zeros([2 ** (N - 1), 2 * p + 2], dtype=complex)
            psi_p[:, 0] = 1 / 2 ** ((N - 1) / 2)
        else:
            psi_p = np.zeros([2 ** N, 2 * p + 2], dtype=complex)
            psi_p[:, 0] = 1 / 2 ** (N / 2)

        # Evolving forward
        for q in range(p):
            psi_p[:, q + 1] = evolve_by_HamB_local(self.beta[q], np.exp(-1j * self.gamma[q] * HamC) * psi_p[:, q])

        # Multiply by HamC
        psi_p[:, p + 1] = HamC * psi_p[:, p]

        # Evolving backwards
        for q in range(p):
            psi_p[:, p + 2 + q] = np.exp(1j * self.gamma[p - 1 - q] * HamC) * evolve_by_HamB_local(-self.beta[p - 1 - q],
                                                                                               psi_p[:, p + 1 + q])

        # Evaluating objective function
        F = np.real(np.vdot(psi_p[:, p], psi_p[:, p + 1]))

        # evaluating gradient analytically
        Fgrad = np.zeros(2 * p)
        for q in range(p):
            Fgrad[q] = -2 * np.imag(np.vdot(psi_p[:, q], HamC * psi_p[:, 2 * p + 1 - q]))

            if not flag_z2_sym:
                psi_temp = np.zeros(2 ** N, dtype=complex)
                for i in range(N):
                    psi_temp += operations.multiply_single_qubit(psi_p[:, 2 * p - q], i, tools.SIGMA_X_IND)
            else:
                psi_temp = np.zeros(2 ** (N - 1), dtype=complex)
                for i in range(N - 1):
                    psi_temp += operations.multiply_single_qubit(psi_p[:, 2 * p - q], i, tools.SIGMA_X_IND)
                psi_temp += np.flipud(psi_p[:, 2 * p - q])

            Fgrad[p + q] = -2 * np.imag(np.vdot(psi_p[:, q + 1], psi_temp))

        return (F, Fgrad)

    def ising_qaoa_expectation_and_variance(self, HamC):
        """For QAOA on Ising problems, calculate the expectation, variance, and
        wavefunction.

        Input:
            N = number of spins
            HamC = a vector of diagonal of objective Hamiltonian in Z basis
            param = parameters of QAOA, should be 2*p in length
            flag_z2_sym = if True, we're only working in the Z2-symmetric sector
                    (saves Hilbert space dimension by a factor of 2)
                    default set to False because we can take more general HamC
                    that is not Z2-symmetric

         Output: (expectation, variance, wavefunction)
        """
        def evolve_by_HamB_local(beta, psi):
            return evolve_B(psi,  False)

        if self.flag_z2_sym:
            psi = np.empty(2 ** (self.N - 1), dtype=complex)
            psi[:] = 1 / 2 ** ((self.N - 1) / 2)
        else:
            psi = np.empty(2 ** self.N, dtype=complex)
            psi[:] = 1 / 2 ** (self.N / 2)

        for q in range(self.p):
            psi = evolve_by_HamB_local(self.beta[q], np.exp(-1j * self.gamma[q] * HamC) * psi)

        expectation = np.real(np.vdot(psi, HamC * psi))
        variance = np.real(np.vdot(psi, HamC ** 2 * psi) - expectation ** 2)

        return expectation, variance, psi

    def run(self):
        if not noisy:
            state = np.ones((2**self.N, 1))
        else:
            state = np.ones((2**self.N, 2**self.N))
        for i in range(self.p):
            for j in self.hamiltonians:
                j['evolve'](state)
                # Apply H_C
                # Apply H_B
            if self.noise_model:
                pass

    def find_optimal_params(self):
        pass
