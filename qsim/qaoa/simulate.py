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

import qsim.ops as qops



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
    return lambda param : ising_qaoa_grad(N, HamC, param, flag_z2_sym)


def create_ZZ_HamC(graph: nx.Graph, flag_z2_sym=True, node_to_index_map=None):
    r"""
    Generate a vector corresponding to the diagonal of the C Hamiltonian

    Setting flag_z2_sym to True means we're only working in the X^\otimes N = +1
    symmetric sector which reduces the Hilbert space dimension by a factor of 2.
    (default: True to save memory)
    """
    N = graph.number_of_nodes()
    HamC = np.zeros([2**N,1])

    SZ = np.asarray([[1],[-1]])
    if node_to_index_map is None:
        node_to_index_map = {q: i for i, q in enumerate(graph.nodes)}

    for a, b in graph.edges:
        HamC += graph[a][b]['weight']*ham_two_local_term(SZ, SZ, node_to_index_map[a], node_to_index_map[b], N)

    if flag_z2_sym:
        return HamC[range(2**(N-1)), 0] # restrict to first half of Hilbert space
    else:
        return HamC[:, 0]


def ham_two_local_term(op1, op2, ind1, ind2, N):
    r"""Utility function for conveniently create a 2-Local term op1 \otimes op2 among N spins"""
    if ind1 > ind2:
        return ham_two_local_term(op2, op1, ind2, ind1, N)

    if op1.shape != op2.shape or op1.ndim != 2:
        raise ValueError('ham_two_local_term: invalid operator input')

    if ind1 < 0 or ind2 > N-1:
        raise ValueError('ham_two_local_term: invalid input indices')

    if op1.shape[0] == 1 or op1.shape[1] == 1:
        myeye = lambda n : np.ones(np.asarray(op1.shape)**n)
    else:
        myeye = lambda n : np.eye(np.asarray(op1.shape)**n)

    return np.kron(myeye(ind1), \
                   np.kron(op1, \
                   np.kron(myeye(ind2-ind1-1), \
                   np.kron(op2, myeye(N-ind2-1)))))



def evolve_by_HamB(N, beta, psi_in, flag_z2_sym=False, copy=True):
    r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
    if copy:
        psi = psi_in.copy()
    else:
        psi = psi_in

    if not flag_z2_sym:
        psi = qops.rotate_all_qubits(psi, N, beta, qops.SIGMA_X_IND)
    else:
        psi = qops.rotate_all_qubits(psi, N-1, beta, qops.SIGMA_X_IND)
        psi = np.cos(beta)*psi - 1j*np.sin(beta)*np.flipud(psi)

    return psi


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
    gammas = param[:p]
    betas = param[p:]

    evolve_by_HamB_local = lambda beta, psi: evolve_by_HamB(N, beta, psi, flag_z2_sym, False)

    # pre-allocate space for storing 2p+2 copies of wavefunction - necessary for efficient computation of analytic gradient
    if flag_z2_sym:
        psi_p = np.zeros([2**(N-1), 2*p+2], dtype=complex)
        psi_p[:,0] = 1/2**((N-1)/2)
    else:
        psi_p = np.zeros([2**N, 2*p+2], dtype=complex)
        psi_p[:,0] = 1/2**(N/2)

    # evolving forward
    for q in range(p):
        psi_p[:,q+1] = evolve_by_HamB_local(betas[q], np.exp(-1j*gammas[q]*HamC)*psi_p[:, q])

    # multiply by HamC
    psi_p[:, p+1] = HamC*psi_p[:, p]

    # evolving backwards
    for q in range(p):
        psi_p[:, p+2+q] = np.exp(1j*gammas[p-1-q]*HamC)*evolve_by_HamB_local(-betas[p-1-q], psi_p[:,p+1+q])

    # evaluating objective function
    F = np.real(np.vdot(psi_p[:,p], psi_p[:,p+1]))

    # evaluating gradient analytically
    Fgrad = np.zeros(2*p)
    for q in range(p):
        Fgrad[q] = -2*np.imag(np.vdot(psi_p[:,q], HamC*psi_p[:,2*p+1-q]))

        if not flag_z2_sym:
            psi_temp = np.zeros(2**N, dtype=complex)
            for i in range(N):
                psi_temp += qops.multiply_single_qubit(psi_p[:,2*p-q], i,
                                                       qops.SIGMA_X_IND)
        else:
            psi_temp = np.zeros(2**(N-1), dtype=complex)
            for i in range(N-1):
                psi_temp += qops.multiply_single_qubit(psi_p[:,2*p-q], i, 
                                                       qops.SIGMA_X_IND)
            psi_temp += np.flipud(psi_p[:, 2*p-q])

        Fgrad[p+q] = -2*np.imag(np.vdot(psi_p[:, q+1], psi_temp))

    return (F, Fgrad)


def ising_qaoa_expectation_and_variance(N, HamC, param, flag_z2_sym=False):
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
    p = len(param) // 2
    gammas = param[:p]
    betas = param[p:]

    def evolve_by_HamB_local(beta, psi):
        return evolve_by_HamB(N, beta, psi, flag_z2_sym, False)

    if flag_z2_sym:
        psi = np.empty(2**(N-1), dtype=complex)
        psi[:] = 1/2**((N-1)/2)
    else:
        psi = np.empty(2**N, dtype=complex)
        psi[:] = 1/2**(N/2)

    for q in range(p):
        psi = evolve_by_HamB_local(betas[q], np.exp(-1j*gammas[q]*HamC)*psi)

    expectation = np.real(np.vdot(psi, HamC * psi))
    variance = np.real(np.vdot(psi, HamC**2 * psi) - expectation**2)

    return expectation, variance, psi


def qaoa_expectation_and_variance_fun(
        graph: nx.Graph,
        flag_z2_sym=True,
        node_to_index_map=None):
    r"""
    Given a graph, return a function f(param) that outputs the expectation,
    variance, and evolved wavefunction for the QAOA evaluated at the parameters.

    The convention here is that C = \sum_{<ij> \in graph} w_{ij} Z_i Z_j
    """
    HamC = create_ZZ_HamC(graph, flag_z2_sym, node_to_index_map)
    N = graph.number_of_nodes()
    return lambda param : ising_qaoa_expectation_and_variance(
            N, HamC, param, flag_z2_sym)
