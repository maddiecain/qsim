import numpy as np
import qsim.evolution.hamiltonian as ham
from qsim.graph_algorithms.graph import line_graph
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, expm


def time_evolve(product_state, disorder, T, method='ED'):
    L = len(disorder)

    graph = line_graph(L)
    heisenberg = ham.HamiltonianHeisenberg(graph, energies=(1/4, 1/4), subspace=0)
    index = np.argwhere(np.sum(np.abs(heisenberg.states-product_state), axis=1) == 0)[0,0]
    state = np.zeros((heisenberg.states.shape[0], 1))
    state[index] = 1

    if method == 'ED':
        disorder_hamiltonian_diag = np.sum((1 / 2 - heisenberg.states) * disorder, axis=1)
        disorder_hamiltonian = np.diag(disorder_hamiltonian_diag)
        eigval, eigvec = np.linalg.eigh(disorder_hamiltonian+heisenberg.hamiltonian)
        return eigvec@(np.multiply(np.e**(-1j*T*np.expand_dims(eigval, axis=-1)),(eigvec.conj().T@state)))
    elif method == 'expm_multiply':
        disorder_hamiltonian = sparse.csc_matrix(
            (np.sum((1 / 2 - heisenberg.states) * disorder, axis=1), (np.arange(heisenberg.states.shape[0]),
                                                                      np.arange(heisenberg.states.shape[0]))),
            shape=(heisenberg.states.shape[0], heisenberg.states.shape[0]))
        return expm_multiply(-1j*T*(disorder_hamiltonian+heisenberg.hamiltonian), state)
    elif method == 'expm':
        disorder_hamiltonian = sparse.csc_matrix(
            (np.sum((1 / 2 - heisenberg.states) * disorder, axis=1), (np.arange(heisenberg.states.shape[0]),
                                                                      np.arange(heisenberg.states.shape[0]))),
            shape=(heisenberg.states.shape[0], heisenberg.states.shape[0]))
        return expm(-1j * T * (disorder_hamiltonian + heisenberg.hamiltonian))@ state
    else:
        raise NotImplementedError

def test_time_evolve(L):
    disorder = np.random.uniform(low=-1, high=1, size=L)
    product_state = np.zeros(len(disorder))
    product_state[:len(disorder)//2] = 1
    res_ED = time_evolve(product_state, disorder, 1e2, method='ED')
    print(np.linalg.norm(res_ED))
    res_expm = time_evolve(product_state, disorder, 1e2, method='expm')
    res_expm_multiply = time_evolve(product_state, disorder, 1e2, method='expm_multiply')
    print(np.sum(np.abs(res_ED-res_expm)))
    print(np.sum(np.abs(res_expm-res_expm_multiply)))

#test_time_evolve(10)