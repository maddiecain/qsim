"""
Single qubit operations
"""

import numpy as np
from .tools import SX, SY, SZ, SIGMA_X_IND, SIGMA_Y_IND, SIGMA_Z_IND

__all__ = ['multiply_single_qubit', 'multiply_single_qubit_mixed', 'operate_single_qubit_mixed',
           'rotate_single_qubit', 'rotate_all_qubit']

def multiply_single_qubit(state, i: int, pauli_ind: int):
    """ Multiply a single pauli operator on the i-th qubit of the input wavefunction

        Input:
            state = input wavefunction (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            pauli_ind = one of (1,2,3) for (X, Y, Z)
    """
    IndL = 2**i

    # note index start from the right (sN,...,s3,s2,s1)
    out = state.reshape((-1, 2, IndL), order='F').copy()

    if pauli_ind == SIGMA_X_IND: # sigma_X
        out = np.flip(out, 1)
    elif pauli_ind == SIGMA_Y_IND: # sigma_Y
        out = np.flip(out, 1).astype(complex, copy=False)
        out[:,0,:] = -1j*out[:,0,:]
        out[:,1,:] = 1j*out[:,1,:]
    elif pauli_ind == SIGMA_Z_IND: # sigma_Z
        out[:,1,:] = -out[:,1,:]

    return out.reshape(state.shape, order='F')


def multiply_single_qubit_mixed(rho, i: int, pauli_ind: int):
    """ Multiply a single pauli operator on the i-th qubit of the input density matrix

        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            pauli_ind = one of (1,2,3) for (X, Y, Z)
    """
    assert len(rho.shape) == 2 # rho should be a matrix
    
    IndL = 2**i
    dim = rho.shape[0]

    out = rho.reshape((-1, 2,  dim//2, 2, IndL), order='F').copy()
    if pauli_ind == SIGMA_X_IND: # sigma_X
        out = np.flip(out, (1,3))
    elif pauli_ind == SIGMA_Y_IND: # sigma_Y
        out = np.flip(out, (1,3))
        out[:,1,:,0,:] = -out[:,1,:,0,:]
        out[:,0,:,1,:] = -out[:,0,:,1,:]
    elif pauli_ind == SIGMA_Z_IND: # sigma_Z
        out[:,1,:,0,:] = -out[:,1,:,0,:]
        out[:,0,:,1,:] = -out[:,0,:,1,:]
    
    return out.reshape(rho.shape, order='F')


def operate_single_qubit_mixed(rho, i: int, operation):
    """ Apply a single qubit operation on the input density matrix.
        Efficient implementation using reshape and transpose.
        
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            operation = 2x2 single-qubit operator to be applied
    """
    assert len(rho.shape)==2

    IndL = 2**i
    dim = rho.shape[0]

    # left multiply
    out = rho.reshape((-1, 2, IndL), order='F').transpose([1,0,2])
    out = np.dot(operation, out.reshape((2,-1), order='F'))
    out = out.reshape((2, -1, IndL), order='F').transpose([1,0,2])

    # right multiply
    out = out.reshape((-1, 2, IndL*dim), order='F').transpose([0,2,1])
    out = np.dot(out.reshape((-1,2), order='F'), operation.conj().T)
    out = out.reshape((-1, IndL*dim, 2), order='F').transpose([0,2,1])

    return out.reshape(rho.shape, order='F')


def rotate_single_qubit(state, i: int, angle: float, pauli_ind: int):
    """ Apply a single qubit rotation exp(-1j * angle * pauli) to wavefunction

        Input:
            state = input wavefunction (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            angle = rotation angle
            pauli_ind = one of (1,2,3) for (X, Y, Z)
    """

    IndL = 2**i

    out = state.reshape((-1, 2, IndL), order='F').copy()

    if pauli_ind == SIGMA_X_IND: # sigma_X
        out = out.astype(complex, copy=False)
        rot = np.array([[np.cos(angle), -1j*np.sin(angle)],
                        [-1j*np.sin(angle), np.cos(angle)]]);
        out = np.einsum('ij, hjk->hik', rot, out)
    elif pauli_ind == SIGMA_Y_IND: # sigma_Y
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]]);
        out = np.einsum('ij, hjk->hik', rot, out)
    elif pauli_ind == SIGMA_Z_IND: # sigma_Z
        out = out.astype(complex, copy=False)
        out[:,0,:] = np.exp(-1j*angle)*out[:,0,:]
        out[:,1,:] = np.exp(1j*angle)*out[:,1,:]

    return out.reshape(state.shape, order='F')


def rotate_all_qubit(state, N: int, angle: float, pauli_ind: int, denop=False):
    """ Apply qubit rotation exp(-1j * angle * pauli) to every qubit
        Input:
            state = input wavefunction or density matrix (as numpy.ndarray)
            N = number of qubits in the system
            angle = rotation angle
            pauli_ind = one of (1,2,3) for (X, Y, Z)
            denop = False (default) if wavefunction, True if density matrix
    """
    if denop:
        state = rotate_all_qubit(state, N, angle, pauli_ind, denop=False).conj().T
        return rotate_all_qubit(state, N, angle, pauli_ind, denop=False).conj().T

    for i in range(N):
        state = rotate_single_qubit(state, i, angle, pauli_ind)
    return state
