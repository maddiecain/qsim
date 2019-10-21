"""
Single qubit operations
"""

import numpy as np

SIGMA_X_IND, SIGMA_Y_IND, SIGMA_Z_IND = (1,2,3)

def multiply_single_spin(state, i: int, pauli_ind: int):
    """ Multiply a single pauli operator on the i-th spin of the input wavefunction

        Input:
            state = input wavefunction (as numpy.ndarray)
            i = zero-based index of spin location to apply pauli
            pauli_ind = one of (1,2,3) for (X, Y, Z)
    """

    IndL = 2**i

    out = state.reshape((IndL, 2, -1), order='F').copy()

    if pauli_ind == SIGMA_X_IND: # sigma_X
        out = np.flip(out, 1)
    elif pauli_ind == SIGMA_Y_IND: # sigma_Y
        out = np.flip(out, 1).astype(complex, copy=False)
        out[:,0,:] = -1j*out[:,0,:]
        out[:,1,:] = 1j*out[:,1,:]
    elif pauli_ind == SIGMA_Z_IND: # sigma_Z
        out[:,1,:] = -out[:,1,:]

    return out.reshape(state.shape, order='F')


def rotate_single_spin(state, i: int, angle: float, pauli_ind: int):
    """ Apply a single spin rotation exp(-1j * angle * pauli) to wavefunction

        Input:
            state = input wavefunction (as numpy.ndarray)
            i = zero-based index of spin location to apply pauli
            angle = rotation angle
            pauli_ind = one of (1,2,3) for (X, Y, Z)
    """

    IndL = 2**i

    out = state.reshape((IndL, 2, -1), order='F').copy()

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


def rotate_all_spin(state, N: int, angle: float, pauli_ind: int, denop=False):
    """ Apply spin rotation exp(-1j * angle * pauli) to every spin
        Input:
            state = input wavefunction or density matrix (as numpy.ndarray)
            N = number of spins in the system
            angle = rotation angle
            pauli_ind = one of (1,2,3) for (X, Y, Z)
            denop = False (default) if wavefunction, True if density matrix
    """
    if denop:
        state = rotate_all_spin(state, N, angle, pauli_ind, denop=False).conj().T
        return rotate_all_spin(state, N, angle, pauli_ind, denop=False).conj().T

    for i in range(N):
        state = rotate_single_spin(state, i, angle, pauli_ind)
    return state
