"""
Helper functions for qubit operations
"""

import numpy as np
from qsim import tools

__all__ = ['two_local_term', 'single_qubit_operation', 'single_qubit_rotation', 'all_qubit_rotation',
           'all_qubit_operation']


def two_local_term(op1, op2, ind1, ind2, N):
    r"""Utility function for conveniently create a 2-Local term op1 \otimes op2 among N spins"""
    if ind1 > ind2:
        return ham_two_local_term(op2, op1, ind2, ind1, N)

    if op1.shape != op2.shape or op1.ndim != 2:
        raise ValueError('ham_two_local_term: invalid operator input')

    if ind1 < 0 or ind2 > N - 1:
        raise ValueError('ham_two_local_term: invalid input indices')

    if op1.shape[0] == 1 or op1.shape[1] == 1:
        myeye = lambda n: np.ones(np.asarray(op1.shape) ** n)
    else:
        myeye = lambda n: np.eye(np.asarray(op1.shape) ** n)

    return np.kron(myeye(ind1), np.kron(op1, np.kron(myeye(ind2 - ind1 - 1), np.kron(op2, myeye(N - ind2 - 1)))))


def single_qubit_operation(state, i: int, op, is_pauli=False, is_ket=False, d=2):
    """ Apply a single qubit operation on the input state.
        Efficient implementation using reshape and transpose.

        Input:
            i = zero-based index of qubit location to apply operation
            operation = 2x2 single-qubit operator to be applied OR a pauli index {0, 1, 2}
            is_pauli = Boolean indicating if op is a pauli index
    """
    N = int(np.log2(state.shape[0]))

    def single_qubit_pauli(state, i: int, pauli_ind: int, is_ket=False):
        """ Multiply a single pauli operator on the i-th qubit of the input wavefunction

            Input:
                state = input wavefunction or density matrix (as numpy.ndarray)
                i = zero-based index of qubit location to apply pauli
                pauli_ind = one of (1,2,3) for (X, Y, Z)
                is_ket = Boolean dictating whether the input is a density matrix (True) or not (False)
        """
        ind = 2 ** i
        if is_ket:
            # Note index start from the right (sN,...,s3,s2,s1)
            out = state.reshape((-1, 2, ind), order='F').copy()
            if pauli_ind == tools.SIGMA_X_IND:  # Sigma_X
                out = np.flip(out, 1)
            elif pauli_ind == tools.SIGMA_Y_IND:  # Sigma_Y
                out = np.flip(out, 1)
                out[:, 0, :] = -1j * out[:, 0, :]
                out[:, 1, :] = 1j * out[:, 1, :]
            elif pauli_ind == tools.SIGMA_Z_IND:  # Sigma_Z
                out[:, 1, :] = -out[:, 1, :]

            state = out.reshape(state.shape, order='F')
        else:
            out = state.reshape((-1, 2, 2 ** (N - 1), 2, ind), order='F').copy()
            if pauli_ind == tools.SIGMA_X_IND:  # Sigma_X
                out = np.flip(out, (1, 3))
            elif pauli_ind == tools.SIGMA_Y_IND:  # Sigma_Y
                out = np.flip(out, (1, 3))
                out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
                out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]
            elif pauli_ind == tools.SIGMA_Z_IND:  # Sigma_Z
                out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
                out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]

            state = out.reshape(state.shape, order='F')
        return state

    if is_pauli:
        assert d == 2
        return single_qubit_pauli(state, i, op, is_ket=is_ket)
    else:
        ind = d ** i
        if is_ket:
            # Left multiply
            out = state.reshape((-1, d, ind), order='F').transpose([1, 0, 2])
            out = np.dot(op, out.reshape((d, -1), order='F'))
            out = out.reshape((d, -1, ind), order='F').transpose([1, 0, 2])
        else:
            # Left multiply
            out = state.reshape((2 ** (N - i - 1), d, -1), order='F').transpose([1, 0, 2])
            out = np.dot(op, out.reshape((d, -1), order='F'))
            out = out.reshape((d, 2 ** (N - i - 1), -1), order='F').transpose([1, 0, 2])
            # Right multiply
            out = out.reshape(-1, d, ind, order='F').transpose([0, 2, 1])
            out = np.dot(out.reshape((-1, d), order='F'), op.conj().T)
            out = out.reshape((-1, ind, d), order='F').transpose([0, 2, 1])

        state = out.reshape(state.shape, order='F')
    return state


def single_qubit_rotation(state, i: int, angle: float, op, is_ket=False):
    """ Apply a single qubit rotation exp(-1j * angle * op) to wavefunction
        Input:
            state = input wavefunction (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            angle = rotation angle
            op = unitary pauli operator or basis pauli index
    """
    rot = np.array([[np.cos(angle), 0], [0, np.cos(angle)]]) - op * 1j * np.sin(angle)
    state = single_qubit_operation(state, i, rot, is_pauli=False, is_ket=is_ket)
    return state


def all_qubit_rotation(state, angle: float, op, is_ket=False):
    """ Apply rotation exp(-1j * angle * pauli) to every qubit
        Input:
            angle = rotation angle
            op = operation on a single qubit
    """
    N = int(np.log2(state.shape[0]))
    for i in range(N):
        state = single_qubit_rotation(state, i, angle, op, is_ket=is_ket)
    return state


def all_qubit_operation(state, op, is_pauli=False, is_ket=False):
    """ Apply qubit operation to every qubit
        Input:
            op = one of (1,2,3) for (X, Y, Z)
    """
    N = int(np.log2(state.shape[0]))
    for i in range(N):
        state = single_qubit_operation(state, i, op, is_pauli=is_pauli, is_ket=is_ket)
    return state
