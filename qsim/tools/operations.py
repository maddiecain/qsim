"""
Helper functions for qubit operations
"""

import numpy as np
from qsim import tools

__all__ = ['single_qubit_pauli', 'single_qubit_operation', 'single_qubit_rotation', 'all_qubit_rotation',
           'all_qubit_operation', 'left_multiply', 'right_multiply', 'expectation']


def left_multiply(state, i: int, op, is_ket=False):
    N = int(np.log2(state.shape[0]))
    d = int(op.shape[0])
    ind = d ** i
    n = int(np.log2(d))

    if is_ket:
        # Left multiply
        out = state.reshape((-1, d, ind), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((d, -1), order='F'))
        out = out.reshape((d, -1, ind), order='F').transpose([1, 0, 2])
    else:
        # Left multiply
        out = state.reshape((2 ** (N - n * i - n), d, -1), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((d, -1), order='F'))
        out = out.reshape((d, 2 ** (N - n * i - n), -1), order='F').transpose([1, 0, 2])

    state = out.reshape(state.shape, order='F')
    return state


def right_multiply(state, i: int, op, is_ket=False):
    N = int(np.log2(state.shape[0]))
    d = int(op.shape[0])
    n = int(np.log2(d))
    # Right multiply
    if is_ket:
        out = state.reshape((2 ** (2 * N - n * (i + 1)), d, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, d), order='F'), op.conj().T)
        out = out.reshape((2 ** (2 * N - n * (i + 1)), -1, d), order='F').transpose([0, 2, 1])

        state = out.reshape(state.shape, order='F')
        return state
    else:
        # Right multiply
        out = state.reshape((2 ** (2 * N - n * (i + 1)), d, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, d), order='F'), op.conj().T)
        out = out.reshape((2 ** (2 * N - n * (i + 1)), -1, d), order='F').transpose([0, 2, 1])
        state = out.reshape(state.shape, order='F')
        return state


def single_qubit_pauli(state, i: int, pauli_ind: str, is_ket=False):
    """ Multiply a single pauli operator on the i-th qubit of the input wavefunction

        Input:
            state = input wavefunction or density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            pauli_ind = one of (X, Y, Z)
            is_ket = Boolean dictating whether the input is a density matrix (True) or not (False)
    """
    N = int(np.log2(state.shape[0]))
    ind = 2 ** i
    if is_ket:
        # Note index start from the right (sN,...,s3,s2,s1)
        out = state.reshape((-1, 2, ind), order='F').copy()
        if pauli_ind == 'X':  # Sigma_X
            out = np.flip(out, 1)
        elif pauli_ind == 'Y':  # Sigma_Y
            out = np.flip(out, 1)
            out[:, 0, :] = -1j * out[:, 0, :]
            out[:, 1, :] = 1j * out[:, 1, :]
        elif pauli_ind == 'Z':  # Sigma_Z
            out[:, 1, :] = -out[:, 1, :]

        state = out.reshape(state.shape, order='F')
    else:
        out = state.reshape((-1, 2, 2 ** (N - 1), 2, ind), order='F').copy()
        if pauli_ind == 'X':  # Sigma_X
            out = np.flip(out, (1, 3))
        elif pauli_ind == 'Y':  # Sigma_Y
            out = np.flip(out, (1, 3))
            out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
            out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]
        elif pauli_ind == 'Z':  # Sigma_Z
            out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
            out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]

        state = out.reshape(state.shape, order='F')
    return state


def single_qubit_operation(state, i: int, op, is_ket=False):
    """ Apply a single qubit operation on the input state.
        Efficient implementation using reshape and transpose.

        Input:
            i = zero-based index of qubit location to apply operation
            operation = 2x2 single-qubit operator to be applied OR a pauli index {0, 1, 2}
            is_pauli = Boolean indicating if op is a pauli index
    """
    """N = int(np.log2(state.shape[0]))
    ind = d ** i
    n = int(np.log2(d))
    if is_ket:
        # Left multiply
        out = state.reshape((-1, d, ind), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((d, -1), order='F'))
        out = out.reshape((d, -1, ind), order='F').transpose([1, 0, 2])
    else:
        # Left multiply
        out = state.reshape((2 ** (N - n * i - n), d, -1), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((d, -1), order='F'))
        out = out.reshape((d, 2 ** (N - n * i - n), -1), order='F').transpose([1, 0, 2])
        # Right multiply
        out = out.reshape((2 ** (2 * N - n * (i + 1)), d, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, d), order='F'), op.conj().T)
        out = out.reshape((2 ** (2 * N - n * (i + 1)), -1, d), order='F').transpose([0, 2, 1])

    state = out.reshape(state.shape, order='F')
    return state"""
    d = int(op.shape[0])
    N = int(np.log2(state.shape[0]))
    ind = d ** i
    n = int(np.log2(d))
    if is_ket:
        # Left multiply
        out = state.reshape((-1, d, ind), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((d, -1), order='F'))
        out = out.reshape((d, -1, ind), order='F').transpose([1, 0, 2])
    else:
        # Left multiply
        out = state.reshape((2 ** (N - n * i - n), d, -1), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((d, -1), order='F'))
        out = out.reshape((d, 2 ** (N - n * i - n), -1), order='F').transpose([1, 0, 2])
        # Right multiply
        out = out.reshape((2 ** (2 * N - n * (i + 1)), d, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, d), order='F'), op.conj().T)
        out = out.reshape((2 ** (2 * N - n * (i + 1)), -1, d), order='F').transpose([0, 2, 1])

    state = out.reshape(state.shape, order='F')
    return state


def single_qubit_rotation(state, i: int, angle: float, op, is_ket=False, d=2):
    """ Apply a single qubit rotation exp(-1j * angle * op) to wavefunction
        Input:
            state = input wavefunction (as numpy.ndarray)
            i = zero-based index of qubit location to apply rotation
            angle = rotation angle
            op = unitary pauli operator or basis pauli index
    """
    rot = np.cos(angle) * tools.identity(int(np.log2(d))) - op * 1j * np.sin(angle)
    return single_qubit_operation(state, i, rot, is_ket=is_ket, d=d)


def all_qubit_rotation(state, angle: float, op, is_ket=False):
    """ Apply rotation exp(-1j * angle * pauli) to every qubit
        Input:
            angle = rotation angle
            op = operation on a single qubit
    """
    d = op.shape[0]
    N = int(np.log2(state.shape[0]))
    for i in range(int(N / np.log2(d))):
        state = single_qubit_rotation(state, i, angle, op, is_ket=is_ket, d=d)
    return state


def all_qubit_operation(state, op, is_ket=False):
    """ Apply qubit operation to every qubit
        Input:
            op = one of (1,2,3) for (X, Y, Z)
    """
    d = op.shape[0]
    N = int(np.log2(state.shape[0]))
    for i in range(int(N / np.log2(d))):
        state = single_qubit_operation(state, i, op, is_ket=is_ket, d=d)
    return state


def expectation(state, op, is_ket=False):
    """
    :param op: Operator to take the expectation of in :py:attr:`state`
     Current support only for `operator.shape==self.state.shape`."""
    if is_ket:
        return state.conj().T @ op @ state
    else:
        return tools.trace(state @ op)
