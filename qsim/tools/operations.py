import numpy as np
from qsim import tools
import math
from scipy.linalg import expm

__all__ = ['single_qubit_pauli', 'single_qubit_operation', 'single_qubit_rotation', 'all_qubit_rotation',
           'all_qubit_operation', 'left_multiply', 'right_multiply', 'expectation']


def left_multiply(state, i: int, op, is_ket=False, d=2):
    """
    Left multiply a single pauli operator on the :math:`i`th qubit of the input wavefunction.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param op: Operator to act with.
    :type op: np.ndarray
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    N = int(math.log(state.shape[0], d))
    N_op = int(op.shape[0])
    ind = N_op ** i
    n = int(math.log(N_op, d))
    if is_ket:
        # Left multiply
        out = state.reshape((-1, N_op, ind), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((N_op, -1), order='F'))
        out = out.reshape((N_op, -1, ind), order='F').transpose([1, 0, 2])
    else:
        # Left multiply
        out = state.reshape((d ** (N - n * i - n), N_op, -1), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((N_op, -1), order='F'))
        out = out.reshape((N_op, d ** (N - n * i - n), -1), order='F').transpose([1, 0, 2])

    out = out.reshape(state.shape, order='F')
    return out


def right_multiply(state, i: int, op, is_ket=False, d=2):
    """
    Right multiply a single pauli operator on the :math:`i`th qubit of the input wavefunction.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param op: Operator to act with.
    :type op: np.ndarray
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    N = int(math.log(state.shape[0], d))
    N_op = int(op.shape[0])
    n = int(math.log(N_op, d))
    # Right multiply
    if is_ket:
        out = state.reshape((d ** (2 * N - n * (i + 1)), N_op, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, N_op), order='F'), op.conj().T)
        out = out.reshape((d ** (2 * N - n * (i + 1)), -1, N_op), order='F').transpose([0, 2, 1])
        state = out.reshape(state.shape, order='F')
        return state
    else:
        # Right multiply
        out = state.reshape((d ** (2 * N - n * (i + 1)), N_op, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, N_op), order='F'), op.conj().T)
        out = out.reshape((d ** (2 * N - n * (i + 1)), -1, N_op), order='F').transpose([0, 2, 1])
        state = out.reshape(state.shape, order='F')
        return state


def single_qubit_pauli(state, i: int, pauli_ind: str, is_ket=False, d=2):
    """
    Apply a single pauli operator on the :math:`i`th qubit of the input wavefunction.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param pauli_ind: one of (X, Y, Z)
    :type pauli_ind: str
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    N = int(math.log(state.shape[0], d))
    ind = d ** i
    if is_ket:
        # Note index start from the right (sN,...,s3,s2,s1)
        out = state.copy().reshape((-1, 2, ind), order='F')
        if pauli_ind == 'X':  # Sigma_X
            out = np.flip(out, 1)
        elif pauli_ind == 'Y':  # Sigma_Y
            out = np.flip(out, 1)
            out[:, 0, :] = -1j * out[:, 0, :]
            out[:, 1, :] = 1j * out[:, 1, :]
        elif pauli_ind == 'Z':  # Sigma_Z
            out[:, 1, :] = -out[:, 1, :]

        out = out.reshape(state.shape, order='F')
    else:
        out = state.copy().reshape((-1, 2, 2 ** (N - 1), 2, ind), order='F')
        if pauli_ind == 'X':  # Sigma_X
            out = np.flip(out, (1, 3))
        elif pauli_ind == 'Y':  # Sigma_Y
            out = np.flip(out, (1, 3))
            out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
            out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]
        elif pauli_ind == 'Z':  # Sigma_Z
            out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
            out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]

        out = out.reshape(state.shape, order='F')
    return out


def single_qubit_operation(state, i: int, op, is_ket=False, d=2):
    """
    Apply a single qubit operator on the ith qubit of the input wavefunction.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param op: Operator to act with.
    :type op: np.ndarray
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    N_op = int(op.shape[0])
    N = int(math.log(state.shape[0], d))
    ind = N_op ** i
    n = int(math.log(N_op, d))
    if is_ket:
        # Left multiply
        out = state.reshape((-1, N_op, ind), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((N_op, -1), order='F'))
        out = out.reshape((N_op, -1, ind), order='F').transpose([1, 0, 2])
    else:
        # Left multiply
        out = state.reshape((d ** (N - n * i - n), N_op, -1), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((N_op, -1), order='F'))
        out = out.reshape((N_op, d ** (N - n * i - n), -1), order='F').transpose([1, 0, 2])
        # Right multiply
        out = out.reshape((d ** (2 * N - n * (i + 1)), N_op, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, N_op), order='F'), op.conj().T)
        out = out.reshape((d ** (2 * N - n * (i + 1)), -1, N_op), order='F').transpose([0, 2, 1])

    state = out.reshape(state.shape, order='F')
    return state


def multi_qubit_operation(state, i: int, op, apply_to, is_ket=False, d=2):
    """
    Apply a single qubit operator on the ith qubit of the input wavefunction.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param op: Operator to act with.
    :type op: np.ndarray
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    N_op = int(op.shape[0])
    N = int(math.log(state.shape[0], d))
    ind = N_op ** i
    n = int(math.log(N_op, d))
    if is_ket:
        # Left multiply
        out = state.reshape((-1, N_op, ind), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((N_op, -1), order='F'))
        out = out.reshape((N_op, -1, ind), order='F').transpose([1, 0, 2])
    else:
        # Left multiply
        out = state.reshape((d ** (N - n * i - n), N_op, -1), order='F').transpose([1, 0, 2])
        out = np.dot(op, out.reshape((N_op, -1), order='F'))
        out = out.reshape((N_op, d ** (N - n * i - n), -1), order='F').transpose([1, 0, 2])
        # Right multiply
        out = out.reshape((d ** (2 * N - n * (i + 1)), N_op, -1), order='F').transpose([0, 2, 1])
        out = np.dot(out.reshape((-1, N_op), order='F'), op.conj().T)
        out = out.reshape((d ** (2 * N - n * (i + 1)), -1, N_op), order='F').transpose([0, 2, 1])

    state = out.reshape(state.shape, order='F')
    return state


def single_qubit_rotation(state, i: int, angle: float, op, is_ket=False, d=2, is_involutary=True):
    """
    Apply a single qubit rotation :math:`e^{-i \\alpha A}` to the input ``state``.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param angle: The angle :math:`\\alpha`` to rotate by.
    :type angle: float
    :param op: Operator to act with.
    :type op: np.ndarray
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    if is_involutary:
        rot = np.cos(angle) * np.identity(op.shape[0]) - op * 1j * np.sin(angle)
        return single_qubit_operation(state, i, rot, is_ket=is_ket, d=d)
    else:
        return single_qubit_operation(state, i, expm(-1j * angle * op), is_ket=is_ket, d=d)


def all_qubit_rotation(state, angle: float, op, is_ket=False, d=2, is_involutary=True):
    """ Apply rotation :math:`e^{-i \\alpha A}` to every qubit in the the input ``state``.

    :param state: input wavefunction or density matrix
    :type state: np.ndarray
    :param i: zero-based index of qudit location to apply the Pauli operator
    :type i: int
    :param angle: The angle :math:`\\alpha`` to rotate by.
    :type angle: float
    :param op: Operator to act with.
    :type op: np.ndarray
    :param is_ket: Boolean dictating whether the input is a density matrix or a ket
    :type is_ket: bool
    :param d: Integer representing the dimension of the qudit
    :type d: int
    """
    N_op = op.shape[0]
    N = int(math.log(state.shape[0], d))
    for i in range(int(N / math.log(N_op, d))):
        state = single_qubit_rotation(state, i, angle, op, is_ket=is_ket, d=d, is_involutary=is_involutary)
    return state


def all_qubit_operation(state, op, is_ket=False, d=2):
    """ Apply qubit operation to every qubit.
    """
    N_op = op.shape[0]
    N = int(math.log(state.shape[0], d))
    for i in range(int(N / math.log(N_op, d))):
        state = single_qubit_operation(state, i, op, is_ket=is_ket)
    return state


def expectation(state, op, is_ket=False):
    """
    :param is_ket: True if the input is a ket, False if the input is a density matrix
    :param op: Operator to take the expectation of in :py:attr:`state`. Current support only for `op.shape==self.state.shape`."""
    if is_ket:
        return state.conj().T @ op @ state
    else:
        return tools.trace(state @ op)

def measurement_outcomes(self, operator):
    """
    Determines the measurement outcomes on an ``operator`` in the given ``state``.

    :param operator: The operator to simulate a measurement on.
    :type operator: np.array
    :return: The probabilities, eigenvalues, and eigenvectors of the possible measurement outcomes
    """
    assert not self.is_ket
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    # Change the basis
    if not self.is_ket:
        state = eigenvectors.conj().T @ self.state.copy() @ eigenvectors
    else:
        state = eigenvectors @ self.state.copy()
    if self.is_ket:
        return np.absolute(state.T) ** 2, eigenvalues, eigenvectors
    else:
        n = eigenvectors.shape[0]
        outcomes = np.matmul(np.reshape(eigenvectors.conj(), (n, n, 1)),
                             np.reshape(eigenvectors, (n, 1, n))) @ state
        probs = np.trace(outcomes, axis1=-2, axis2=-1)
        return probs, eigenvalues, outcomes


def measurement(state, operator, is_ket=False):
    """
    Simulates measuring ``operator`` in the given ``state``.

    :param operator: The operator to simulate a measurement on
    :type operator: np.array
    :return: The eigenvalue and eigenvector of the measurement outcome
    """
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    if not is_ket:
        state = eigenvectors.conj().T @ state.copy() @ eigenvectors
    else:
        state = eigenvectors @ state.copy()
    if is_ket:
        probs = np.absolute(state.T) ** 2
        i = np.random.choice(operator.shape[0], p=probs[0])
        return eigenvalues[i], eigenvectors[i]
    else:
        n = eigenvectors.shape[0]
        outcomes = np.matmul(np.reshape(eigenvectors.conj(), (n, n, 1)),
                             np.reshape(eigenvectors, (n, 1, n))) @ state
        probs = np.trace(outcomes, axis1=-2, axis2=-1)
        i = np.random.choice(operator.shape[0], p=np.absolute(probs))
        return eigenvalues[i], outcomes[i] / probs

