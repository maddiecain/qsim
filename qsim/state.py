import numpy as np
from qsim import tools
from qsim.tools.operations import *

__all__ = ['State', 'TwoQubitCode', 'JordanFarhiShor', 'ThreeQubitCode', 'ThreeQubitCodeTwoAncillas']


class State(object):
    r""":class:`State` stores a density matrix or ket, and contains methods to perform individual or bulk
    operations on qudits.

    :param state: The quantum state. Should have dimension :math:`(1, d^N)` if ``is_ket`` is True, and dimension :math:`(d^N, d^N)` if ``is_ket`` is False.
    :type state: np.array
    :param N: The number of qubits in the state
    :type N: int
    :param is_ket: Defaults to ``False`` if the state is represented as a density matrix
    :type is_ket: Boolean
    :param d: The dimension of the qudit
    :type d: int
    """
    # Define Pauli matrices
    X = tools.X()
    Y = tools.Y()
    Z = tools.Z()
    n = 1
    basis = np.array([[[1], [0]], [[0], [1]]])
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])

    def __init__(self, state, N, is_ket=False, d=2):
        # Cast into complex type
        # If is_ket, should be dimension (d**N, 1)
        self.state = state.astype(np.complex128, copy=False)
        self.is_ket = is_ket
        self.N = N
        self.d = d

    def is_pure_state(self):
        """Returns ``True`` if :py:attr:`state` is a pure state."""
        return np.array_equal(self.state @ self.state, self.state) or self.is_ket

    def is_valid_dmatrix(self, verbose=True):
        """Returns ``True`` if :py:attr:`state` is a valid density matrix or a ket."""
        if self.is_ket:
            return np.linalg.norm(self.state) == 1
        else:
            print('Eigenvalues real?', (np.allclose(np.imag(np.linalg.eigvals(self.state)), np.zeros(2 ** self.N))))
            print('Eigenvalues positive?', np.all(np.real(np.linalg.eigvals(self.state)) >= -1e-10))
            print('Trace 1?', np.isclose(np.absolute(np.trace(self.state)), 1))
            if verbose:
                print('Eigenvalues:', np.linalg.eigvals(self.state))
                print('Trace:', np.trace(self.state))
            return (np.allclose(np.imag(np.linalg.eigvals(self.state)), np.zeros(2 ** self.N), atol=1e-06) and
                    np.all(np.real(np.linalg.eigvals(self.state)) >= -1 * 1e-05) and
                    np.isclose(np.absolute(np.trace(self.state)), 1))

    def single_qubit_pauli(self, i: int, pauli_ind: str, state=None, overwrite=True):
        """Applies :math:`\sigma^x_L`, :math:`\sigma^y_L`, or :math:`\sigma^z_L` to :py:attr:`state`.

        :param i: The logical qudit index to operate on
        :type i: int
        :param pauli_ind: One of (X, Y, Z), indicating which logical Pauli to operate with.
        :type pauli_ind: str
        :param state: If ``None``, the operation is performed on :py:attr:`state`. Otherwise, the user should input the state to operate on.
        :type state: np.array
        :param overwrite: If ``True``, :py:attr:`state` will be overwritten with the result of the operation.
        :type overwrite: Boolean
        :return: The result of the single qubit pauli operation on ``state``.
        """
        if state is None:
            state = self.state
        state = single_qubit_pauli(state, i, pauli_ind, is_ket=self.is_ket, d=self.d)
        if overwrite:
            self.state = state
        return state

    def single_qubit_pauli_rotation(self, i: int, pauli_ind: str, angle, state=None, overwrite=True):
        """Applies :math:`e^{-i \\theta \\sigma^x_L}`, :math:`e^{-i\\theta\\sigma^y_L}`, or :math:`e^{-i\\theta\\sigma^z_L}` to :py:attr:`state`.

        :param i: The logical qudit index to operate on
        :type i: int
        :param pauli_ind: One of (X, Y, Z), indicating which logical Pauli to operate with.
        :type pauli_ind: str
        :param state: If ``None``, the operation is performed on :py:attr:`state`. Otherwise, the user should input the state to operate on.
        :type state: np.array
        :param overwrite: If ``True``, :py:attr:`state` will be overwritten with the result of the operation.
        :type overwrite: Boolean
        :return: The result of the single qubit pauli rotation on ``state``.
        """
        if state is None:
            state = self.state
        state = single_qubit_rotation(state, i, angle, pauli_ind, is_ket=self.is_ket, d=self.d)
        if overwrite:
            self.state = state
        return state

    def single_qubit_operation(self, i: int, op, state=None, overwrite=True):
        """Apply a single qubit operation on the input state.
        Efficient implementation using reshape and transpose.

        :param i: zero-based index of qubit location to apply operation
        :type i: Boolean
        :param op: :math:`d \\times d` single-qubit operator to be applied
        :type op: np.array
        :param state: If ``None``, the operation is performed on :py:attr:`state`. Otherwise, the user should input the state to operate on.
        :type state: np.array
        :param overwrite: If ``True``, :py:attr:`state` will be overwritten with the result of the operation.
        :type overwrite: Boolean
        :return: The result of the single qubit operation on ``state``.
        """
        if state is None:
            state = self.state
        state = single_qubit_operation(state, i, op, is_ket=self.is_ket, d=self.d)
        if overwrite:
            self.state = state
        return state

    def single_qubit_rotation(self, i: int, angle: float, op, state=None, overwrite=True):
        """Apply a single qubit rotation :math:`\\exp(-i \\theta A)` to wavefunction, where :math:`A` is the operator
        specified by ``op``.

        :param i: zero-based index of logical qubit location to apply pauli
        :type i: int
        :param angle: rotation angle
        :type angle: float
        :param op: projection operator or basis pauli index
        :type op: np.array
        :param overwrite: If ``True``, :py:attr:`state` will be overwritten with the result of the operation.
        :type overwrite: Boolean
        :return: The result of the single qubit rotation on ``state``.
        """
        if state is None:
            state = self.state
        state = single_qubit_rotation(state, i, angle, op, is_ket=self.is_ket, d=self.d)
        if overwrite:
            self.state = state
        return state

    def all_qubit_rotation(self, angle: float, op, state=None, overwrite=True):
        """Apply the rotation :math:`e^{-i \\theta A}` to every qubit, where :math:`A` is the operator
        specified by ``op``.

        :param angle: rotation angle
        :type angle: float
        :param op: projection operator or basis pauli index
        :type op: np.array
        :param overwrite: If ``True``, :py:attr:`state` will be overwritten with the result of the operation.
        :type overwrite: Boolean
        :return: The result of the all qubit rotation on ``state``.
        """
        if state is None:
            state = self.state
        state = all_qubit_rotation(state, angle, op, is_ket=self.is_ket, d=self.d)
        if overwrite:
            self.state = state
        return state

    def all_qubit_operation(self, op, state=None, overwrite=True):
        """ Apply a qubit operation ``op`` to every qubit.

        :param angle: rotation angle
        :type angle: float
        :param op: projection operator or basis pauli index
        :type op: np.array
        :param overwrite: If ``True``, :py:attr:`state` will be overwritten with the result of the operation.
        :type overwrite: Boolean
        """
        if state is None:
            state = self.state
        state = all_qubit_operation(state, op, is_ket=self.is_ket, d=self.d)
        if overwrite:
            self.state = state
        return state

    def expectation(self, operator):
        """
        :param operator: Operator to take the expectation of in :py:attr:`state`. Currently is required that `operator.shape` equals `self.state.shape`.
        :type operator: np.array
        :return: The expectation of ``operator`` in ``state``.
         """
        assert operator.shape == self.state.shape
        return expectation(self.state, operator, is_ket=self.is_ket)

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

    def measurement(self, operator):
        """
        Simulates measuring ``operator`` in the given ``state``.

        :param operator: The operator to simulate a measurement on
        :type operator: np.array
        :return: The eigenvalue and eigenvector of the measurement outcome
        """
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        if not self.is_ket:
            state = eigenvectors.conj().T @ self.state.copy() @ eigenvectors
        else:
            state = eigenvectors @ self.state.copy()
        if self.is_ket:
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

    def multiply(self, operator, overwrite=True):
        """Multiplies an ``operator`` :math:`A` to :py:attr:`state`."""
        if overwrite:
            self.state = tools.multiply(self.state, operator, is_ket=self.is_ket)
        return tools.multiply(self.state, operator, is_ket=self.is_ket)


class TwoQubitCode(State):
    """
    :class:`TwoQubitCode` is an error detecting code which detects phase flip (Z-type) errors. Phase flip errors cannot
    be corrected unambiguously. The basis states for this code are :math:`|{0_L}\\rangle=\\frac{1}{\\sqrt{2}}(|{00}\\rangle+|{11})\\rangle`
    and :math:`|{1_L}\\rangle=\\frac{1}{\\sqrt{2}}(|{01}\\rangle+|{10}\\rangle)`, and the logical Pauli operators are
    :math:`X_L = XI`, :math:`Y_L = YZ`, and :math:`Z_L = ZZ`.
    """
    X = tools.tensor_product((tools.X(), tools.identity()))
    Y = tools.tensor_product((tools.Y(), tools.Z()))
    Z = tools.tensor_product((tools.Z(), tools.Z()))
    n = 2
    basis = np.array([[[1], [0], [0], [1]], [[0], [1], [1], [0]]]) / np.sqrt(2)
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])

    def __init__(self, state, N, is_ket=True):
        # Simple two qubit code with |0>_L = |00>, |1>_L = |11>
        super().__init__(state, TwoQubitCode.n * N, is_ket, d=2)

    def single_qubit_pauli(self, i: int, pauli_ind: str, state=None, overwrite=True):
        # I_i X_{i+1}
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = super().single_qubit_pauli(TwoQubitCode.n * i, 'X', state=state, overwrite=overwrite)
        elif pauli_ind == 'Y':
            state = single_qubit_pauli(state, TwoQubitCode.n * i, 'Y', is_ket=self.is_ket)
            state = single_qubit_pauli(state, TwoQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
        elif pauli_ind == 'Z':
            state = single_qubit_pauli(state, TwoQubitCode.n * i, 'Z', is_ket=self.is_ket)
            state = single_qubit_pauli(state, TwoQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = state
        return state

    def single_qubit_pauli_rotation(self, i: int, angle: float, pauli_ind, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_rotation(state, TwoQubitCode.n * i, angle, tools.X(), is_ket=self.is_ket)
        if pauli_ind == 'Y':
            state = single_qubit_rotation(state, TwoQubitCode.n * i, angle, TwoQubitCode.Y, is_ket=self.is_ket)
        if pauli_ind == 'Z':
            state = single_qubit_rotation(state, TwoQubitCode.n * i, angle, TwoQubitCode.Z, is_ket=self.is_ket)
        if overwrite:
            self.state = state
        return state



class JordanFarhiShor(State):
    """
    :class:`JordanFarhiShor` is an error detecting code which detects phase flip (Z-type) and bit flip (X-type) errors.
    These errors cannot be corrected unambiguously.
    """
    X = tools.tensor_product((tools.Y(), tools.identity(), tools.Y(), tools.identity()))
    Y = tools.tensor_product((-1 * tools.identity(), tools.X(), tools.X(), tools.identity()))
    Z = tools.tensor_product((tools.Z(), tools.Z(), tools.identity(), tools.identity()))
    n = 4
    basis = np.array([[[1], [0], [0], [1j], [0], [0], [0], [0], [0], [0], [0], [0], [1j], [0], [0], [1]],
                      [[0], [0], [0], [0], [0], [-1], [1j], [0], [0], [1j], [-1], [0], [0], [0], [0], [0]]]) / 2
    stabilizers = np.array(
        [tools.X(4), tools.Z(4), tools.tensor_product([tools.X(), tools.Y(), tools.Z(), tools.identity()])])
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, self.n * N, is_ket)

    def single_qubit_pauli(self, i: int, pauli_ind: str, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_pauli(state, JordanFarhiShor.n * i, 'Y', is_ket=self.is_ket)
            state = single_qubit_pauli(state, JordanFarhiShor.n * i + 2, 'Y', is_ket=self.is_ket)
        if pauli_ind == 'Y':
            state = -1 * state
            state = single_qubit_pauli(state, JordanFarhiShor.n * i + 1, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, JordanFarhiShor.n * i + 2, 'X', is_ket=self.is_ket)
        if pauli_ind == 'Z':
            state = single_qubit_pauli(state, JordanFarhiShor.n * i, 'Z', is_ket=self.is_ket)
            state = single_qubit_pauli(state, JordanFarhiShor.n * i + 1, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = state
        return state

    def single_qubit_pauli_rotation(self, i: int, pauli_ind: str, angle, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_rotation(state, JordanFarhiShor.n * i, angle, JordanFarhiShor.X, is_ket=self.is_ket,
                                          is_involutary=True)
        if pauli_ind == 'Y':
            state = single_qubit_rotation(state, JordanFarhiShor.n * i, angle, JordanFarhiShor.Y, is_ket=self.is_ket,
                                          is_involutary=True)
        if pauli_ind == 'Z':
            state = single_qubit_rotation(state, JordanFarhiShor.n * i, angle, JordanFarhiShor.Z, is_ket=self.is_ket,
                                          is_involutary=True)
        if overwrite:
            self.state = state
        return state


class ThreeQubitCode(State):
    """
    :class:`ThreeQubitCode` is an error correcting code which can correct bit flip (X-type) errors.
    """
    X = tools.tensor_product((tools.X(), tools.X(), tools.X()))
    Y = -1 * tools.tensor_product((tools.Y(), tools.Y(), tools.Y()))
    Z = tools.tensor_product((tools.Z(), tools.Z(), tools.Z()))
    n = 3
    basis = np.array([[[1], [0], [0], [0], [0], [0], [0], [0]],
                      [[0], [0], [0], [0], [0], [0], [0], [1]]])
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])
    stabilizers = np.array(
        [tools.tensor_product([tools.Z(2), tools.identity()]), tools.tensor_product([tools.identity(), tools.Z(2)])])

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, ThreeQubitCode.n * N, is_ket)

    def single_qubit_pauli(self, i: int, pauli_ind: str, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_pauli(state, ThreeQubitCode.n * i, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'X', is_ket=self.is_ket)
        if pauli_ind == 'Y':
            state = single_qubit_operation(state, ThreeQubitCode.n * i, -1 * tools.Y(), is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'Y', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'Y', is_ket=self.is_ket)
        if pauli_ind == 'Z':
            state = single_qubit_pauli(state, ThreeQubitCode.n * i, 'Z', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = state
        return state

    def single_qubit_pauli_rotation(self, i: int, pauli_ind: str, angle, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_rotation(state, ThreeQubitCode.n * i, angle, ThreeQubitCode.X, is_ket=self.is_ket,
                                          is_involutary=True)
        if pauli_ind == 'Y':
            state = single_qubit_rotation(state, ThreeQubitCode.n * i, angle, ThreeQubitCode.Y, is_ket=self.is_ket,
                                          is_involutary=True)
        if pauli_ind == 'Z':
            state = single_qubit_rotation(state, ThreeQubitCode.n * i, angle, ThreeQubitCode.Z, is_ket=self.is_ket,
                                          is_involutary=True)
        if overwrite:
            self.state = state
        return state


class ThreeQubitCodeTwoAncillas(State):
    X = tools.tensor_product((tools.X(), tools.X(), tools.X(), tools.identity(), tools.identity()))
    Y = -1 * tools.tensor_product((tools.Y(), tools.Y(), tools.Y(), tools.identity(), tools.identity()))
    Z = tools.tensor_product((tools.Z(), tools.Z(), tools.Z(), tools.identity(), tools.identity()))
    n = 5
    basis = np.array([[[0]] * (2 ** n),
                      [[0]] * (2 ** n)])
    basis[0, 28] = 1
    basis[1, 1] = 1
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, ThreeQubitCodeTwoAncillas.n * N, is_ket)

    def single_qubit_pauli(self, i: int, pauli_ind: str, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_pauli(state, ThreeQubitCode.n * i, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'X', is_ket=self.is_ket)
        if pauli_ind == 'Y':
            state = single_qubit_operation(state, ThreeQubitCode.n * i, -1 * tools.Y(), is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'Y', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'Y', is_ket=self.is_ket)
        if pauli_ind == 'Z':
            state = single_qubit_pauli(state, ThreeQubitCode.n * i, 'Z', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = state
        return state


class ETThreeQubitCode(State):
    X = tools.tensor_product((tools.X(), tools.X(), tools.X()))
    Z = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1]])
    n = 3
    basis = np.array([[[1], [0], [0], [0], [0], [0], [0], [0]],
                      [[0], [0], [0], [0], [0], [0], [0], [1]]])
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])
    stabilizers = np.array(
        [tools.tensor_product([tools.Z(2), tools.identity()]), tools.tensor_product([tools.identity(), tools.Z(2)])])

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, ThreeQubitCode.n * N, is_ket)

    def single_qubit_pauli(self, i: int, pauli_ind: str, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_pauli(state, ThreeQubitCode.n * i, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'X', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'X', is_ket=self.is_ket)
        if pauli_ind == 'Y':
            state = single_qubit_operation(state, ThreeQubitCode.n * i, -1 * tools.Y(), is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 1, 'Y', is_ket=self.is_ket)
            state = single_qubit_pauli(state, ThreeQubitCode.n * i + 2, 'Y', is_ket=self.is_ket)
        if pauli_ind == 'Z':
            state = self.single_qubit_operation(i, ETThreeQubitCode.Z, state=state, overwrite=overwrite)
        if overwrite:
            self.state = state
        return state

    def single_qubit_pauli_rotation(self, i: int, pauli_ind: str, angle, state=None, overwrite=True):
        if state is None:
            state = self.state
        if pauli_ind == 'X':
            state = single_qubit_rotation(state, ETThreeQubitCode.n * i, angle, ETThreeQubitCode.X, is_ket=self.is_ket,
                                          is_involutary=True)
        if pauli_ind == 'Y':
            state = single_qubit_rotation(state, ETThreeQubitCode.n * i, angle, ETThreeQubitCode.Y, is_ket=self.is_ket,
                                          is_involutary=True)
        if pauli_ind == 'Z':
            state = single_qubit_rotation(state, ETThreeQubitCode.n * i, angle, ETThreeQubitCode.Z, is_ket=self.is_ket,
                                          is_involutary=True)
        if overwrite:
            self.state = state
        return state
