import networkx as nx
import numpy as np
from qsim import tools
from qsim.tools.operations import *

__all__ = ['State', 'TwoQubitCode', 'JordanFarhiShor', 'ThreeQubitCode', 'ThreeQubitCodeTwoAncillas', 'MarvianCode']


class State(object):
    r""":class:`State` stores a density matrix or ket, and contains methods to perform individual or bulk
    operations on qudits.

    :param state: The quantum state. Should have dimension :math:`2^N`
    :type state: np.array
    :param N: The number of qubits in the state
    :type N: int
    :param is_ket: Defaults to ``False`` if the state is represented as a density matrix
    :type is_ket: Boolean
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
        # If is_ket, should be dimension (2**N, 1)
        self.state = state.astype(np.complex128, copy=False)
        self.is_ket = is_ket
        self.N = N
        self.d = d

    def is_pure_state(self):
        """Returns ``True`` if :py:attr:`state` is a pure state."""
        return np.array_equal(self.state @ self.state, self.state) or self.is_ket

    def is_valid_dmatrix(self, verbose = True):
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

    def opX(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, i, 'X', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opY(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, i, 'Y', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opZ(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, i, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def rotX(self, i: int, angle, overwrite=True):
        result = single_qubit_rotation(self.state, i, angle, State.X, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def rotY(self, i: int, angle, overwrite=True):
        result = single_qubit_rotation(self.state, i, angle, State.Y, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def rotZ(self, i: int, angle, overwrite=True):
        result = single_qubit_rotation(self.state, i, angle, State.Z, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def single_qubit_operation(self, i: int, op, overwrite=True):
        """Apply a single qubit operation on the input state.
        Efficient implementation using reshape and transpose.

        :param i: zero-based index of qubit location to apply operation
        :type i: Boolean
        :param op: :math:`2 \\times 2` single-qubit operator to be applied
        """
        result = single_qubit_operation(self.state, i, op, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def single_qubit_rotation(self, i: int, angle: float, op, overwrite=True):
        """Apply a single qubit rotation :math:`\\exp(-i \\theta * op)` to wavefunction

        :param i: zero-based index of qubit location to apply pauli
        :param angle: rotation angle
        :param op: projection operator or basis pauli index
        """
        result = single_qubit_rotation(self.state, i, angle, op, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def all_qubit_rotation(self, angle: float, op, overwrite=True):
        """Apply rotation :math:`\\exp(-i \\theta * op)` to every qubit

        :param angle: rotation angle :math:`\theta`
        :param op: operation to perform on a single qubit
        """
        result = all_qubit_rotation(self.state, angle, op, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def all_qubit_operation(self, op, overwrite=True):
        """ Apply a qubit operation ``op`` to every qubit

        :param is_pauli: If True, op should be a string denoting the Pauli matrix to apply.
        :param op: :math:`2 \\times 2` single-qubit operator to be applied
        """
        result = all_qubit_operation(self.state, op, is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def expectation(self, operator):
        """
        :param operator: Operator to take the expectation of in :py:attr:`state`
         It is required that `operator.shape==self.state.shape`."""
        assert operator.shape == self.state.shape
        return expectation(self.state, operator, is_ket=self.is_ket)

    def measurement_outcomes(self, operator):
        """
        Determines the measurement outcomes an ``operator`` in the given ``state``.

        :param operator: The operator to simulate a measurement on
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

    def multiply(self, operator, overwrite = True):
        """Applies ``operator`` to :py:attr:`state`."""
        if overwrite:
            self.state = tools.multiply(self.state, operator, is_ket=self.is_ket)
        return tools.multiply(self.state, operator, is_ket=self.is_ket)


class TwoQubitCode(State):
    X = tools.tensor_product((tools.X(), tools.identity()))
    Y = tools.tensor_product((tools.Y(), tools.Z()))
    Z = tools.tensor_product((tools.Z(), tools.Z()))
    n = 2
    basis = np.array([[[1], [0], [0], [1]], [[0], [1], [1], [0]]]) / np.sqrt(2)
    proj = tools.outer_product(basis[0], basis[0]) + tools.outer_product(basis[1], basis[1])

    def __init__(self, state, N, is_ket=True):
        # Simple two qubit code with |0>_L = |00>, |1>_L = |11>
        super().__init__(state, TwoQubitCode.n * N, is_ket)

    def opX(self, i: int, overwrite=True):
        # I_i X_{i+1}
        return super().opX(TwoQubitCode.n * i + 1, overwrite=overwrite)

    def opY(self, i: int, overwrite=True):
        # Y_i Z_{i+1}
        result = single_qubit_pauli(self.state, TwoQubitCode.n * i, 'Y', is_ket=self.is_ket)
        result = single_qubit_pauli(result, TwoQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opZ(self, i: int, overwrite=True):
        # Z_i Z_{i+1}
        result = single_qubit_pauli(self.state, TwoQubitCode.n * i, 'Z', is_ket=self.is_ket)
        result = single_qubit_pauli(result, TwoQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def rotX(self, i: int, angle, overwrite=True):
        # TODO
        pass

    def rotY(self, i: int, angle, overwrite=True):
        # TODO
        pass

    def rotZ(self, i: int, angle, overwrite=True):
        # TODO
        pass


class JordanFarhiShor(State):
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

    def opX(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, JordanFarhiShor.n * i, 'Y', is_ket=self.is_ket)
        result = single_qubit_pauli(result, JordanFarhiShor.n * i + 2, 'Y', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opY(self, i: int, overwrite=True):
        result = -1 * self.state
        result = single_qubit_pauli(result, JordanFarhiShor.n * i + 1, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, JordanFarhiShor.n * i + 2, 'X', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opZ(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, JordanFarhiShor.n * i, 'Z', is_ket=self.is_ket)
        result = single_qubit_pauli(result, JordanFarhiShor.n * i + 1, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result


class ThreeQubitCode(State):
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

    def opX(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, ThreeQubitCode.n * i, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 1, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 2, 'X', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opY(self, i: int, overwrite=True):
        result = single_qubit_operation(self.state, ThreeQubitCode.n * i, -1 * tools.Y(), is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 1, 'Y', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 2, 'Y', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opZ(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, ThreeQubitCode.n * i, 'Z', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 1, 'Z', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 2, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result


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

    def opX(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, ThreeQubitCodeTwoAncillas.n * i, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCodeTwoAncillas.n * i + 1, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCodeTwoAncillas.n * i + 2, 'X', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opY(self, i: int, overwrite=True):
        result = single_qubit_operation(self.state, ThreeQubitCodeTwoAncillas.n * i, -1 * tools.Y(), is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCodeTwoAncillas.n * i + 1, 'Y', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCodeTwoAncillas.n * i + 2, 'Y', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result

    def opZ(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, ThreeQubitCodeTwoAncillas.n * i, 'Z', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCodeTwoAncillas.n * i + 1, 'Z', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCodeTwoAncillas.n * i + 2, 'Z', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result


class FTThreeQubitCode(State):
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

    def opX(self, i: int, overwrite=True):
        result = single_qubit_pauli(self.state, ThreeQubitCode.n * i, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 1, 'X', is_ket=self.is_ket)
        result = single_qubit_pauli(result, ThreeQubitCode.n * i + 2, 'X', is_ket=self.is_ket)
        if overwrite:
            self.state = result
        return result


    def opZ(self, i: int, overwrite=True):
        result = self.single_qubit_operation(i, FTThreeQubitCode.Z, overwrite=overwrite)
        if overwrite:
            self.state = result
        return result
