import numpy as np
from qsim import tools
from qsim.tools.operations import *

__all__ = ['State', 'TwoQubitCode', 'JordanFarhiShor', 'ThreeQubitCode', 'ThreeQubitCodeTwoAncillas']


class State(object):
    r""":class:`State` stores a density matrix or ket, and contains methods to perform individual or bulk
    operations on qubits.

    :param state: The quantum state. Should have dimension :math:`2^N`
    :type state: np.array
    :param N: The number of qubits in the state
    :type N: int
    :param is_ket: Defaults to ``False`` if the state is represented as a density matrix
    :type is_ket: Boolean
    """
    # Define Pauli matrices
    SX = tools.SX
    SY = tools.SY
    SZ = tools.SZ
    n = 1
    basis = np.array([[[1], [0]], [[0], [1]]])

    def __init__(self, state, N, is_ket=False):
        # Cast into complex type
        # If is_ket, should be dimension (2**N, 1)
        self.state = state.astype(np.complex128, copy=False)
        self.is_ket = is_ket
        self.N = N

    def is_pure_state(self):
        """Returns ``True`` if :py:attr:`state` is a pure state."""
        return np.array_equal(self.state @ self.state, self.state) or self.is_ket

    def is_valid_dmatrix(self):
        """Returns ``True`` if :py:attr:`state` is a valid density matrix or a ket."""
        if self.is_ket:
            return np.linalg.norm(self.state) == 1
        else:
            return (np.allclose(np.imag(np.linalg.eigvals(self.state)), np.zeros(2 ** self.N)) and
                    np.all(np.real(np.linalg.eigvals(self.state)) >= -1e-10) and
                    np.isclose(np.absolute(np.trace(self.state)), 1))

    def change_basis(self, state, basis):
        """
        :param state: The state to transform, must match the representation (state or density matrix) of :py:attr:`state`. Should be written in the standard basis.
        :type state: np.array
        :param basis: The new basis, where the new basis vectors are the columns. B is assumed to be orthonormal.
        :type basis: np.array
        :returns: :py:attr:`state` represented in new basis
        """
        # TODO: remove this as a class method
        if self.is_ket:
            return basis.T.conj() @ state
        else:
            return basis.T.conj() @ state @ basis

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        """Apply a single qubit operation on the input state.
        Efficient implementation using reshape and transpose.

        :param i: zero-based index of qubit location to apply operation
        :type i: Boolean
        :param op: :math:`2 \\times 2` single-qubit operator to be applied, OR a Pauli index (see :py:const:`qsim.tools.SIGMA_X_IND`, :py:const:`qsim.tools.SIGMA_Y_IND`, :py:const:`qsim.tools.SIGMA_Z_IND`)
        :param is_pauli: Indicates if ``op`` is a pauli index or a single-qubit operator, defaults to ``False``
        :type is_pauli: Boolean
        """
        self.state = single_qubit_operation(self.state, i, op, is_pauli=is_pauli, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        """Apply a single qubit rotation :math:`\\exp(-i \\theta * op)` to wavefunction

        :param i: zero-based index of qubit location to apply pauli
        :param angle: rotation angle
        :param op: unitary pauli operator or basis pauli index
        """
        self.state = single_qubit_rotation(self.state, i, angle, op, is_ket=self.is_ket)

    def all_qubit_rotation(self, angle: float, op):
        """Apply rotation :math:`\\exp(-i \\theta * op)` to every qubit

        :param angle: rotation angle :math:`\theta`
        :param op: operation to perform on a single qubit
        """
        self.state = all_qubit_rotation(self.state, angle, op, is_ket=self.is_ket)

    def all_qubit_operation(self, op, is_pauli=False):
        """ Apply a qubit operation ``op`` to every qubit

        :param op: :math:`2 \\times 2` single-qubit operator to be applied
        """
        self.state = all_qubit_operation(self.state, op, is_pauli=is_pauli, is_ket=self.is_ket)

    def expectation(self, operator):
        """
        :param operator: Operator to take the expectation of in :py:attr:`state`
         Current support only for `operator.shape==self.state.shape`."""
        if self.is_ket:
            return self.state.conj().T @ operator @ self.state
        else:
            return tools.trace(self.state @ operator)

    def measurement_outcomes(self, operator):
        """
        Determines the measurement outcomes an ``operator`` in the given ``state``.

        :param operator: The operator to simulate a measurement on
        :type operator: np.array
        :return: The probabilities, eigenvalues, and eigenvectors of the possible measurement outcomes
        """
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        state = self.change_basis(self.state.copy(), eigenvectors)
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
        state = self.change_basis(self.state.copy(), eigenvectors)
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

    def multiply(self, operator):
        """Applies ``operator`` to :py:attr:`state`."""
        self.state = tools.multiply(self.state, operator, is_ket=self.is_ket)

    def two_qubit_operation(self, i: int, j: int, op):
        """Performs a two-qubit operation on qubits :math:`i` and :math:`j`.

        :param i: The first qubit to operate on
        :type i: int
        :param j: The second qubit to operate on
        :type j: int
        :param op: The :math:`4\\times 4` operation to apply
        :type op: np.array
        :return: The eigenvalue and eigenvector of the measurement outcome
        """
        self.state = two_qubit_operation(self.state, i, j, is_pauli=False, is_ket=self.is_ket)

    @staticmethod
    def equal_superposition(N):
        """Returns a positive equal superposition of all possible states"""
        return tools.equal_superposition(N)


class TwoQubitCode(State):
    SX = tools.tensor_product((tools.SX, np.identity(2)))
    SY = tools.tensor_product((tools.SY, tools.SZ))
    SZ = tools.tensor_product((tools.SZ, tools.SZ))
    n = 2
    basis = np.array([[[1], [0], [0], [1]], [[0], [1], [1], [0]]]) / np.sqrt(2)

    def __init__(self, state, N, is_ket=True):
        # Simple two qubit code with |0>_L = |00>, |1>_L = |11>
        super().__init__(state, TwoQubitCode.n * N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        # The logical qubit starts at index 2 ** (2*i)
        if is_pauli:
            if op == tools.SIGMA_X_IND:
                # I_i SX_{i+1}
                super().single_qubit_operation(TwoQubitCode.n * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                # SY_i SZ_{i+1}
                super().single_qubit_operation(TwoQubitCode.n * i, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(TwoQubitCode.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                # SZ_i SZ_{i+1}
                super().single_qubit_operation(TwoQubitCode.n * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(TwoQubitCode.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
        else:
            self.state = single_qubit_operation(self.state, i, op, is_pauli=False, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(2 ** TwoQubitCode.n) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(int(self.N / TwoQubitCode.n)):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        for i in range(int(self.N / TwoQubitCode.n)):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)

    @staticmethod
    def equal_superposition(N):
        return tools.equal_superposition(N, basis=TwoQubitCode.basis)


class JordanFarhiShor(State):
    SX = tools.tensor_product((tools.SY, np.identity(2), tools.SY, np.identity(2)))
    SY = tools.tensor_product((-1 * np.identity(2), tools.SX, tools.SX, np.identity(2)))
    SZ = tools.tensor_product((tools.SZ, tools.SZ, np.identity(2), np.identity(2)))
    n = 4
    basis = np.array([[[1], [0], [0], [1j], [0], [0], [0], [0], [0], [0], [0], [0], [1j], [0], [0], [1]],
                      [[0], [0], [0], [0], [0], [-1], [1j], [0], [0], [1j], [0], [-1], [0], [0], [0], [0]]]) / 2

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, self.n * N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        if is_pauli:
            if op == tools.SIGMA_X_IND:
                super().single_qubit_operation(JordanFarhiShor.n * i, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(JordanFarhiShor.n * i + 2, tools.SIGMA_Y_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                super().single_qubit_operation(JordanFarhiShor.n * i, -1 * np.identity(2), is_pauli=not is_pauli)
                super().single_qubit_operation(JordanFarhiShor.n * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(JordanFarhiShor.n * i + 2, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                super().single_qubit_operation(JordanFarhiShor.n * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(JordanFarhiShor.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
        else:
            self.state = single_qubit_operation(self.state, i, op, is_pauli=False, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(2 ** self.n) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(int(self.N / JordanFarhiShor.n)):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        for i in range(int(self.N / JordanFarhiShor.n)):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)

    @staticmethod
    def equal_superposition(N):
        return tools.equal_superposition(N, basis=JordanFarhiShor.basis)


class ThreeQubitCode(State):
    SX = tools.tensor_product((tools.SX, tools.SX, tools.SX))
    SY = -1 * tools.tensor_product((tools.SY, tools.SY, tools.SY))
    SZ = tools.tensor_product((tools.SZ, tools.SZ, tools.SZ))
    n = 3
    basis = np.array([[[1], [0], [0], [0], [0], [0], [0], [0], [0]],
                      [[0], [0], [0], [0], [0], [0], [0], [0], [1]]])

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, ThreeQubitCode.n * N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        if is_pauli:
            if op == tools.SIGMA_X_IND:
                super().single_qubit_operation(ThreeQubitCode.n * i, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCode.n * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCode.n * i + 2, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                super().single_qubit_operation(ThreeQubitCode.n * i, -1 * tools.SY, is_pauli=not is_pauli)
                super().single_qubit_operation(ThreeQubitCode.n * i + 1, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCode.n * i + 2, tools.SIGMA_Y_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                super().single_qubit_operation(ThreeQubitCode.n * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCode.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCode.n * i + 2, tools.SIGMA_Z_IND, is_pauli=is_pauli)
        else:
            self.state = single_qubit_operation(self.state, i, op, is_pauli=False, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(2 ** ThreeQubitCode.n) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(int(self.N / ThreeQubitCode.n)):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        for i in range(int(self.N / ThreeQubitCode.n)):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)

    @staticmethod
    def equal_superposition(N):
        return tools.equal_superposition(N, basis=ThreeQubitCode.basis)


class ThreeQubitCodeTwoAncillas(State):
    SX = tools.tensor_product((tools.SX, tools.SX, tools.SX, np.identity(2), np.identity(2)))
    SY = -1 * tools.tensor_product((tools.SY, tools.SY, tools.SY, np.identity(2), np.identity(2)))
    SZ = tools.tensor_product((tools.SZ, tools.SZ, tools.SZ, np.identity(2), np.identity(2)))
    n = 5
    basis = np.array([[[0]]*(2**5),
                      [[0]]*(2**5)])
    basis[0,28] = 1
    basis[1,1] = 1

    def __init__(self, state, N, is_ket=True):
        super().__init__(state, ThreeQubitCodeTwoAncillas.n * N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        if is_pauli:
            if op == tools.SIGMA_X_IND:
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i + 2, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i, -1 * tools.SY, is_pauli=not is_pauli)
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i + 1, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i + 2, tools.SIGMA_Y_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(ThreeQubitCodeTwoAncillas.n * i + 2, tools.SIGMA_Z_IND, is_pauli=is_pauli)
        else:
            self.state = single_qubit_operation(self.state, i, op, is_pauli=False, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(2 ** ThreeQubitCodeTwoAncillas.n) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(int(self.N / ThreeQubitCodeTwoAncillas.n)):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        for i in range(int(self.N / ThreeQubitCodeTwoAncillas.n)):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)

    @staticmethod
    def equal_superposition(N):
        return tools.equal_superposition(N, basis=ThreeQubitCodeTwoAncillas.basis)



