import numpy as np
from qsim import tools
from qsim.tools.operations import *

__all__ = ['State', 'TwoQubitCode', 'JordanFarhiShor']


class State(object):
    """Contains information about the system density matrix
    is_ket: bool
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
        return np.array_equal(self.state @ self.state, self.state) or self.is_ket

    def is_valid_dmatrix(self):
        if self.is_ket:
            return np.linalg.norm(self.state) == 1
        else:
            return (np.allclose(np.imag(np.linalg.eigvals(self.state)), np.zeros(2 ** self.N)) and
                    np.all(np.real(np.linalg.eigvals(self.state)) >= -1e-10) and
                    np.isclose(np.absolute(np.trace(self.state)), 1))

    def change_basis(self, state, B):
        """B is the new basis, where the new basis vectors are the columns. B is assumed to be orthonormal.
        The original basis is assumed to be the standard basis."""
        if self.is_ket:
            return B.T.conj() @ state
        else:
            return B.T.conj() @ state @ B

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        """ Apply a single qubit operation on the input state.
            Efficient implementation using reshape and transpose.

            Input:
                i = zero-based index of qubit location to apply operation
                operation = 2x2 single-qubit operator to be applied OR a pauli index {0, 1, 2}
                is_pauli = Boolean indicating if op is a pauli index
        """
        self.state = single_qubit_operation(self.state, i, op, is_pauli=is_pauli, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        """ Apply a single qubit rotation exp(-1j * angle * op) to wavefunction
            Input:
                i = zero-based index of qubit location to apply pauli
                angle = rotation angle
                op = unitary pauli operator or basis pauli index
        """
        self.state = single_qubit_rotation(self.state, i, angle, op, is_ket=self.is_ket)

    def all_qubit_rotation(self, angle: float, op):
        """ Apply rotation exp(-1j * angle * pauli) to every qubit
            Input:
                angle = rotation angle
                op = operation on a single qubit
        """
        self.state = all_qubit_rotation(self.state, angle, op, is_ket=self.is_ket)

    def all_qubit_operation(self, op, is_pauli=False):
        """ Apply qubit operation to every qubit
            Input:
                op = one of (1,2,3) for (X, Y, Z)
        """
        self.state = all_qubit_operation(self.state, op, is_pauli=is_pauli, is_ket=self.is_ket)

    def expectation(self, operator):
        """Operator is a numpy array. Current support only for operator.shape==self.state.shape."""
        if self.is_ket:
            return self.state.conj().T @ operator @ self.state
        else:
            return tools.trace(self.state @ operator)

    def measurement_outcomes(self, operator):
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
        self.state = tools.multiply(self.state, operator, is_ket=self.is_ket)

    @staticmethod
    def equal_superposition(N):
        return tools.equal_superposition(N)


class TwoQubitCode(State):
    SX = tools.tensor_product((tools.SX, np.identity(2)))
    SY = tools.tensor_product((tools.SY, tools.SZ))
    SZ = tools.tensor_product((tools.SZ, tools.SZ))
    n = 2
    basis = np.array([[[1], [0], [0], [1]], [[0], [1], [1], [0]]]) / np.sqrt(2)

    def __init__(self, state, N, is_ket=True):
        # Simple two qubit code with |0>_L = |00>, |1>_L = |11>
        super().__init__(state, self.n * N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        # The logical qubit starts at index 2 ** (2*i)
        if is_pauli:
            if op == tools.SIGMA_X_IND:
                # I_i SX_{i+1}
                super().single_qubit_operation(self.n * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                # SY_i SZ_{i+1}
                super().single_qubit_operation(self.n * i, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(self.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                # SZ_i SZ_{i+1}
                super().single_qubit_operation(self.n * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(self.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
        else:
            self.state = single_qubit_operation(self.state, i, op, is_pauli=False, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(2 ** self.n) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(int(self.N / self.n)):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        for i in range(int(self.N / self.n)):
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
        # Simple two qubit code with |0>_L = |00>, |1>_L = |11>
        super().__init__(state, self.n * N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        # The logical qubit is at index 2 ** (4*i)
        if is_pauli:
            if op == tools.SIGMA_X_IND:
                super().single_qubit_operation(self.n * i, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(self.n * i + 2, tools.SIGMA_Y_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                super().single_qubit_operation(self.n * i, -1 * np.identity(2), is_pauli=not is_pauli)
                super().single_qubit_operation(self.n * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(self.n * i + 2, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                super().single_qubit_operation(self.n * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(self.n * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)
        else:
            self.state = single_qubit_operation(self.state, i, op, is_pauli=False, is_ket=self.is_ket, d=2 ** self.n)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(2 ** self.n) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(int(self.N / self.n)):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        for i in range(int(self.N / self.n)):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)

    @staticmethod
    def equal_superposition(N):
        return tools.equal_superposition(N, basis=JordanFarhiShor.basis)
