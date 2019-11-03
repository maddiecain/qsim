import numpy as np
from qsim import tools
from qsim.tools.operations import *

class State(object):
    """Contains information about the system density matrix
    is_ket: bool
    """

    def __init__(self, state, N, is_ket=False):
        # Cast into complex type
        # If is_ket, should be dimension (2**N, 1)
        self.state = state.astype(np.complex128, copy=False)
        self.is_ket = is_ket
        self.N = N

    def is_pure_state(self):
        return np.array_equal(self.state @ self.state, self.state) or self.is_ket

    def is_valid_dmatrix(self):
        return (np.allclose(np.imag(np.linalg.eigvals(self.state)), np.zeros(2**self.N)) and
                np.all(np.real(np.linalg.eigvals(self.state)) >= -1e-10) and
                np.absolute(np.trace(self.state)) == 1) or (self.is_ket and np.linalg.norm(self.state) == 1)

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
        self.state = single_qubit_operation(self.state, i, op, is_pauli=is_pauli, is_ket=self.is_ket)

    def single_qubit_rotation(self, i: int, angle: float, op):
        """ Apply a single qubit rotation exp(-1j * angle * op) to wavefunction
            Input:
                i = zero-based index of qubit location to apply pauli
                angle = rotation angle
                op = unitary pauli operator or basis pauli index
        """
        self.state = single_qubit_rotation(self.state, i, angle, op, is_ket=self.is_ket)

    def double_qubit_operation(self, i, j, op, is_pauli=False):
        # op is a 4x4 matrix
        pass

    def all_qubit_rotation(self, angle: float, op):
        # TODO: Change this to a k-qubit rotation
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
        print(self.state, operator)
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


class TwoQubitCode(State):
    def __init__(self, state, N, is_ket=True):
        # Simple two qubit code with |0>_L = |00>, |1>_L = |11>
        super().__init__(state, N, is_ket)

    def single_qubit_operation(self, i: int, op, is_pauli=False):
        # i indexes the logical qubit
        # The logical qubit starts at index 2 ** (2*i)

        if is_pauli:
            if op == tools.SIGMA_X_IND:
                # SX_i SX_{i+1}
                super().single_qubit_operation(2 * i, tools.SIGMA_X_IND, is_pauli=is_pauli)
                super().single_qubit_operation(2 * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Y_IND:
                # SY_i SX_{i+1}
                super().single_qubit_operation(2 * i, tools.SIGMA_Y_IND, is_pauli=is_pauli)
                super().single_qubit_operation(2 * i + 1, tools.SIGMA_X_IND, is_pauli=is_pauli)
            elif op == tools.SIGMA_Z_IND:
                # SZ_i SZ_{i+1}
                super().single_qubit_operation(2 * i, tools.SIGMA_Z_IND, is_pauli=is_pauli)
                super().single_qubit_operation(2 * i + 1, tools.SIGMA_Z_IND, is_pauli=is_pauli)

        else:
            # TODO: Implement this based off of the double qubit operation for a general state
            super().double_qubit_operation(2 * i, 2 * i + 1, op)

    def single_qubit_rotation(self, i: int, angle: float, op):
        rot = np.cos(angle) * np.identity(4) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)

    def all_qubit_rotation(self, angle: float, op):
        for i in range(self.N):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False):
        if end is None:
            end = self.N
        for i in range(end):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)
