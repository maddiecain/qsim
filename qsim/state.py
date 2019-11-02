import numpy as np
from qsim import tools


class State(object):
    """Contains information about the system density matrix
    is_ket: bool
    """

    def __init__(self, state, N, is_ket=True):
        # Cast into complex type
        self.state = state.astype(np.complex128, copy=False)
        self.is_ket = is_ket
        self.N = N

    def is_ket_state(self):
        return np.array_equal(self.state @ self.state, self.state) or self.is_ket

    def is_valid_dmatrix(self):
        return (np.allclose(np.imag(np.linalg.eigvals(self.state)), np.zeros(2**self.N)) and
                np.all(np.real(np.linalg.eigvals(self.state)) >= -1e-10) and
                np.absolute(np.trace(self.state)) == 1) or self.is_ket

    def change_basis(self, state, B):
        """B is the new basis. B is assumed to be orthonormal. The original basis is assumed
        to be the standard basis."""
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

        def single_qubit_pauli(self, i: int, pauli_ind: int):
            """ Multiply a single pauli operator on the i-th qubit of the input wavefunction

                Input:
                    state = input wavefunction or density matrix (as numpy.ndarray)
                    i = zero-based index of qubit location to apply pauli
                    pauli_ind = one of (1,2,3) for (X, Y, Z)
                    is_ket = Boolean dictating whether the input is a density matrix (True) or not (False)
            """
            ind = 2 ** i
            if self.is_ket:
                # Note index start from the right (sN,...,s3,s2,s1)
                out = self.state.reshape((ind, 2, -1), order='F').copy()

                if pauli_ind == tools.SIGMA_X_IND:  # Sigma_X
                    out = np.flip(out, 1)
                elif pauli_ind == tools.SIGMA_Y_IND:  # Sigma_Y
                    out = np.flip(out, 1)
                    out[:, 0, :] = -1j * out[:, 0, :]
                    out[:, 1, :] = 1j * out[:, 1, :]
                elif pauli_ind == tools.SIGMA_Z_IND:  # Sigma_Z
                    out[:, 1, :] = -out[:, 1, :]

                self.state = out.reshape(self.state.shape, order='F')
            else:
                out = self.state.reshape((ind, 2, 2 ** (self.N - 1), 2, -1), order='F').copy()
                if pauli_ind == tools.SIGMA_X_IND:  # Sigma_X
                    out = np.flip(out, (1, 3))
                elif pauli_ind == tools.SIGMA_Y_IND:  # Sigma_Y
                    out = np.flip(out, (1, 3))
                    out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
                    out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]
                elif pauli_ind == tools.SIGMA_Z_IND:  # Sigma_Z
                    out[:, 1, :, 0, :] = -out[:, 1, :, 0, :]
                    out[:, 0, :, 1, :] = -out[:, 0, :, 1, :]

                self.state = out.reshape(self.state.shape, order='F')

        if is_pauli:
            single_qubit_pauli(self, i, op)
        else:
            ind = 2 ** i
            # Left multiply
            out = self.state.reshape((ind, 2, -1), order='F').transpose([1, 0, 2])
            out = np.dot(op, out.reshape((2, -1), order='F'))
            out = out.reshape((2, ind, -1), order='F').transpose([1, 0, 2])

            if not self.is_ket:
                # Right multiply
                out = out.reshape((2 ** (self.N + i), 2, -1), order='F').transpose([0, 2, 1])
                out = np.dot(out.reshape((-1, 2), order='F'), op.conj().T)
                out = out.reshape((2 ** (self.N + i), -1, 2), order='F').transpose([0, 2, 1])

            self.state = out.reshape(self.state.shape, order='F')
        return self.state

    def single_qubit_rotation(self, i: int, angle: float, op):
        """ Apply a single qubit rotation exp(-1j * angle * op) to wavefunction
            Input:
                state = input wavefunction (as numpy.ndarray)
                i = zero-based index of qubit location to apply pauli
                angle = rotation angle
                op = unitary pauli operator or basis pauli index
        """
        rot = np.array([[np.cos(angle), 0], [0, np.cos(angle)]]) - op * 1j * np.sin(angle)
        self.single_qubit_operation(i, rot, is_pauli=False)
        return self.state

    def double_qubit_operation(i, j, op, is_pauli=False):
        # op is a 4x4 matrix
        pass

    def all_qubit_rotation(self, angle: float, op, end=None):
        # TODO: Change this to a k-qubit rotation
        """ Apply rotation exp(-1j * angle * pauli) to every qubit
            Input:
                angle = rotation angle
                op = operation on a single qubit
        """
        if end is None:
            end = self.N
        for i in range(end):
            self.single_qubit_rotation(i, angle, op)
        return self.state

    def all_qubit_operation(self, op, is_pauli=False, end=None):
        """ Apply qubit operation to every qubit
            Input:
                op = one of (1,2,3) for (X, Y, Z)
        """
        if end is None:
            end = self.N
        for i in range(end):
            self.single_qubit_operation(i, op, is_pauli)
        return self.state

    def expectation(self, operator):
        """Operator is an Operator-class object"""
        if self.is_ket:
            return self.state.conj().T @ operator @ self.state
        else:
            tools.trace(self.state @ operator)

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

    def all_qubit_rotation(self, angle: float, op, end=None):
        if end is None:
            end = self.N
        for i in range(end):
            self.single_qubit_rotation(i, angle, op)

    def all_qubit_operation(self, op, is_pauli=False, end=None):
        if end is None:
            end = self.N
        for i in range(end):
            self.single_qubit_operation(i, op, is_pauli=is_pauli)
