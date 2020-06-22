import unittest
from qsim.state import rydberg_EIT
import numpy as np
from qsim import tools

mixed_state = np.array([[.75, 0, 0],[0, 0, 0], [0, 0, .25]])
pure_state = np.array([[1, 0, 0]]).T
invalid_state = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])


class TestRydbergEIT(unittest.TestCase):
    def test_single_qubit_pauli(self):
        # Tests single qubit operation given a pauli index
        N = 6
        # Initialize in |000000>
        psi0 = np.zeros((rydberg_EIT.d ** N, 1), dtype=np.complex128)
        psi0[0][0] = 1

        # Apply sigma_y on the second qubit to get 1j|020000>
        psi0 = rydberg_EIT.multiply(psi0, [1], ['Y'], is_ket=True, pauli=True)
        self.assertTrue(psi0[(rydberg_EIT.d-1) * rydberg_EIT.d ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, state is -1j|020000>
        psi0 = rydberg_EIT.multiply(psi0, [1], 'Z', is_ket=True, pauli=True)
        self.assertTrue(psi0[(rydberg_EIT.d-1) * rydberg_EIT.d ** (N - 2), 0] == -1j)

        # Apply sigma_x on all qubits but 1
        psi0 = rydberg_EIT.multiply(psi0, [0, 2, 3, 4, 5], ['X'] * (N - 1), is_ket=True, pauli=True)

        # Vector is still normalized
        self.assertTrue(np.vdot(psi0, psi0) == 1)

        # Should be -1j|111111>
        self.assertTrue(psi0[-1, 0] == -1j)

        # Density matrix test
        psi1 = np.array([tools.tensor_product([np.array([1, 0, 1]), np.array([1, 0, 0])]) / 2 ** (1 / 2)]).T.astype(
            np.complex128)
        psi1 = tools.outer_product(psi1, psi1)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([-1j, 0, 1j], [1, 0, 0]) / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        psi1 = rydberg_EIT.multiply(psi1, [0], 'Y', is_ket=False, pauli=True)
        self.assertTrue(np.linalg.norm(psi1 - rho2) <= 1e-10)

    def test_single_qubit_operation(self):
        N = 6
        # Test single qubit operation
        psi0 = np.zeros((rydberg_EIT.d ** N, 1), dtype=np.complex128)
        psi0[0][0] = 1
        # Apply sigma_y on the second qubit to get 1j|020000>
        psi0 = rydberg_EIT.multiply(psi0, [1], rydberg_EIT.Y, is_ket=True, pauli=False)
        self.assertTrue(psi0[(rydberg_EIT.d-1) * rydberg_EIT.d ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, state is -1j|010000>
        psi0 = rydberg_EIT.multiply(psi0, [1], rydberg_EIT.Z, is_ket=True, pauli=False)
        self.assertTrue(psi0[(rydberg_EIT.d-1) * rydberg_EIT.d ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubits
        psi0 = rydberg_EIT.multiply(psi0, [0, 2, 3, 4, 5], tools.tensor_product([rydberg_EIT.X]*(N-1)), is_ket=True, pauli=False)

        # Vector is still normalized
        self.assertTrue(np.vdot(psi0, psi0) == 1)

        # Should be -1j|111111>
        self.assertTrue(psi0[-1, 0] == -1j)

        # Test qubit operations with density matrices
        psi1 = np.array([tools.tensor_product([np.array([1, 0, 1]), np.array([1, 0, 0])]) / 2 ** (1 / 2)]).T
        psi1 = tools.outer_product(psi1, psi1).astype(np.complex128)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, 0, -1], [1, 0, 0]) * -1j / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        psi1 = rydberg_EIT.multiply(psi1, [0], ['Y'], is_ket=False, pauli=True)
        self.assertTrue(np.linalg.norm(psi1 - rho2) <= 1e-10)

        psi0 = np.array([tools.tensor_product([np.array([1, 0, 1]), np.array([1, 0, 0])]) / 2 ** (1 / 2)]).T
        psi0 = tools.outer_product(psi0, psi0)
        psi1 = psi0[:]

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, 0, -1], [1, 0, 0]) * -1j / 2 ** (1 / 2)]).T
        psi2 = tools.outer_product(psi2, psi2)

        # Apply single qubit operation to dmatrix
        psi0 = rydberg_EIT.multiply(psi0, [0], rydberg_EIT.Y, is_ket=False, pauli=False)
        self.assertTrue(np.linalg.norm(psi0 - psi2) <= 1e-10)

        # Test on ket
        psi1 = rydberg_EIT.multiply(psi1, [0], rydberg_EIT.Y, is_ket=False, pauli=False)
        self.assertTrue(np.linalg.norm(psi1 - psi2) <= 1e-10)

    def test_single_qubit_rotation(self):
        # Initialize in |000000>
        N = 6
        psi0 = np.zeros((rydberg_EIT.d ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi1 = psi0[:]
        # Rotate by exp(-1i*pi/4*sigma_y) every qubit to get |++++++>
        for i in range(N):
            psi0 = rydberg_EIT.rotation(psi0, [i], np.pi / 4, rydberg_EIT.Y, is_ket=True, is_involutary=True)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.vdot(psi0, np.ones((rydberg_EIT.d ** N, 1)) / 2 ** (N / 2)), 1)

        # Apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z) on every qubit to get exp(-1j*N*pi/4)*|000000>
        for i in range(N):
            psi0 = rydberg_EIT.rotation(psi0, [i], np.pi / 4, ['Z'], is_ket=True, is_involutary=True, pauli=True)
            psi0 = rydberg_EIT.rotation(psi0, [i], np.pi / 4, rydberg_EIT.X, is_ket=True, is_involutary=True)

        self.assertTrue(np.abs(np.vdot(psi1, psi0) * np.exp(1j * np.pi / 4 * N) - 1) <= 1e-10)

    def test_multi_qubit_operation(self):
        N = 6
        psi0 = np.zeros((rydberg_EIT.d ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi1 = psi0[:]
        op = tools.tensor_product([rydberg_EIT.X, rydberg_EIT.Y, rydberg_EIT.Z])
        psi0 = rydberg_EIT.multiply(psi0, [1, 3, 4], op, is_ket=True, pauli=False)
        psi1 = rydberg_EIT.multiply(psi1, [1], 'X', is_ket=True, pauli=True)
        psi1 = rydberg_EIT.multiply(psi1, [3], 'Y', is_ket=True, pauli=True)
        psi1 = rydberg_EIT.multiply(psi1, [4], 'Z', is_ket=True, pauli=True)
        self.assertTrue(np.allclose(psi0, psi1))

    def test_multi_qubit_pauli(self):
        N = 6
        psi0 = np.zeros((rydberg_EIT.d ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi1 = psi0[:]
        psi0 = rydberg_EIT.multiply(psi0, [1, 3, 4], ['X', 'Y', 'Z'], is_ket=True, pauli=True)
        psi1 = rydberg_EIT.multiply(psi1, [1], 'X', is_ket=True, pauli=True)
        psi1 = rydberg_EIT.multiply(psi1, [3], 'Y', is_ket=True, pauli=True)
        psi1 = rydberg_EIT.multiply(psi1, [4], 'Z', is_ket=True, pauli=True)
        self.assertTrue(np.allclose(psi0, psi1))


if __name__ == '__main__':
    unittest.main()
