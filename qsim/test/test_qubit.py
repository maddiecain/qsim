import unittest
from qsim.codes import qubit
import numpy as np
from qsim import tools
from qsim.codes.quantum_state import State

mixed_state = np.array([[.75, 0], [0, .25]])
pure_state = np.array([[1, 0]]).T
invalid_state = [[-1, 0], [0, 0]]


class TestQubit(unittest.TestCase):
    def test_single_qubit_multiply_pauli(self):
        # Tests single qubit operation given a pauli index
        N = 6
        # Initialize in |000000>
        psi0 = State(np.zeros((2 ** N, 1)))
        psi0[0,0] = 1
        # Apply sigma_y on the second qubit to get 1j|010000>
        # Check Typing
        psi0 = qubit.multiply(psi0, 1, 'Y')
        self.assertTrue(psi0[2 ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, codes is -1j|010000>
        psi0 = qubit.multiply(psi0, [1], 'Z')
        self.assertTrue(psi0[2 ** (N - 2), 0] == -1j)

        # Apply sigma_x on all qubits but 1
        psi0 = qubit.multiply(psi0, [0, 2, 3, 4, 5], ['X'] * (N - 1))

        # Vector is still normalized
        self.assertTrue(np.vdot(psi0, psi0) == 1)

        # Should be -1j|111111>
        self.assertTrue(psi0[-1, 0] == -1j)

        # Density matrix test
        psi1 = np.array([tools.tensor_product([np.array([1, 1]), np.array([1, 0])]) / 2 ** (1 / 2)]).T
        psi1 = State(tools.outer_product(psi1, psi1))

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1j, -1j], [1, 0]) / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        psi1 = qubit.multiply(psi1, [0], 'Y')
        self.assertTrue(np.linalg.norm(psi1 - rho2) <= 1e-10)

    def test_single_qubit_multiply(self):
        N = 6
        # Test single qubit operation
        psi0 = State(np.zeros((2 ** N, 1)))
        psi0[0][0] = 1
        # Apply sigma_y on the second qubit to get 1j|010000>
        psi0 = qubit.multiply(psi0, [1], qubit.Y)
        self.assertTrue(psi0[2 ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, codes is -1j|010000>
        psi0 = qubit.multiply(psi0, [1], qubit.Z)
        self.assertTrue(psi0[2 ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubits
        psi0 = qubit.multiply(psi0, [0, 2, 3, 4, 5], tools.tensor_product([qubit.X]*(N-1)))

        # Vector is still normalized
        self.assertTrue(np.vdot(psi0, psi0) == 1)

        # Should be -1j|111111>
        self.assertTrue(psi0[-1, 0] == -1j)

        # Test qubit operations with density matrices
        psi1 = np.array([tools.tensor_product([np.array([1, 1]), np.array([1, 0])]) / 2 ** (1 / 2)]).T
        psi1 = State(tools.outer_product(psi1, psi1))

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)]).T
        rho2 = State(tools.outer_product(psi2, psi2))

        psi1 = qubit.multiply(psi1, [0], ['Y'])
        self.assertTrue(np.linalg.norm(psi1 - rho2) <= 1e-10)

        psi0 = np.array([tools.tensor_product([np.array([1, 1]), np.array([1, 0])]) / 2 ** (1 / 2)]).T
        psi0 = State(tools.outer_product(psi0, psi0))
        psi1 = psi0.copy()

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)]).T
        psi2 = tools.outer_product(psi2, psi2)

        # Apply single qubit operation to dmatrix
        psi0 = qubit.multiply(psi0, [0], qubit.Y)
        self.assertTrue(np.linalg.norm(psi0 - psi2) <= 1e-10)

        # Test on ket
        psi1 = qubit.multiply(psi1, [0], qubit.Y)
        self.assertTrue(np.linalg.norm(psi1 - psi2) <= 1e-10)

    def test_rotation(self):
        # Initialize in |000000>
        N = 6
        psi0 = np.zeros((2 ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi1 = State(psi0)
        psi2 = State(psi0)
        psi0 = State(psi0)
        # Rotate by exp(-1i*pi/4*sigma_y) every qubit to get |++++++>
        for i in range(N):
            psi0 = qubit.rotation(psi0, [i], np.pi / 4, qubit.Y, is_involutary=True)
            psi2 = qubit.rotation(psi2, [i], np.pi / 4, qubit.Y)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.vdot(psi0, np.ones((2 ** N, 1)) / 2 ** (N / 2)), 1)
        self.assertAlmostEqual(np.vdot(psi2, np.ones((2 ** N, 1)) / 2 ** (N / 2)), 1)

        # Apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z) on every qubit to get exp(-1j*N*pi/4)*|000000>
        for i in range(N):
            psi0 = qubit.rotation(psi0, [i], np.pi / 4, 'Z', is_involutary=True)
            psi0 = qubit.rotation(psi0, [i], np.pi / 4, qubit.X, is_involutary=True)

        self.assertTrue(np.abs(np.vdot(psi1, psi0) * np.exp(1j * np.pi / 4 * N) - 1) <= 1e-10)

    def test_multi_qubit_multiply(self):
        N = 6
        psi0 = np.zeros((2 ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi1 = State(psi0)
        psi0 = State(psi0)
        op = tools.tensor_product([qubit.X, qubit.Y, qubit.Z])
        psi0 = qubit.multiply(psi0, [1, 3, 4], op)
        psi1 = qubit.multiply(psi1, [1], 'X')
        psi1 = qubit.multiply(psi1, [3], 'Y')
        psi1 = qubit.multiply(psi1, [4], 'Z')
        self.assertTrue(np.allclose(psi0, psi1))

    def test_multi_qubit_pauli(self):
        N = 6
        psi0 = np.zeros((2 ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi1 = State(psi0)
        psi2 = State(psi0)
        psi3 = State(psi0)
        psi4 = State(psi0)
        psi0 = State(psi0)
        psi0 = qubit.multiply(psi0, [1, 3, 4], ['X', 'Y', 'Z'])
        psi2 = qubit.multiply(psi2, [4, 1, 3], ['Z', 'X', 'Y'])
        psi3 = qubit.multiply(psi3, [1, 3, 4], tools.tensor_product([qubit.X, qubit.Y, qubit.Z]))
        psi4 = qubit.multiply(psi4, [4, 1, 3], tools.tensor_product([qubit.Z, qubit.X, qubit.Y]))
        psi1 = qubit.multiply(psi1, [1], 'X')
        psi1 = qubit.multiply(psi1, [3], 'Y')
        psi1 = qubit.multiply(psi1, [4], 'Z')
        self.assertTrue(np.allclose(psi0, psi1))
        self.assertTrue(np.allclose(psi1, psi2))
        self.assertTrue(np.allclose(psi2, psi3))
        self.assertTrue(np.allclose(psi3, psi4))




if __name__ == '__main__':
    unittest.main()
