import unittest
from qsim.codes import rydberg
import numpy as np
from qsim import tools
from qsim.codes.quantum_state import State

mixed_state = np.array([[.75, 0, 0],[0, 0, 0], [0, 0, .25]])
pure_state = np.array([[1, 0, 0]]).T
invalid_state = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])


class TestRydbergEIT(unittest.TestCase):
    def test_single_qubit_pauli(self):
        # Tests single qubit operation given a pauli index
        N = 6
        # Initialize in |000000>
        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[0][0] = 1

        # Apply sigma_y on the second qubit to get 1j|020000>
        psi0 = rydberg.multiply(psi0, [1], ['Y'])
        self.assertTrue(psi0[(rydberg.d - 1) * rydberg.d ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, codes is -1j|020000>
        psi0 = rydberg.multiply(psi0, [1], 'Z')
        self.assertTrue(psi0[(rydberg.d - 1) * rydberg.d ** (N - 2), 0] == -1j)

        # Apply sigma_x on all qubits but 1
        psi0 = rydberg.multiply(psi0, [0, 2, 3, 4, 5], ['X'] * (N - 1))

        # Vector is still normalized
        self.assertTrue(np.vdot(psi0, psi0) == 1)

        # Should be -1j|111111>
        self.assertTrue(psi0[-1, 0] == -1j)

        # Density matrix test
        psi1 = np.array([tools.tensor_product([np.array([1, 0, 1]), np.array([1, 0, 0])]) / 2 ** (1 / 2)]).T
        psi1 = State(tools.outer_product(psi1, psi1), code=rydberg)
        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([-1j, 0, 1j], [1, 0, 0]) / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        psi1 = rydberg.multiply(psi1, [0], 'Y')

        self.assertTrue(np.linalg.norm(psi1 - rho2) <= 1e-10)

    def test_single_qubit_operation(self):
        N = 6
        # Test single qubit operation
        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[0][0] = 1
        # Apply sigma_y on the second qubit to get 1j|020000>
        psi0 = rydberg.multiply(psi0, [1], rydberg.Y)
        self.assertTrue(psi0[(rydberg.d - 1) * rydberg.d ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, codes is -1j|010000>
        psi0 = rydberg.multiply(psi0, [1], rydberg.Z)
        self.assertTrue(psi0[(rydberg.d - 1) * rydberg.d ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubits
        psi0 = rydberg.multiply(psi0, [0, 2, 3, 4, 5], tools.tensor_product([rydberg.X] * (N - 1)))

        # Vector is still normalized
        self.assertTrue(np.vdot(psi0, psi0) == 1)

        # Should be -1j|111111>
        self.assertTrue(psi0[-1, 0] == -1j)

        # Test rydberg operations with density matrices
        psi1 = np.array([tools.tensor_product([np.array([1, 0, 1]), np.array([1, 0, 0])]) / 2 ** (1 / 2)]).T
        psi1 = State(tools.outer_product(psi1, psi1), code=rydberg)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, 0, -1], [1, 0, 0]) * -1j / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        psi1 = rydberg.multiply(psi1, [0], ['Y'])
        self.assertTrue(np.linalg.norm(psi1 - rho2) <= 1e-10)

        psi0 = np.array([tools.tensor_product([np.array([1, 0, 1]), np.array([1, 0, 0])]) / 2 ** (1 / 2)]).T
        psi0 = State(tools.outer_product(psi0, psi0), code=rydberg)
        psi1 = State(psi0.copy(), code=rydberg)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, 0, -1], [1, 0, 0]) * -1j / 2 ** (1 / 2)]).T
        psi2 = tools.outer_product(psi2, psi2)

        # Apply single qubit operation to dmatrix
        psi0 = rydberg.multiply(psi0, [0], rydberg.Y)
        self.assertTrue(np.linalg.norm(psi0 - psi2) <= 1e-10)

        # Test on ket
        psi1 = rydberg.multiply(psi1, [0], rydberg.Y)
        self.assertTrue(np.linalg.norm(psi1 - psi2) <= 1e-10)

    def test_single_qubit_rotation(self):
        # Initialize in |000000>
        N = 6
        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[0, 0] = 1
        psi1 = State(psi0, code=rydberg)
        # Rotate by exp(-1i*pi/4*sigma_y) every qubit to get |++++++>
        for i in range(N):
            psi0 = rydberg.rotation(psi0, [i], np.pi / 4, rydberg.Y, is_involutary=True)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.vdot(psi0, np.ones((rydberg.d ** N, 1)) / 2 ** (N / 2)), 1)

        # Apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z) on every qubit to get exp(-1j*N*pi/4)*|000000>
        for i in range(N):
            psi0 = rydberg.rotation(psi0, [i], np.pi / 4, ['Z'], is_involutary=True)
            psi0 = rydberg.rotation(psi0, [i], np.pi / 4, rydberg.X, is_involutary=True)

        self.assertTrue(np.abs(np.vdot(psi1, psi0) * np.exp(1j * np.pi / 4 * N) - 1) <= 1e-10)

    def test_multi_qubit_operation(self):
        N = 6
        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[0, 0] = 1
        psi1 = psi0.copy()
        op = tools.tensor_product([rydberg.X, rydberg.Y, rydberg.Z])
        psi0 = rydberg.multiply(psi0, [1, 3, 4], op)
        psi1 = rydberg.multiply(psi1, [1], 'X')
        psi1 = rydberg.multiply(psi1, [3], 'Y')
        psi1 = rydberg.multiply(psi1, [4], 'Z')
        self.assertTrue(np.allclose(psi0, psi1))

    def test_multi_qubit_pauli(self):
        N = 6
        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[0, 0] = 1
        psi1 = psi0.copy()
        psi2 = psi0.copy()
        psi3 = psi0.copy()
        psi4 = psi0.copy()
        psi0 = rydberg.multiply(psi0, [1, 3, 4], ['X', 'Y', 'Z'])
        psi1 = rydberg.multiply(psi1, [1], 'X')
        psi1 = rydberg.multiply(psi1, [3], 'Y')
        psi1 = rydberg.multiply(psi1, [4], 'Z')
        psi2 = rydberg.multiply(psi2, [4, 1, 3], ['Z', 'X', 'Y'])
        psi3 = rydberg.multiply(psi3, [1, 3, 4], tools.tensor_product([rydberg.X, rydberg.Y, rydberg.Z]))
        psi4 = rydberg.multiply(psi4, [4, 1, 3], tools.tensor_product([rydberg.Z, rydberg.X, rydberg.Y]))
        self.assertTrue(np.allclose(psi0, psi1))
        self.assertTrue(np.allclose(psi1, psi2))
        self.assertTrue(np.allclose(psi2, psi3))
        self.assertTrue(np.allclose(psi3, psi4))



if __name__ == '__main__':
    unittest.main()
