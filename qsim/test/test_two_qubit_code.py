import unittest
from qsim.state import two_qubit_code
import numpy as np
from qsim import tools


class TestTwoQubitCode(unittest.TestCase):
    def test_single_qubit(self):
        psi0 = tools.tensor_product([two_qubit_code.logical_basis[0], two_qubit_code.logical_basis[0]])
        psi1 = psi0[:]
        # Density matrix test
        psi2 = tools.outer_product(psi1[:], psi1[:])
        psi0 = two_qubit_code.multiply(psi0, [1], ['Y'], is_ket=True, pauli=True)
        # Test non-pauli operation
        psi1 = two_qubit_code.multiply(psi1, [1], two_qubit_code.Y, is_ket=True, pauli=False)
        psi2 = two_qubit_code.multiply(psi2, [1], two_qubit_code.Y, is_ket=False, pauli=False)
        res = 1j*tools.tensor_product([two_qubit_code.logical_basis[0], two_qubit_code.logical_basis[1]])
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(psi0, res))
        self.assertTrue(np.allclose(psi1, res))
        self.assertTrue(np.allclose(psi2, tools.outer_product(res, res)))

        self.assertTrue(np.allclose(two_qubit_code.multiply(psi0, [1], ['Z'], is_ket=True, pauli=True),
                                    -1j*tools.tensor_product([two_qubit_code.logical_basis[0], two_qubit_code.logical_basis[1]])))
        psi0 = two_qubit_code.multiply(psi0, [0], ['X'], is_ket=True, pauli=True)
        # Should get out -1j|1L>|1L>
        self.assertTrue(np.allclose(psi0, 1j*tools.tensor_product([two_qubit_code.logical_basis[1],
                                                                   two_qubit_code.logical_basis[1]])))
        # Rotate all qubits
        for i in range(2):
            psi0 = two_qubit_code.rotation(psi0, [i], np.pi / 2, two_qubit_code.X, is_ket=True, pauli=False)
        self.assertTrue(np.allclose(psi0, -1j*tools.tensor_product([two_qubit_code.logical_basis[0],
                                                                    two_qubit_code.logical_basis[0]])))

    def test_multi_qubit(self):
        N = 5
        psi0 = tools.tensor_product([two_qubit_code.logical_basis[0]]*N)
        psi1 = psi0.copy()
        op = tools.tensor_product([two_qubit_code.X, two_qubit_code.Y, two_qubit_code.Z])
        psi0 = two_qubit_code.multiply(psi0, [1, 3, 4], op, is_ket=True, pauli=False)
        psi1 = two_qubit_code.multiply(psi1, [1, 3, 4], ['X', 'Y', 'Z'], is_ket=True, pauli=True)

        self.assertTrue(np.allclose(psi0, psi1))

if __name__ == '__main__':
    unittest.main()
