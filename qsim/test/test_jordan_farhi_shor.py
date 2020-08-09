import unittest
from qsim.codes import jordan_farhi_shor
import numpy as np
from qsim import tools
from qsim.codes.quantum_state import State


class TestJordanFarhiShor(unittest.TestCase):
    def test_single_qubit(self):
        psi0 = State(tools.tensor_product([jordan_farhi_shor.logical_basis[0], jordan_farhi_shor.logical_basis[0]]))
        psi1 = psi0.copy()
        # Density matrix test
        psi2 = State(tools.outer_product(psi1[:], psi1[:]))
        psi0 = jordan_farhi_shor.multiply(psi0, [1], ['Y'])
        # Test non-pauli operation
        psi1 = jordan_farhi_shor.multiply(psi1, [1], jordan_farhi_shor.Y)
        psi2 = jordan_farhi_shor.multiply(psi2, [1], jordan_farhi_shor.Y)
        res = 1j * tools.tensor_product([jordan_farhi_shor.logical_basis[0], jordan_farhi_shor.logical_basis[1]])
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(psi0, res))
        self.assertTrue(np.allclose(psi1, res))
        self.assertTrue(np.allclose(psi2, tools.outer_product(res, res)))

        self.assertTrue(np.allclose(jordan_farhi_shor.multiply(psi0, [1], ['Z']),
                                    -1j * tools.tensor_product(
                                        [jordan_farhi_shor.logical_basis[0], jordan_farhi_shor.logical_basis[1]])))
        psi0 = jordan_farhi_shor.multiply(psi0, [0], ['X'])
        # Should get out -1j|1L>|1L>
        self.assertTrue(np.allclose(psi0, 1j * tools.tensor_product([jordan_farhi_shor.logical_basis[1],
                                                                     jordan_farhi_shor.logical_basis[1]])))
        # Rotate all qubits
        for i in range(2):
            psi0 = jordan_farhi_shor.rotation(psi0, [i], np.pi / 2, jordan_farhi_shor.X)
        self.assertTrue(np.allclose(psi0, -1j * tools.tensor_product([jordan_farhi_shor.logical_basis[0],
                                                                      jordan_farhi_shor.logical_basis[0]])))

    def test_multi_qubit(self):
        n = 5
        psi0 = State(tools.tensor_product([jordan_farhi_shor.logical_basis[0]] * n))
        psi1 = psi0.copy()
        op = tools.tensor_product([jordan_farhi_shor.X, jordan_farhi_shor.Y, jordan_farhi_shor.Z])
        psi0 = jordan_farhi_shor.multiply(psi0, [1, 3, 4], op)
        psi1 = jordan_farhi_shor.multiply(psi1, [1, 3, 4], ['X', 'Y', 'Z'])

        self.assertTrue(np.allclose(psi0, psi1))


if __name__ == '__main__':
    unittest.main()
