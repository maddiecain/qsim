import unittest
from qsim.codes import three_qubit_code
import numpy as np
from qsim import tools
from qsim.codes.quantum_state import State


class TestThreeQubitCode(unittest.TestCase):
    def test_single_qubit(self):
        psi0 = State(tools.tensor_product([three_qubit_code.logical_basis[0], three_qubit_code.logical_basis[0]]))
        psi1 = psi0.copy()
        # Density matrix test
        psi2 = State(tools.outer_product(psi1, psi1))
        psi0 = three_qubit_code.multiply(psi0, [1], ['Y'])
        # Test non-pauli operation
        psi1 = three_qubit_code.multiply(psi1, [1], three_qubit_code.Y)
        psi2 = three_qubit_code.multiply(psi2, [1], three_qubit_code.Y)
        res = 1j*tools.tensor_product([three_qubit_code.logical_basis[0], three_qubit_code.logical_basis[1]])
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(psi0, res))
        self.assertTrue(np.allclose(psi1, res))
        self.assertTrue(np.allclose(psi2, tools.outer_product(res, res)))

        self.assertTrue(np.allclose(three_qubit_code.multiply(psi0, [1], ['Z']),
                                    -1j*tools.tensor_product([three_qubit_code.logical_basis[0], three_qubit_code.logical_basis[1]])))
        psi0 = three_qubit_code.multiply(psi0, [0], ['X'])
        # Should get out -1j|1L>|1L>
        self.assertTrue(np.allclose(psi0, 1j*tools.tensor_product([three_qubit_code.logical_basis[1],
                                                                   three_qubit_code.logical_basis[1]])))
        # Rotate all qubits
        for i in range(2):
            psi0 = three_qubit_code.rotation(psi0, [i], np.pi / 2, three_qubit_code.X)
        self.assertTrue(np.allclose(psi0, -1j*tools.tensor_product([three_qubit_code.logical_basis[0],
                                                                    three_qubit_code.logical_basis[0]])))

    def test_multi_qubit(self):
        N = 5
        psi0 = State(tools.tensor_product([three_qubit_code.logical_basis[0]]*N))
        psi1 = psi0.copy()
        op = tools.tensor_product([three_qubit_code.X, three_qubit_code.Y, three_qubit_code.Z])
        psi0 = three_qubit_code.multiply(psi0, [1, 3, 4], op)
        psi1 = three_qubit_code.multiply(psi1, [1, 3, 4], ['X', 'Y', 'Z'])

        self.assertTrue(np.allclose(psi0, psi1))

if __name__ == '__main__':
    unittest.main()
