import numpy as np
import unittest

from qsim.evolution import quantum_channels
from qsim import tools



class TestDissipation(unittest.TestCase):
    def test_depolarize_single_qubit(self):
        psi0 = np.zeros((4, 1))
        psi0[0] = 1
        psi0 = tools.outer_product(psi0, psi0)
        psi1 = psi0[:]
        p = 0.093
        op0 = quantum_channels.DepolarizingChannel(p)
        psi0 = op0.single_qubit_channel(psi0, 1)
        op1 = quantum_channels.DepolarizingChannel(2 * p)
        psi1 = op1.single_qubit_channel(psi1, 0)
        self.assertTrue(psi1[2, 2] == 0.124)
        self.assertTrue(psi0[1, 1] == 0.062)


if __name__ == '__main__':
    unittest.main()
