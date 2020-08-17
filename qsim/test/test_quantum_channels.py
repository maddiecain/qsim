import numpy as np
import unittest

from qsim.evolution import quantum_channels
from qsim import tools
from qsim.codes.quantum_state import State


class TestDissipation(unittest.TestCase):
    def test_depolarize(self):
        # First test single qubit channel
        psi0 = State(np.zeros((4, 1)))
        psi0[0] = 1
        psi0 = State(tools.outer_product(psi0, psi0))
        psi1 = psi0.copy()
        psi2 = psi0.copy()
        psi3 = psi0.copy()
        psi4 = psi0.copy()
        p = 0.093
        op0 = quantum_channels.DepolarizingChannel(p)
        psi0 = op0.channel(psi0, 1)
        op1 = quantum_channels.DepolarizingChannel(2 * p)
        psi1 = op1.channel(psi1, 0)
        self.assertTrue(psi1[2, 2] == 0.124)
        self.assertTrue(psi0[1, 1] == 0.062)

        # Now test multi qubit channel
        psi2 = op0.channel(psi2, [0, 1])
        psi3 = op0.channel(psi3, 0)
        psi3 = op0.channel(psi3, 1)
        psi4 = op0.channel(psi4)
        self.assertTrue(np.allclose(psi2, psi3))
        self.assertTrue(np.allclose(psi2, psi4))


if __name__ == '__main__':
    unittest.main()
