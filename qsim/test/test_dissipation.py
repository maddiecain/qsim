import numpy as np
import unittest

from qsim.dissipation import quantum_channels
from qsim import tools
from qsim.state import State

psi0 = np.zeros((4, 1))
psi0[0] = 1
rho0 = tools.outer_product(psi0, psi0)


class TestDissipation(unittest.TestCase):
    def test_depolarize_single_qubit(self):
        state0 = State(rho0, 2, is_ket=False)
        state1 = State(rho0, 2, is_ket=False)
        p = 0.093
        op0 = quantum_channels.DepolarizingChannel(p)
        state0.state = op0.single_qubit_channel(state0.state, 1)
        op1 = quantum_channels.DepolarizingChannel(2 * p)
        state1.state = op1.single_qubit_channel(state1.state, 0)
        self.assertTrue(state1.state[2, 2] == 0.124)
        self.assertTrue(state0.state[1, 1] == 0.062)


if __name__ == '__main__':
    unittest.main()
