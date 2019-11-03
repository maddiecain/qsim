import numpy as np
import unittest

from qsim import noise_models
from qsim import tools
from qsim.state import State

psi0 = np.zeros((4, 1))
psi0[0] = 1
rho0 = tools.outer_product(psi0, psi0)
class TestNoiseModels(unittest.TestCase):
    def test_depolarize_single_qubit(self):
        state0 = State(rho0, 2, is_ket=False)
        p = 0.093
        op0 = noise_models.depolarize_single_qubit(state0.state, 1, p)
        op1 = noise_models.depolarize_single_qubit(state0.state, 0, 2 * p)
        self.assertTrue(op1[1, 1] == 0.124)
        self.assertTrue(op0[2, 2] == 0.062)

if __name__ == '__main__':
    unittest.main()
