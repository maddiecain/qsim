import numpy as np
import unittest

from qsim import noise_models
from qsim import tools
from qsim.state import State

class TestNoiseModels(unittest.TestCase):
    def test_depolarize_single_qubit(self):
        state0 = State(np.kron(tools.SX, tools.SY), 2, is_ket=False)
        op0 = state0.state
        p = 0.093
        noise_models.depolarize_single_qubit(state0, 1, p)
        op1 = state0.state
        noise_models.depolarize_single_qubit(state0, 0, 2 * p)
        op2 = state0.state
        # For some reason this is failing???
        self.assertTrue(np.linalg.norm(op1 - op0*(1-4*p/3)) <= 1e-10)
        self.assertTrue(np.linalg.norm(op2 - op0*(1-4*p/3)*(1-8*p/3)) <= 1e-10)

if __name__ == '__main__':
    unittest.main()
