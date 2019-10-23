import numpy as np
import unittest

from qsim import noise_models
from qsim import tools

class TestNoiseModels(unittest.TestCase):
    def test_depolarize_single_qubit(self):
        op1 = np.kron(tools.SX, tools.SY)
        p = 0.093
        op2 = noise_models.depolarize_single_qubit(op1, 1, p)
        op3 = noise_models.depolarize_single_qubit(op2, 0, 2 * p)
        self.assertTrue(np.linalg.norm(op2 - op1*(1-4*p/3)) <= 1e-10)
        self.assertTrue(np.linalg.norm(op3 - op1*(1-4*p/3)*(1-8*p/3)) <= 1e-10)

if __name__ == '__main__':
    unittest.main()
