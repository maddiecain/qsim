import unittest
from qsim.state import State
import numpy as np

mixed_state = State(np.array([[.5, 0, 1], [0, .25, 0], [1, 0, .25]]))
pure_state = State(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
invalid_state = State(np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]]))

class TestState(unittest.TestCase):
    def test_is_pure_state(self):
        self.assertTrue(pure_state.is_pure_state())
        self.assertTrue(not mixed_state.is_pure_state())

    def test_is_valid_dmatrix(self):
        self.assertTrue(pure_state.is_valid_dmatrix())
        self.assertTrue(not invalid_state.is_valid_dmatrix())

    def test_is_orthonormal(self):
        self.assertTrue(pure_state.is_orthonormal(pure_state.basis))

    def test_change_basis(self):
        self.assertTrue(True)

    def test_expectation(self):
        self.assertTrue(True)

    def test_measurement(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()