import unittest
from qsim import tools
import numpy as np


class TestTools(unittest.TestCase):
    def test_int_to_binary(self):
        self.assertTrue(np.array_equal(tools.int_to_binary(17), np.array([[1, 0, 0, 0, 1]])))

    def test_binary_to_int(self):
        self.assertEqual(17, tools.binary_to_int(np.array([[1, 0, 0, 0, 1]])))

    def test_X(self):
        self.assertTrue(np.array_equal(tools.X(2), np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])))

    def test_Y(self):
        self.assertTrue(np.array_equal(tools.Y(2), np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])))

    def test_Z(self):
        self.assertTrue(np.array_equal(tools.Z(2), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])))

    def test_hadamard(self):
        self.assertTrue(
            np.allclose(tools.hadamard(2), np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2))

    def test_trace(self):
        # Basic test
        self.assertAlmostEqual(1, tools.trace(np.array([[.5, 0], [0, .5]])))
        self.assertAlmostEqual(0, tools.trace(np.array([[1, 1], [1, -1]]) / np.sqrt(2)))

    def test_is_orthonormal(self):
        self.assertTrue(tools.is_orthonormal(np.array([[1, 0], [0, 1]])))

if __name__ == '__main__':
    unittest.main()
