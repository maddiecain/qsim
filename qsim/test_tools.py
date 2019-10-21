import unittest
import tools
import numpy as np


class TestTools(unittest.TestCase):
    def test_int_to_binary(self):
        self.assertTrue(np.array_equal(int_to_binary(17), np.array([[1, 0, 0, 0, 1]])))

    def test_binary_to_int(self):
        self.assertEqual(17, binary_to_int(np.array([[1, 0, 0, 0, 1]])))

    def test_X(self):
        self.assertTrue(np.array_equal(X(2), np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])))

    def test_Y(self):
        self.assertTrue(np.array_equal(Y(2), np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])))

    def test_Z(self):
        self.assertTrue(np.array_equal(Z(2), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])))

    def test_hadamard(self):
        self.assertTrue(
            np.allclose(hadamard(2), np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2))

    def test_trace(self):
        # Basic test
        self.assertAlmostEqual(1, trace(np.array([[.5, 0], [0, .5]]), np.array([[1, 1], [1, -1]]) / np.sqrt(2)))


if __name__ == '__main__':
    unittest.main()
