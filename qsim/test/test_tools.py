import unittest
from qsim import tools
import numpy as np


class TestTools(unittest.TestCase):
    def test_int_to_nary(self):
        self.assertTrue(np.array_equal(tools.int_to_nary(17), np.array([1, 0, 0, 0, 1])))

    def test_nary_to_int(self):
        self.assertEqual(17, tools.nary_to_int(np.array([1, 0, 0, 0, 1])))
        self.assertEqual(7, tools.nary_to_int(np.array([2, 1]), base=3))

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

    def test_fidelity(self):
        a = np.ones((2,1))/2**.5
        a[0] = -1 * a[0]
        b = a * 1j
        a = tools.outer_product(a, a)
        b = tools.outer_product(b, b)
        assert tools.fidelity(a, b) == 1

    def test_make_valid_state(self):
        rho = np.array([[1/4, 0], [0, 3/4]], dtype=np.complex128)+1e-4
        assert not tools.is_valid_state(rho)
        rho = tools.make_valid_state(rho)
        assert tools.is_valid_state(rho)

if __name__ == '__main__':
    unittest.main()
