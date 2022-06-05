import unittest
import numpy as np
from qsim.test import tools_test
from qsim.graph_algorithms.graph import independence_polynomial, independent_sets


class TestGraph(unittest.TestCase):
    def test_independence_polynomial(self):
        g = tools_test.sample_graph()
        ip = independence_polynomial(g)
        self.assertTrue(np.sum(ip) == 13)
        self.assertTrue(len(ip) - 1 == 2)

    def test_independent_sets(self):
        g = tools_test.sample_graph()
        sets_true = independent_sets(g, preallocate=True)
        sets_false = independent_sets(g, preallocate=False)
        self.assertTrue(np.allclose(sets_true, sets_false))


if __name__ == '__main__':
    unittest.main()
