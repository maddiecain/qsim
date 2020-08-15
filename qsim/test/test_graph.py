import unittest
from qsim.graph_algorithms import graph
from qsim.test import tools_test


class TestGraph(unittest.TestCase):
    def test_IS_subspace(self):
        g = tools_test.sample_graph()
        self.assertTrue(g.num_independent_sets == 13)
        self.assertTrue(g.mis_size == 2)


if __name__ == '__main__':
    unittest.main()
