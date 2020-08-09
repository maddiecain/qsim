import unittest
from qsim.graph_algorithms.graph import Graph
from qsim.test import tools_test
import numpy as np
from qsim.codes import jordan_farhi_shor
from qsim.codes.quantum_state import State


class TestState(unittest.TestCase):
    def test_state(self):
        # This method just tests if State initializes property
        state = np.zeros((16, 1))
        state[-1, -1] = 1
        state = State(state, code=jordan_farhi_shor)
        self.assertTrue(state.code == jordan_farhi_shor)
        self.assertTrue(state.number_physical_qudits == 4)
        self.assertTrue(state.number_logical_qudits == 1)


if __name__ == '__main__':
    unittest.main()
