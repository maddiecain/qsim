import numpy as np
import unittest

from qsim.evolution import quantum_channels
from qsim import tools
from qsim.codes.quantum_state import State
from qsim.graph_algorithms.graph import line_graph


class TestDissipation(unittest.TestCase):
    def test_depolarize(self):
        # First test single qubit channel
        psi0 = State(np.zeros((4, 1)))
        psi0[0] = 1
        psi0 = State(tools.outer_product(psi0, psi0))
        psi1 = psi0.copy()
        psi2 = psi0.copy()
        psi3 = psi0.copy()
        psi4 = psi0.copy()
        p = 0.093
        op0 = quantum_channels.DepolarizingChannel()
        psi0 = op0.channel(psi0, p, apply_to=1)
        op1 = quantum_channels.DepolarizingChannel()
        psi1 = op1.channel(psi1, 2 * p, apply_to=0)
        self.assertTrue(psi1[2, 2] == 0.124)
        self.assertTrue(psi0[1, 1] == 0.062)

        # Now test multi qubit channel
        psi2 = op0.channel(psi2, p, apply_to=[0, 1])
        psi3 = op0.channel(psi3, p, apply_to=0)
        psi3 = op0.channel(psi3, p, apply_to=1)
        psi4 = op0.channel(psi4, p)
        self.assertTrue(np.allclose(psi2, psi3))
        self.assertTrue(np.allclose(psi2, psi4))

        expected = np.zeros((4, 4))
        expected[0, 0] = 0.46827
        expected[1, 1] = 0.03173
        expected[2, 2] = 0.46827
        expected[3, 3] = 0.03173
        psi0 = op0.evolve(psi0, 20)
        self.assertTrue(np.allclose(expected, psi0))

        psi0 = State(np.array([[1, 0], [0, 0]]))
        psi0 = op0.evolve(psi0, 20)
        self.assertTrue(np.allclose(psi0, .5 * np.identity(2)))

    def test_amplitude_damping_channel(self):
        psi0 = State(np.zeros((2, 2)))
        psi0[0, 0] = 1
        spontaneous_emission = quantum_channels.AmplitudeDampingChannel()
        for i in range(200):
            psi0 = spontaneous_emission.channel(psi0, .1)
        self.assertTrue(np.allclose(psi0, np.array([[0, 0], [0, 1]])))
        psi0 = State(np.zeros((2, 2)))
        psi0[0, 0] = 1
        psi0 = spontaneous_emission.evolve(psi0, 20)
        self.assertTrue(np.allclose(psi0, np.array([[0, 0], [0, 1]])))
        psi0 = State(np.zeros((2, 2)), IS_subspace=True)
        psi0[0, 0] = 1
        spontaneous_emission = quantum_channels.AmplitudeDampingChannel(IS_subspace=True, graph=line_graph(1),
                                                                        rates=(2,))
        psi0 = spontaneous_emission.evolve(psi0, 20)
        self.assertTrue(np.allclose(psi0, np.array([[0, 0], [0, 1]])))


if __name__ == '__main__':
    unittest.main()
