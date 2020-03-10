import numpy as np
import networkx as nx
import unittest

from qsim.qaoa import simulate
from qsim.state import State
from qsim import tools, hamiltonian

# Generate sample graph
g = nx.Graph()

g.add_edge(0, 1, weight=1)
g.add_edge(0, 2, weight=1)
g.add_edge(2, 3, weight=1)
g.add_edge(0, 4, weight=1)
g.add_edge(1, 4, weight=1)
g.add_edge(3, 4, weight=1)
g.add_edge(1, 5, weight=1)
g.add_edge(2, 5, weight=1)
g.add_edge(3, 5, weight=1)

sim = simulate.SimulateQAOA(g, 1, 2, is_ket=False, mis=False)

N = 10
# Initialize in |000000>
psi0 = np.zeros((2 ** N, 1))
psi0[0, 0] = 1

sim.hamiltonian = [hamiltonian.HamiltonianC(g, mis=False), hamiltonian.HamiltonianB()]

class TestHamiltonian(unittest.TestCase):
    def test_evolve_B(self):
        state0 = State(psi0, N, is_ket=True)
        state1 = State(tools.outer_product(psi0, psi0), N, is_ket=False)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        sim.hamiltonian[1].evolve(state0, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)
        self.assertTrue(state0.state[-1,0] == (-1j) ** N)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        sim.hamiltonian[1].evolve(state1, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(state1.is_valid_dmatrix)
        self.assertAlmostEqual(state1.state[-1, -1], 1)

    def test_multiply_B(self):
        state0 = State(psi0, N, is_ket=True)

        sim.hamiltonian[1].left_multiply(state0)
        psi1 = np.zeros((2 ** N, 1))
        for i in range(N):
            psi1[2 ** i, 0] = 1

        self.assertTrue(np.allclose(state0.state, psi1))


if __name__ == '__main__':
    unittest.main()