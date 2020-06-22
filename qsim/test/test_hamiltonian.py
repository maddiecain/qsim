import numpy as np
import networkx as nx
import unittest

from qsim.graph_algorithms import qaoa
from qsim import tools
from qsim.evolution import hamiltonian
from qsim.state import rydberg_EIT

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

sim = qaoa.SimulateQAOA(g, 1, 2, is_ket=False, C = hamiltonian.HamiltonianC(g, mis=False))

N = 10
# Initialize in |000000>

sim.hamiltonian = [hamiltonian.HamiltonianC(g, mis=False), hamiltonian.HamiltonianB()]

class TestHamiltonian(unittest.TestCase):
    def test_hamiltonian_B(self):
        psi0 = np.zeros((2 ** N, 1))
        psi0[0, 0] = 1
        psi1 = tools.outer_product(psi0, psi0)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        psi0 = sim.hamiltonian[1].evolve(psi0, np.pi / 2, is_ket=True)

        # Should get (-1j)^N |111111>
        self.assertTrue(np.vdot(psi0, psi0) == 1)
        self.assertTrue(psi0[-1,0] == (-1j) ** N)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        psi1 = sim.hamiltonian[1].evolve(psi1, np.pi / 2, is_ket=False)

        # Should get (-1j)^N |111111>
        self.assertTrue(tools.is_valid_state(psi1))
        self.assertAlmostEqual(psi1[-1, -1], 1)

        psi0 = np.zeros((2 ** N, 1), dtype=np.complex128)
        psi0[0, 0] = 1
        psi0 = sim.hamiltonian[1].left_multiply(psi0, is_ket=True)
        psi1 = np.zeros((2 ** N, 1), dtype=np.complex128)
        for i in range(N):
            psi1[2 ** i, 0] = 1
        self.assertTrue(np.allclose(psi0, psi1))


    def test_hamiltonian_C(self):
        # Graph has six nodes and nine edges
        # First compute MIS energies
        hc = hamiltonian.HamiltonianC(g, mis=True)
        self.assertTrue(hc.hamiltonian[0,0] == -3)
        self.assertTrue(hc.hamiltonian[-1,0] == -15)
        self.assertTrue(hc.hamiltonian[2, 0] == 1)

        # Then compute MaxCut energies
        hc = hamiltonian.HamiltonianC(g, mis=False)
        self.assertTrue(hc.hamiltonian[0, 0] == -9)
        self.assertTrue(hc.hamiltonian[-1, 0] == -9)
        self.assertTrue(hc.hamiltonian[2, 0] == -3)

    def test_rydberg_hamiltonian(self):
        # Test for qubits
        hr = hamiltonian.HamiltonianRydberg(g, blockade_energy=100, detuning=1)
        self.assertTrue(hr.hamiltonian[0, 0] == 906)
        self.assertTrue(hr.hamiltonian[-1, 0] == 0)
        self.assertTrue(hr.hamiltonian[2, 0] == 605)

        psi0 = np.zeros((2 ** hr.N, 1))
        psi0[0, 0] = 1
        psi1 = tools.outer_product(psi0, psi0)

        self.assertTrue(hr.cost_function(psi1) == 0)
        self.assertTrue(hr.cost_function(psi0) == 0)

        # Test for rydberg EIT
        hr = hamiltonian.HamiltonianRydberg(g, blockade_energy=100, detuning=1, code=rydberg_EIT)
        self.assertTrue(hr.hamiltonian[0, 0] == 906)
        self.assertTrue(hr.hamiltonian[-1, 0] == 0)
        self.assertTrue(hr.hamiltonian[6, 0] == 605)

        psi0 = np.zeros((rydberg_EIT.d ** hr.N, 1))
        psi0[0, 0] = 1
        psi1 = tools.outer_product(psi0, psi0)
        self.assertTrue(hr.cost_function(psi1) == 0)
        self.assertTrue(hr.cost_function(psi0) == 0)






if __name__ == '__main__':
    unittest.main()