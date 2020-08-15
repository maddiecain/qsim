import numpy as np
import unittest

from qsim.graph_algorithms.graph import Graph
from qsim import tools
from qsim.evolution import hamiltonian
from qsim.codes import rydberg
from qsim.test import tools_test
from qsim.codes.quantum_state import State

# Generate sample graph
g = tools_test.sample_graph()


class TestHamiltonian(unittest.TestCase):
    def test_hamiltonian_C(self):
        pass

        # Then compute MaxCut energies
        hc = hamiltonian.HamiltonianMaxCut(g)
        self.assertTrue(hc.hamiltonian[0, 0] == 0)
        self.assertTrue(hc.hamiltonian[-1, -1] == 0)
        self.assertTrue(hc.hamiltonian[2, 2] == 3)

    def test_rydberg_hamiltonian(self):
        # Test normal MIS Hamiltonian
        # Graph has six nodes and nine edges
        hc = hamiltonian.HamiltonianMIS(g)
        self.assertTrue(hc.hamiltonian[0, 0] == -3)
        self.assertTrue(hc.hamiltonian[-1, -1] == 0)
        self.assertTrue(hc.hamiltonian[-2, -2] == 1)
        self.assertTrue(hc.hamiltonian[2, 2] == -1)

        # Test for qubits
        hq = hamiltonian.HamiltonianMIS(g, energies=(1, 100))
        self.assertTrue(hq.hamiltonian[0, 0] == -894)
        self.assertTrue(hq.hamiltonian[-1, -1] == 0)
        self.assertTrue(hq.hamiltonian[2, 2] == -595)

        psi0 = State(np.zeros((2 ** hq.N, 1)))
        psi0[-1, -1] = 1
        psi1 = State(tools.outer_product(psi0, psi0))
        self.assertTrue(hq.cost_function(psi1) == 0)
        self.assertTrue(hq.cost_function(psi0) == 0)
        self.assertTrue(hq.optimum_overlap(psi0) == 0)
        psi2 = State(np.zeros((2 ** hq.N, 1)))
        psi2[27, -1] = 1
        self.assertTrue(hq.optimum_overlap(psi2) == 1)
        self.assertTrue(hq.cost_function(psi2) == 2)
        psi2[30, -1] = 1
        psi2 = psi2/np.sqrt(2)
        self.assertTrue(np.isclose(hq.optimum_overlap(psi2), 1))
        self.assertTrue(np.isclose(hq.cost_function(psi2), 2))

        # Test for rydberg EIT
        hr = hamiltonian.HamiltonianMIS(g, energies=(1, 100), code=rydberg)
        self.assertTrue(hr.hamiltonian[0, 0] == -894)
        self.assertTrue(hr.hamiltonian[-1, -1] == 0)
        self.assertTrue(hr.hamiltonian[6, 6] == -595)

        psi0 = State(np.zeros((rydberg.d ** hr.N, 1)))
        psi0[-1, -1] = 1
        psi1 = State(tools.outer_product(psi0, psi0))
        self.assertTrue(hr.cost_function(psi1) == 0)
        self.assertTrue(hr.cost_function(psi0) == 0)

    def test_hamiltonian_laser(self):
        N = 6

        hl_qubit = hamiltonian.HamiltonianDriver()

        psi0 = State(np.zeros((2 ** N, 1)))
        psi0[0, 0] = 1
        psi1 = State(tools.outer_product(psi0, psi0))

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        psi0 = hl_qubit.evolve(psi0, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(np.vdot(psi0, psi0) == 1)
        self.assertTrue(psi0[-1, 0] == (-1j) ** N)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        psi1 = hl_qubit.evolve(psi1, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(tools.is_valid_state(psi1))
        self.assertAlmostEqual(psi1[-1, -1], 1)

        psi0 = State(np.zeros((2 ** N, 1)))
        psi0[0, 0] = 1
        psi0 = hl_qubit.left_multiply(psi0)
        psi1 = np.zeros((2 ** N, 1), dtype=np.complex128)
        for i in range(N):
            psi1[2 ** i, 0] = 1
        self.assertTrue(np.allclose(psi0, psi1))

        N = 3
        hl = hamiltonian.HamiltonianDriver(transition=(0, 1), code=rydberg)
        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[5, 0] = 1
        psi1 = State(tools.outer_product(psi0, psi0), code=rydberg)
        psi0 = hl.left_multiply(psi0)

        self.assertTrue(psi0[2, 0] == 1)
        self.assertTrue(psi0[14, 0] == 1)
        psi1 = hl.left_multiply(psi1)
        self.assertTrue(psi1[2, 5] == 1)
        self.assertTrue(psi1[14, 5] == 1)

        psi0 = State(np.zeros((rydberg.d ** N, 1)), code=rydberg)
        psi0[5, 0] = 1
        psi0 = hl.evolve(psi0, np.pi / 2)
        self.assertTrue(np.isclose(psi0[11, 0], -1))


if __name__ == '__main__':
    unittest.main()
