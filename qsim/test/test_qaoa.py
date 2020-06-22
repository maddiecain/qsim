import numpy as np
import networkx as nx
import unittest

from qsim.graph_algorithms import qaoa
from qsim.evolution import quantum_channels, hamiltonian

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

hc = hamiltonian.HamiltonianC(g, mis=False)
hb = hamiltonian.HamiltonianB()
sim = qaoa.SimulateQAOA(g, 1, 2, is_ket=False, C=hc)
sim_ket = qaoa.SimulateQAOA(g, 1, 2, is_ket=True, C=hc)
sim_noisy = qaoa.SimulateQAOA(g, 1, 2, is_ket=False, C=hc)
sim_noisy.noise = []

N = 10
# Initialize in |000000>
psi0 = np.zeros((2 ** N, 1))
psi0[0, 0] = 1

sim.hamiltonian = [hc, hb]
sim_ket.hamiltonian = [hc, hb]
sim_noisy.hamiltonian = [hc, hb]

sim.noise = [quantum_channels.QuantumChannel(), quantum_channels.QuantumChannel()]
sim_ket.noise = [quantum_channels.QuantumChannel(), quantum_channels.QuantumChannel()]
sim_noisy.noise = [quantum_channels.DepolarizingChannel(.001), quantum_channels.DepolarizingChannel(.001)]


class TestSimulateQAOA(unittest.TestCase):
    def test_variational_grad(self):
        # Test that the calculated objective function and gradients are correct
        # p = 1
        F, Fgrad = sim_ket.variational_grad(np.array([1, 0.5]))
        self.assertTrue(np.abs(F - 0.6803639446061733) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([5.88054974, -3.9215107])) <= 1e-5))

        # p = 1 density matrix
        F, Fgrad = sim.variational_grad(np.array([1, 0.5]))
        self.assertTrue(np.abs(F - 0.6803639446061733) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([5.88054974, -3.9215107])) <= 1e-5))

        # p = 1 noisy
        F, Fgrad = sim_noisy.variational_grad(np.array([1, 0.5]))
        self.assertTrue(np.abs(F - 0.6767425876683083) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([5.84924948, -3.90063777])) <= 1e-5))

        # p = 2
        F, Fgrad = sim_ket.variational_grad(np.array([3, 4, 2, 5]))
        self.assertTrue(np.abs(F + 2.0226700483187735) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([10.81352397, 0.71592146, 4.23395355, 4.61234153])) <= 1e-5))

        F, Fgrad = sim.variational_grad(np.array([3, 4, 2, 5]))
        self.assertTrue(np.abs(F + 2.0226700483187735) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([10.81352397, 0.71592146, 4.23395355, 4.61234153])) <= 1e-5))

        # p = 3
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F - 0.22553595528885761) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([0.4198359, -7.88749038, -3.13170928, 11.61837198, 0.89197165,
                                                        -2.60541832])) <= 1e-5))

        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F - 0.22553595528885761) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([0.4198359, -7.88749038, -3.13170928, 11.61837198, 0.89197165,
                                                        -2.60541832])) <= 1e-5))

    def test_run(self):
        sim.p = 1
        sim_ket.p = 1
        sim_noisy.p = 1
        # p = 1 density matrix
        self.assertAlmostEqual(sim_ket.run([1, .5]), 0.6803639446061733)
        self.assertAlmostEqual(sim.run([1, .5]), 0.6803639446061733)

        # See how things look with evolution
        self.assertAlmostEqual(sim_noisy.run([1, .5]), 0.6767425876683083)

        # Higher depth circuit
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F - 0.22553595528885761) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([0.4198359,  -7.88749038, -3.13170928, 11.61837198,  0.89197165, -2.60541832]) < 1e-5)))

    def test_find_optimal_params(self):
        sim_ket.p = 3
        sim_noisy.p = 3
        print('Noiseless:')
        params = sim_ket.find_parameters_minimize()
        self.assertTrue(np.allclose(params.x, np.array([-1.19850235, -0.12806045, 0.68848687, -1.11236156, 0.68044366,
                                                        -0.21707696])))

        print('Noisy:')
        params = sim_noisy.find_parameters_minimize()
        self.assertTrue(np.allclose(params.x, np.array([-1.20106506, -0.12751502, 0.68816092, -1.10802233, 0.67443895, -0.21750499])))

    def test_fix_param_gauge(self):
        """
        Test to show optimize.fix_param_gauge() can fully reduce redundancies of QAOA parameters
        """
        tolerance = 1e-10

        # original parameters (at p=3) in preferred gauge
        param = np.array([0.2, 0.4, 0.7, -0.6, -0.5, -0.3])

        # copy of parameters
        param2 = param.copy()

        # test that gammas are periodic in pi
        param2[:3] += np.random.choice([1, -1], 3) * np.pi
        param_fixed = sim.fix_param_gauge(param2)
        self.assertTrue(np.linalg.norm(param - param_fixed) <= tolerance)

        # test that betas are periodic in pi/2
        param2[3:] += np.random.choice([1, -1], 3) * np.pi / 2
        param_fixed = sim.fix_param_gauge(param2)
        self.assertTrue(np.linalg.norm(param - param_fixed) <= tolerance)

        ## Case: ODD_DEGREE_ONLY
        # test that shifting gamma_i by (n+1/2)*pi and beta_{j>=i} -> -beta_{j>=i}
        # gives equivalent parameters
        param2[2] -= np.pi / 2
        param2[5] = -param2[5]
        param_fixed = sim.fix_param_gauge(param2, degree_parity=1)
        self.assertTrue(np.linalg.norm(param - param_fixed) <= tolerance)

        param2[1] += np.pi * 3 / 2
        param2[4:] = -param2[4:]
        param_fixed = sim.fix_param_gauge(param2, degree_parity=1)
        self.assertTrue(np.linalg.norm(param - param_fixed) <= tolerance)

        # test that two inequivalent parameters should not be the same after fixing gauge
        param2 = param.copy()
        param2[0] -= np.pi * 3 / 2
        param_fixed = sim.fix_param_gauge(param2)
        self.assertTrue(np.linalg.norm(param - param_fixed) > tolerance)

        ## Case: EVEN_DEGREE_ONLY - test parameters are periodic in pi/2
        param2 = param.copy()
        param2 += np.random.choice([1, -1], 6) * np.pi / 2
        param_fixed = sim.fix_param_gauge(param2, degree_parity=0)
        self.assertTrue(np.linalg.norm(param - param_fixed) <= tolerance)


if __name__ == '__main__':
    unittest.main()
