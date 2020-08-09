import numpy as np
import unittest

from qsim.test.tools_test import sample_graph
from qsim.tools.tools import equal_superposition, outer_product
from qsim.graph_algorithms import qaoa
from qsim.graph_algorithms.graph import Graph, ring_graph
from qsim.codes.quantum_state import State
import networkx as nx
from qsim.evolution import quantum_channels, hamiltonian
from qsim.graph_algorithms import qaoa
from qsim.codes import two_qubit_code, jordan_farhi_shor

# Generate sample graph
N = 6

g = Graph(sample_graph())
ring = Graph(ring_graph(N))

hc = hamiltonian.HamiltonianMaxCut(g)
hc_ring = hamiltonian.HamiltonianMaxCut(ring)
hb = hamiltonian.HamiltonianDriver()
hamiltonians = [hc, hb]
ring_hamiltonians = [hc_ring, hb]

sim = qaoa.SimulateQAOA(g, cost_hamiltonian=hc, hamiltonian=hamiltonians)
sim_ring = qaoa.SimulateQAOA(ring, cost_hamiltonian=hc_ring, hamiltonian=ring_hamiltonians)
sim_ket = qaoa.SimulateQAOA(g, cost_hamiltonian=hc, hamiltonian=hamiltonians)
sim_noisy = qaoa.SimulateQAOA(g, cost_hamiltonian=hc, hamiltonian=hamiltonians, noise_model='channel')

# Initialize in |000000>
psi0 = State(equal_superposition(N))
rho0 = State(outer_product(psi0, psi0))

noises = [quantum_channels.DepolarizingChannel(.001)]
sim_noisy.noise = noises * 2


class TestSimulateQAOA(unittest.TestCase):
    def test_variational_grad(self):
        # Test that the calculated objective function and gradients are correct
        # p = 1
        sim_ket.hamiltonian = hamiltonians
        F, Fgrad = sim_ket.variational_grad(np.array([1, 0.5]), initial_state=psi0)
        self.assertTrue(np.abs(F - 5.066062984904652) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-1.68165197, -2.96780002])) <= 1e-5))

        # p = 1 density matrix
        sim.hamiltonian = hamiltonians
        F, Fgrad = sim.variational_grad(np.array([1, 0.5]), initial_state=psi0)
        self.assertTrue(np.abs(F - 5.066062984904652) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-1.68165197, -2.96780002])) <= 1e-5))

        # p = 1 noisy
        sim_noisy.hamiltonian = hamiltonians
        sim_noisy.noise = noises * 2
        F, Fgrad = sim_noisy.variational_grad(np.array([1, 0.5]), initial_state=rho0)
        # Appears to be not applying the noise
        self.assertTrue(np.abs(F - 5.063050014958339) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-1.67270108, -2.95200338])) <= 1e-5))

        # p = 2
        sim_ket.hamiltonian = hamiltonians * 2
        F, Fgrad = sim_ket.variational_grad(np.array([3, 4, 2, 5]), initial_state=psi0)
        self.assertTrue(np.abs(F - 3.1877403125643746) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([3.93046563, -1.05949913, -3.36292985, -2.55656785])) <= 1e-5))

        sim.hamiltonian = hamiltonians * 2
        F, Fgrad = sim.variational_grad(np.array([3, 4, 2, 5]), initial_state=psi0)
        self.assertTrue(np.abs(F - 3.1877403125643746) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([3.93046563, -1.05949913, -3.36292985, -2.55656785])) <= 1e-5))

        # p = 3
        sim.hamiltonian = hamiltonians * 3
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim.variational_grad(params, initial_state=psi0)
        self.assertTrue(np.abs(F - 3.456027668366075) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-1.88407679, 5.27624478, 4.59242754, -0.84219134, 2.82847871,
                                                        1.00351662])) <= 1e-5))

        sim_ket.hamiltonian = hamiltonians * 3
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim_ket.variational_grad(params, initial_state=psi0)
        self.assertTrue(np.abs(F - 3.456027668366075) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-1.88407679, 5.27624478, 4.59242754, -0.84219134, 2.82847871,
                                                        1.00351662])) <= 1e-5))

    def test_run(self):
        # p = 1 density matrix
        sim_ket.hamiltonian = hamiltonians
        self.assertAlmostEqual(sim_ket.run([1, .5], initial_state=psi0), 5.066062984904651)
        self.assertAlmostEqual(sim.run([1, .5], initial_state=psi0), 5.066062984904651)

        # See how things look with evolution
        sim_noisy.hamiltonian = hamiltonians
        sim_noisy.noise = noises * 2
        self.assertAlmostEqual(sim_noisy.run(np.array([1, .5]), initial_state=rho0), 5.063050014958339)

        F, Fgrad = sim.variational_grad([1, .5], initial_state=psi0)
        self.assertAlmostEqual(F, 5.066062984904651)

        # Higher depth circuit
        sim_noisy.hamiltonian = hamiltonians * 3
        sim_noisy.noise = noises * 6
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim_noisy.variational_grad(params, initial_state=rho0)
        self.assertTrue(np.abs(F - 3.474432565774367) <= 1e-5)
        self.assertTrue(np.all(np.abs(
            Fgrad - np.array([-1.86039494, 5.15858468, 4.49666081, -0.82924304, 2.7650584, 0.98504574]) < 1e-5)))

    def test_find_optimal_params(self):
        # Test on a known graph
        for p in [1, 2, 3]:
            sim_ring.hamiltonian = ring_hamiltonians * p
            results = sim_ring.find_parameters_minimize(verbose=False, initial_state=psi0)
            if p == 3:
                self.assertTrue(np.isclose(results['approximation_ratio'], 1))
            else:
                self.assertTrue(np.isclose(results['approximation_ratio'], (p * 2 + 1) / (p * 2 + 2)))
        # Repeat test, but this time using the gradient
        for p in [1, 2, 3]:
            sim_ring.hamiltonian = ring_hamiltonians * p
            results = sim_ring.find_parameters_minimize(verbose=False, analytic_gradient=True, initial_state=psi0)
            if p == 3:
                self.assertTrue(np.isclose(results['approximation_ratio'], 1))
            else:
                self.assertTrue(np.isclose(results['approximation_ratio'], (p * 2 + 1) / (p * 2 + 2)))

        sim_ket.hamiltonian = hamiltonians * 3
        sim_noisy.hamiltonian = hamiltonians * 3
        sim_noisy.noise = noises * 6
        print('Noiseless:')

        results = sim_ket.find_parameters_minimize(verbose=False, analytic_gradient=True, initial_state=psi0)
        print(results)
        self.assertTrue(np.isclose(results['approximation_ratio'], 0.9825745815865963))
        self.assertTrue(
            np.allclose(results['params'], np.array([0.40852239, 2.07747771, 0.85753934, 0.34297861, 1.04480957,
                                                     0.16922441])))

        print('Noisy:')
        results = sim_noisy.find_parameters_minimize(verbose=False, analytic_gradient=True, initial_state=rho0)
        self.assertTrue(np.isclose(results['approximation_ratio'], 0.9775935942298778))
        self.assertTrue(
            np.allclose(results['params'], np.array([0.40681189, 2.07749373, 0.85463347, 0.34377621, 1.04039812,
                                                     0.17029583])))

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

    def test_logical_codes(self):
        # Construct a known graph
        G = nx.random_regular_graph(1, 2)
        for e in G.edges:
            G[e[0]][e[1]]['weight'] = 1
        nx.draw_networkx(G)
        # Uncomment to visualize graph
        #plt.draw_graph(G)
        G = Graph(G)
        print('No logical encoding:')
        hc_qubit = hamiltonian.HamiltonianMIS(G)
        hamiltonians = [hc_qubit, hamiltonian.HamiltonianDriver()]
        sim = qaoa.SimulateQAOA(G, cost_hamiltonian=hc_qubit, hamiltonian=hamiltonians, noise_model=None,
                                noise=noises)
        # Set the default variational operators
        results = sim.find_parameters_brute(n=10)
        self.assertTrue(np.isclose(results['approximation_ratio'], 1))

        print('Two qubit code:')
        hc_two_qubit_code = hamiltonian.HamiltonianMIS(G, code=two_qubit_code)
        hamiltonians = [hc_two_qubit_code, hamiltonian.HamiltonianDriver(code=two_qubit_code)]

        sim_code = qaoa.SimulateQAOA(G, code=two_qubit_code, cost_hamiltonian=hc_two_qubit_code,
                                     hamiltonian=hamiltonians)

        # Find optimal parameters via brute force search
        sim_code.find_parameters_brute(n=10)
        self.assertTrue(np.isclose(results['approximation_ratio'], 1))

        print('Two qubit code with penalty:')
        # Set the default variational operators with a penalty Hamiltonian
        hc_qubit = hamiltonian.HamiltonianMIS(G, code=two_qubit_code)
        hamiltonians = [hc_qubit, hamiltonian.HamiltonianBookatzPenalty(code=two_qubit_code),
                        hamiltonian.HamiltonianDriver(code=two_qubit_code)]
        sim_penalty = qaoa.SimulateQAOA(G, cost_hamiltonian=hc_qubit, hamiltonian=hamiltonians, code=two_qubit_code)
        # You should get the same thing
        results = sim_penalty.find_parameters_brute(n=10)
        self.assertTrue(np.isclose(results['approximation_ratio'], 1))

        print('Jordan-Farhi-Shor code:')
        hc_jordan_farhi_shor = hamiltonian.HamiltonianMIS(G, code=jordan_farhi_shor)
        hamiltonians = [hc_jordan_farhi_shor, hamiltonian.HamiltonianDriver(code=jordan_farhi_shor)]

        sim_code = qaoa.SimulateQAOA(G, code=jordan_farhi_shor, cost_hamiltonian=hc_jordan_farhi_shor,
                                     hamiltonian=hamiltonians)

        sim_code.find_parameters_brute(n=10)
        self.assertTrue(np.isclose(results['approximation_ratio'], 1))


if __name__ == '__main__':
    unittest.main()
