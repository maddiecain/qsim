import numpy as np
import networkx as nx
import unittest

from qsim.qaoa import simulate
from qsim import noise_models
from qsim.state import State
from qsim import tools
from qsim.qaoa import variational_parameters

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

sim = simulate.SimulateQAOA(g, 1, 2, is_ket=False)
sim_noisy = simulate.SimulateQAOA(g, 1, 2,  noise_model=noise_models.depolarize_single_qubit, is_ket=False)
sim_optimize = simulate.SimulateQAOA(g, 6, 2,  is_ket=False)
sim_noisy_optimize = simulate.SimulateQAOA(g, 6, 2, noise_model=noise_models.depolarize_single_qubit,
                                           error_probability=.001, is_ket=False)

N = 10
# Initialize in |000000>
psi0 = np.zeros((2 ** N, 1))
psi0[0, 0] = 1

sim.variational_operators = [variational_parameters.HamiltonianC(sim.C), variational_parameters.HamiltonianB()]
sim_noisy.variational_operators = [variational_parameters.HamiltonianC(sim.C), variational_parameters.HamiltonianB()]
sim_optimize.variational_operators = [variational_parameters.HamiltonianC(sim.C), variational_parameters.HamiltonianB()]
sim_noisy_optimize.variational_operators = [variational_parameters.HamiltonianC(sim.C), variational_parameters.HamiltonianB()]


class TestSimulate(unittest.TestCase):
    def test_evolve_B(self):
        state0 = State(psi0, N, is_ket=True)
        state1 = State(tools.outer_product(psi0, psi0), N, is_ket=False)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        sim.variational_operators[1].evolve(state0, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)
        self.assertTrue(state0.state[-1] == (-1j) ** 6)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        sim.variational_operators[1].evolve(state1, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(state1.is_valid_dmatrix())
        self.assertTrue(state1.state[-1, -1] == 1)

    def test_multiply_B(self):
        state0 = State(psi0, N, is_ket=True)

        sim.variational_operators[1].multiply(state0)
        psi1 = np.zeros((2 ** N, 1))
        for i in range(N):
            psi1[2 ** i, 0] = 1

        self.assertTrue(np.allclose(state0.state, psi1))

    def test_variational_grad(self):
        # Test that the calculated objective function and gradients are correct
        # p = 1
        F, Fgrad = sim.variational_grad(np.array([1, 0.5]))
        self.assertTrue(np.abs(F - 1.897011131463) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([14.287009047096, -0.796709998210])) <= 1e-5))

        # p = 1 density matrix
        F, Fgrad = sim.variational_grad(np.array([1, 0.5]))

        self.assertTrue(np.abs(F - 1.897011131463) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([14.287009047096, -0.796709998210])) <= 1e-5))

        # p = 2
        F, Fgrad = sim.variational_grad(np.array([3, 2, 4, 5]))
        self.assertTrue(np.abs(F + 0.5868545288327245) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([3.82877928, 2.10271544, 5.21809702, 4.99717856])) <= 1e-5))

        F, Fgrad = sim.variational_grad(np.array([3, 2, 4, 5]))
        self.assertTrue(np.abs(F + 0.5868545288327245) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([3.82877928, 2.10271544, 5.21809702, 4.99717856])) <= 1e-5))

        # p = 3
        params = np.array([-1, 5, 4, -6, 15, 7])
        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F + 1.2541687509598878) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-5.5862387, -3.99650097, -2.43971594, -0.29729297, -3.66785565,
                                                        -3.35531478])) <= 1e-5))

        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F + 1.2541687509598878) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-5.5862387, -3.99650097, -2.43971594, -0.29729297, -3.66785565,
                                                        -3.35531478])) <= 1e-5))

    def test_run(self):
        # p = 1 density matrix
        sim.variational_operators[0].param = np.array([[1]])
        sim.variational_operators[1].param = np.array([[0.5]])
        self.assertAlmostEqual(sim.run(), 1.897011131463)

        # See how things look with noise
        sim_noisy.variational_operators[0].param = np.array([[1]])
        sim_noisy.variational_operators[1].param = np.array([[0.5]])
        self.assertAlmostEqual(sim_noisy.run(), 1.88857778)

        # Higher depth circuit
        gammas = np.array([[0.17332659181015764, 1.091473300662595, 0.7719939217625529, 0.05322040660876185, 1.103646670462646,
                 0.28187402270267076]]).T
        betas = np.array([[-1.3657952653481733, -1.7104025614326148, -2.047579754894323, -1.464823733400459, -2.228623944977355,
                -2.9730309886756365]]).T

        sim_noisy.variational_operators[0].param = gammas
        sim_noisy.variational_operators[1].param = betas
        self.assertAlmostEqual(sim_noisy.run(), 2.065356423346786)

    def test_find_optimal_params(self):
        print('Noiseless:', sim_optimize.find_optimal_params())
        print('Noisy:', sim_noisy_optimize.find_optimal_params())

if __name__ == '__main__':
    unittest.main()
