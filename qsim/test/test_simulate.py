import numpy as np
import networkx as nx
import unittest

from qsim.qaoa import simulate
from qsim.noise import noise_models
from qsim import hamiltonian

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
sim_ket = simulate.SimulateQAOA(g, 1, 2, is_ket=True, mis=False)
sim_noisy = simulate.SimulateQAOA(g, 1, 2, is_ket=False, mis=False)
sim_noisy.noise = []


N = 10
# Initialize in |000000>
psi0 = np.zeros((2 ** N, 1))
psi0[0, 0] = 1

sim.hamiltonian = [hamiltonian.HamiltonianC(g, mis=False), hamiltonian.HamiltonianB()]
sim_ket.hamiltonian = [hamiltonian.HamiltonianC(g, mis=False), hamiltonian.HamiltonianB()]
sim_noisy.hamiltonian = [hamiltonian.HamiltonianC(g, mis=False), hamiltonian.HamiltonianB()]

sim.noise = [noise_models.LindbladNoise(), noise_models.LindbladNoise()]
sim_ket.noise = [noise_models.LindbladNoise(), noise_models.LindbladNoise()]
sim_noisy.noise = [noise_models.DepolarizingNoise(.001), noise_models.DepolarizingNoise(.001)]


class TestSimulate(unittest.TestCase):
    def test_variational_grad(self):
        # Test that the calculated objective function and gradients are correct
        # p = 1
        F, Fgrad = sim_ket.variational_grad(np.array([1, 0.5]))
        self.assertTrue(np.abs(F - 1.897011131463) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([14.287009047096, -0.796709998210])) <= 1e-5))

        # p = 1 density matrix
        F, Fgrad = sim.variational_grad(np.array([1, 0.5]))
        print(Fgrad)
        self.assertTrue(np.abs(F - 1.897011131463) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([14.287009047096, -0.796709998210])) <= 1e-5))

        # p = 1 noiY
        F, Fgrad = sim_noisy.variational_grad(np.array([1, 0.5]))
        self.assertTrue(np.abs(F - 1.8869139555669938) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([14.21096392, -0.79246937])) <= 1e-5))

        # p = 2
        F, Fgrad = sim_ket.variational_grad(np.array([3, 4, 2, 5]))
        self.assertTrue(np.abs(F + 0.5868545288327245) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([3.82877928, 2.10271544, 5.21809702, 4.99717856])) <= 1e-5))

        F, Fgrad = sim.variational_grad(np.array([3, 4, 2, 5]))
        self.assertTrue(np.abs(F + 0.5868545288327245) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([3.82877928, 2.10271544, 5.21809702, 4.99717856])) <= 1e-5))

        # p = 3
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F + 1.2541687509598878) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-5.5862387, -3.99650097, -2.43971594, -0.29729297, -3.66785565,
                                                        -3.35531478])) <= 1e-5))

        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F + 1.2541687509598878) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-5.5862387, -3.99650097, -2.43971594, -0.29729297, -3.66785565,
                                                        -3.35531478])) <= 1e-5))

    def test_run(self):
        sim.p = 1
        sim_noisy.p = 1
        # p = 1 density matrix
        self.assertAlmostEqual(sim.run([1, .5]), 1.897011131463)
        self.assertAlmostEqual(sim_ket.run([1, .5]), 1.897011131463)

        # See how things look with noise
        self.assertAlmostEqual(sim_noisy
                               .run([1, .5]), 1.8869139555669938)

        # Higher depth circuit
        params = np.array([-1, 4, 15, 5, -6, 7])
        F, Fgrad = sim.variational_grad(params)
        self.assertTrue(np.abs(F + 1.2541687509598878) <= 1e-5)
        self.assertTrue(np.all(np.abs(Fgrad - np.array([-5.5862387, -3.99650097, -2.43971594, -0.29729297, -3.66785565,
                                                        -3.35531478])) <= 1e-5))

    def test_find_optimal_params(self):
        sim.p = 3
        sim_noisy.p = 3
        print('Noiseless:')
        params = sim.find_parameters_minimize()
        self.assertTrue(np.allclose(params.x, np.array([0.2042597,0.42876983, 0.52240463,-0.50668092,-0.34297845,-0.16922362])))
        print('Noisy:')
        params = sim_noisy.find_parameters_minimize()
        self.assertTrue(np.allclose(params.x, np.array([0.20340663,  0.42731716 , 0.52019853, -0.50669633, -0.3437759,  -0.17029569])))

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
