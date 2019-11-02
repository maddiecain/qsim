import numpy as np
import networkx as nx
import unittest

from qsim.qaoa import simulate
from qsim.state import State
from qsim import tools

g = nx.Graph()

g.add_edge(0,1,weight=1)
g.add_edge(0,2,weight=1)
g.add_edge(2,3,weight=1)
g.add_edge(0,4,weight=1)
g.add_edge(1,4,weight=1)
g.add_edge(3,4,weight=1)
g.add_edge(1,5,weight=1)
g.add_edge(2,5,weight=1)
g.add_edge(3,5,weight=1)


sim = simulate.SimulateQAOA(g, 5, 2, flag_z2_sym=False)

N = 6
# Initialize in |000000>
psi0 = np.zeros((2**N, 1))
psi0[0, 0] = 1

# TODO: add these in automatically?
sim.variational_params = [simulate.HamiltonianC(sim.C), simulate.HamiltonianB()]

class TestSimulate(unittest.TestCase):
    def test_evolve_B(self):
        state0 = State(psi0, N, is_ket=True)
        state1 = State(tools.outer_product(psi0, psi0), N, is_ket=False)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        sim.variational_params[1].evolve(state0, np.pi/2)

        # Should get (-1j)^N |111111>
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)
        self.assertTrue(state0.state[-1] == (-1j)**6)

        # Evolve by e^{-i (\pi/2) \sum_i X_i}
        sim.variational_params[1].evolve(state1, np.pi / 2)

        # Should get (-1j)^N |111111>
        self.assertTrue(state1.is_valid_dmatrix())
        self.assertTrue(state1.state[-1, -1] == 1)

    def test_multiply_B(self):
        state0 = State(psi0, N, is_ket=True)

        sim.variational_params[1].multiply(state0)
        psi1 = np.zeros((2**N, 1))
        for i in range(N):
            psi1[2**i, 0] = 1

        self.assertTrue(np.allclose(state0.state, psi1))

    def test_variational_grad(self):
        # Test that the calculated objective function and gradients are correct
        print('p=1:')
        F, Fgrad = sim.variational_grad(np.array([[1,0.5]]), is_ket=True)
        print(F, Fgrad)

        self.assertTrue(np.abs(F - 1.897011131463) <= 1e-10)
        self.assertTrue(np.all(np.abs(Fgrad - [14.287009047096, -0.796709998210]) <= 1e-10))

        F, Fgrad = sim.variational_grad(np.array([[1, 0.5]]), is_ket=False)
        print(F, Fgrad)

        self.assertTrue(np.abs(F - 1.897011131463) <= 1e-10)
        self.assertTrue(np.all(np.abs(Fgrad - [14.287009047096, -0.796709998210]) <= 1e-10))

        #do again for a p=2 case:
        print('p=2:')
        F, Fgrad=sim.variational_grad(np.array([[3, 2], [4, 5]]), is_ket=True)
        print(F, Fgrad)

        F, Fgrad = sim.variational_grad(np.array([[3, 2], [4, 5]]), is_ket=False)
        print(F, Fgrad)

        #and again, p=3
        print('p=3')
        param_matrix=np.array([[-1, 5], [4, -6], [15, 7]])
        F, Fgrad = sim.variational_grad(param_matrix, is_ket=True)
        print(F, Fgrad)

        F, Fgrad = sim.variational_grad(param_matrix, is_ket=False)
        print(F, Fgrad)



if __name__ == '__main__':
    unittest.main()
