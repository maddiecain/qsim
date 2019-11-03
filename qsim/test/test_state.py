import unittest
from qsim.state import State
import numpy as np
from qsim import tools

mixed_state = State(np.array([[.75, 0], [0, .25]]), 1, is_ket=False)
pure_state = State(np.array([[1, 0]]).T, 1, is_ket=True)
invalid_state = State(np.array([[-1, 0], [0, 0]]), 1, is_ket=False)


class TestState(unittest.TestCase):
    def test_is_pure_state(self):
        self.assertTrue(not mixed_state.is_pure_state())

    def test_is_valid_dmatrix(self):
        self.assertTrue(not invalid_state.is_valid_dmatrix())

    def test_change_basis(self):
        self.assertTrue(True)

    def test_expectation(self):
        self.assertTrue(np.real(mixed_state.expectation(np.identity(mixed_state.state.shape[0]))))

    def test_measurement(self):
        a = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.absolute(pure_state.measurement(a)[0])==1)
        b = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.absolute(mixed_state.measurement(b)[0])==1)

    def test_single_qubit_pauli(self):
        # Tests single qubit operation given a pauli index
        N = 3
        # Initialize in |000000>
        psi0 = np.zeros((2 ** N, 1))
        psi0[0][0] = 1
        state0 = State(psi0, N, is_ket=True)

        # Apply sigma_y on the first qubit to get 1j|000010>
        state0.single_qubit_operation(1, tools.SY, is_pauli=False)
        self.assertTrue(state0.state[2, 0] == 1j)

        # Apply sigma_z on the first qubit, state is -1j|000010>
        state0.single_qubit_operation(1, tools.SZ, is_pauli=False)
        self.assertTrue(state0.state[2, 0] == -1j)

        # Apply sigma_x on qubit 0 through 3
        for i in [0, 2]:
            state0.single_qubit_operation(i, tools.SX, is_pauli=False)

        # Vector is still normalized
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)

        # Should be -1j|111111>
        self.assertTrue(state0.state[-1, 0] == -1j)

        psi1 = tools.tensor_product(np.array([1, 1]), np.array([1, 0])) / 2 ** (1 / 2)
        state1 = State(tools.outer_product(psi1, psi1), 2, is_ket=False)

        # Apply sigma_Y to first qubit
        psi2 = np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)
        rho2 = tools.outer_product(psi2, psi2)

        state1.single_qubit_operation(1, tools.SY, is_pauli=False)
        self.assertTrue(np.linalg.norm(state1.state - rho2) <= 1e-10)

        # TEST PAULI
        state0 = State(psi0, N, is_ket=True)

        # Apply sigma_y on the first qubit to get 1j|000010>
        state0.single_qubit_operation(1, tools.SIGMA_Y_IND, is_pauli=True)
        #print(state0.state)
        self.assertTrue(state0.state[2, 0] == 1j)

        # Apply sigma_z on the first qubit, state is -1j|000010>
        state0.single_qubit_operation(1, tools.SIGMA_Z_IND, is_pauli=True)
        self.assertTrue(state0.state[2, 0] == -1j)

        # Apply sigma_x on qubit 0 through 4
        for i in [0, 2]:
            state0.single_qubit_operation(i, tools.SIGMA_X_IND, is_pauli=True)

        # Vector is still normalized
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)

        # Should be -1j|111111>
        self.assertTrue(state0.state[-1, 0] == -1j)

        psi1 = tools.tensor_product(np.array([1, 1]), np.array([1, 0])) / 2 ** (1 / 2)
        state1 = State(tools.outer_product(psi1, psi1), 2, is_ket=False)

        # Apply sigma_Y to first qubit
        psi2 = np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)
        rho2 = tools.outer_product(psi2, psi2)

        state1.single_qubit_operation(1, tools.SIGMA_Y_IND, is_pauli=True)
        self.assertTrue(np.linalg.norm(state1.state - rho2) <= 1e-10)

    def test_single_qubit_rotation(self):
        # Initialize in |000000>
        psi0 = np.zeros(2 ** 6)
        psi0[0] = 1
        state0 = State(psi0, 6, is_ket=True)
        state1 = State(psi0, 6, is_ket=True)

        # Rotate by exp(-1i*pi/4*sigma_y) every qubit to get |++++++>
        for i in range(state0.N):
            state0.single_qubit_rotation(i, np.pi / 4, tools.SY)
        self.assertAlmostEqual(np.vdot(state0.state, np.ones(2 ** state0.N) / 2 ** (state0.N / 2)), 1)

        # Apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z) on every qubit to get exp(-1j*N*pi/4)*|000000>
        for i in range(state0.N):
            state0.single_qubit_rotation(i, np.pi / 4, tools.SZ)
            state0.single_qubit_rotation(i, np.pi / 4, tools.SX)

        self.assertTrue(np.abs(np.vdot(state1.state, state0.state) * np.exp(1j * np.pi / 4 * state1.N) - 1) <= 1e-10)

    def test_single_qubit_operation(self):
        psi = tools.tensor_product(np.array([1, 1]), np.array([1, 0])) / 2 ** (1 / 2)
        state0 = State(tools.outer_product(psi, psi), 2, is_ket=False)
        state1 = State(np.array([psi]), 2, is_ket=True)

        # Apply sigma_Y to first qubit
        psi2 = np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)
        rho2 = tools.outer_product(psi2, psi2)

        # Apply single qubit operation to dmatrix
        state0.single_qubit_operation(1, tools.SY)
        self.assertTrue(np.linalg.norm(state0.state - rho2) <= 1e-10)

        # Test on ket
        state1.single_qubit_operation(1, tools.SY)
        self.assertTrue(np.linalg.norm(state1.state - psi2) <= 1e-10)


if __name__ == '__main__':
    unittest.main()
