import unittest
from qsim.state import State, TwoQubitCode
import numpy as np
from qsim import tools
import time
mixed_state = State(np.array([[.75, 0], [0, .25]]), 1, is_ket=False)
pure_state = State(np.array([[1, 0]]).T, 1, is_ket=True)
invalid_state = State(np.array([[-1, 0], [0, 0]]), 1, is_ket=False)
psi = np.zeros((2**4, 1))
psi[0,0] = 1
logical_state = TwoQubitCode(psi, 2, is_ket=True)
normal_state = State(psi, 4, is_ket=True)

class TestState(unittest.TestCase):
    def test_is_pure_state(self):
        self.assertTrue(not mixed_state.is_pure_state())

    def test_is_valid_dmatrix(self):
        self.assertTrue(not invalid_state.is_valid_dmatrix())

    def test_expectation(self):
        self.assertTrue(np.real(mixed_state.expectation(np.identity(mixed_state.state.shape[0]))))

    def test_measurement(self):
        a = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.absolute(pure_state.measurement(a)[0]) == 1)
        b = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.absolute(mixed_state.measurement(b)[0]) == 1)

    def test_single_qubit_pauli(self):
        # Tests single qubit operation given a pauli index
        N = 6
        # Initialize in |000000>
        psi0 = np.zeros((2 ** N, 1))
        psi0[0][0] = 1
        state0 = State(psi0, N, is_ket=True)

        # Apply sigma_y on the second qubit to get 1j|010000>
        state0.opY(1)
        self.assertTrue(state0.state[2 ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, state is -1j|010000>
        state0.opZ(1)
        self.assertTrue(state0.state[2 ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubit 0 through 3
        for i in range(N):
            if i != 1:
                state0.opX(i)

        # Vector is still normalized
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)

        # Should be -1j|111111>
        self.assertTrue(state0.state[-1, 0] == -1j)

        # Density matrix test
        psi1 = np.array([tools.tensor_product((np.array([1, 1]), np.array([1, 0]))) / 2 ** (1 / 2)]).T
        state1 = State(tools.outer_product(psi1, psi1), 2, is_ket=False)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        state1.opY(0)
        self.assertTrue(np.linalg.norm(state1.state - rho2) <= 1e-10)

        # Test pauli
        state0 = State(psi0, N, is_ket=True)

        # Apply sigma_y on the second qubit to get 1j|010000>
        state0.opY(1)
        self.assertTrue(state0.state[2 ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, state is -1j|010000>
        state0.opZ(1)
        self.assertTrue(state0.state[2 ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubits
        for i in range(N):
            if i != 1:
                state0.opX(i)

        # Vector is still normalized
        self.assertTrue(np.vdot(state0.state, state0.state) == 1)

        # Should be -1j|111111>
        self.assertTrue(state0.state[-1, 0] == -1j)

        # Test pauli operations with density matrices
        psi1 = np.array([tools.tensor_product((np.array([1, 1]), np.array([1, 0]))) / 2 ** (1 / 2)]).T
        state1 = State(tools.outer_product(psi1, psi1), 2, is_ket=False)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        state1.opY(0)
        self.assertTrue(np.linalg.norm(state1.state - rho2) <= 1e-10)

    def test_single_qubit_rotation(self):
        # Initialize in |000000>
        psi0 = np.zeros(2 ** 6)
        psi0[0] = 1
        state0 = State(psi0, 6, is_ket=True)
        state1 = State(psi0, 6, is_ket=True)

        # Rotate by exp(-1i*pi/4*sigma_y) every qubit to get |++++++>
        for i in range(state0.N):
            state0.single_qubit_rotation(i, np.pi / 4, tools.Y())
        self.assertAlmostEqual(np.vdot(state0.state, np.ones(2 ** state0.N) / 2 ** (state0.N / 2)), 1)

        # Apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z) on every qubit to get exp(-1j*N*pi/4)*|000000>
        for i in range(state0.N):
            state0.single_qubit_rotation(i, np.pi / 4, tools.Z())
            state0.single_qubit_rotation(i, np.pi / 4, tools.X())

        self.assertTrue(np.abs(np.vdot(state1.state, state0.state) * np.exp(1j * np.pi / 4 * state1.N) - 1) <= 1e-10)

    def test_single_qubit_operation(self):
        psi = np.array([tools.tensor_product((np.array([1, 1]), np.array([1, 0]))) / 2 ** (1 / 2)]).T
        state0 = State(tools.outer_product(psi, psi), 2, is_ket=False)
        state1 = State(psi, 2, is_ket=True)

        # Apply sigma_Y to first qubit
        psi2 = np.array([np.kron([1, -1], [1, 0]) * -1j / 2 ** (1 / 2)]).T
        rho2 = tools.outer_product(psi2, psi2)

        # Apply single qubit operation to dmatrix
        state0.single_qubit_operation(0, tools.Y())
        self.assertTrue(np.linalg.norm(state0.state - rho2) <= 1e-10)

        # Test on ket
        state1.single_qubit_operation(0, tools.Y())
        self.assertTrue(np.linalg.norm(state1.state - psi2) <= 1e-10)

    def test_logical_qubit(self):
        logical_state.opY(1)
        normal_state.opY(2)
        normal_state.opZ(3)
        # Should get out 1j|0010>
        self.assertTrue(np.allclose(logical_state.state, normal_state.state))
        logical_state.opX(0)
        normal_state.opX(1)
        self.assertTrue(np.allclose(logical_state.state, normal_state.state))

        logical_state.all_qubit_rotation(np.pi / 2, logical_state.X)
        normal_state.single_qubit_rotation(0, np.pi / 2, normal_state.X)
        normal_state.single_qubit_rotation(2, np.pi / 2, normal_state.X)

        # Should get 1j|0010>
        self.assertTrue(np.allclose(logical_state.state, normal_state.state))


if __name__ == '__main__':
    unittest.main()
