import unittest
from qsim.state import *
import numpy as np
from qsim import tools

mixed_state = State(np.array([[.75, 0], [0, .25]]), 1, is_ket=False)
pure_state = State(np.array([[1, 0]]).T, 1, is_ket=True)
invalid_state = State(np.array([[-1, 0], [0, 0]]), 1, is_ket=False)
two_qubit_code_logical_state = TwoQubitCode(tools.tensor_product([TwoQubitCode.basis[0], TwoQubitCode.basis[0]]), 2, is_ket=True)
jfs_logical_state = JordanFarhiShor(tools.tensor_product([JordanFarhiShor.basis[0], JordanFarhiShor.basis[0]]), 2, is_ket=True)
three_qubit_code_logical_state = ThreeQubitCode(tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[0]]), 2, is_ket=True)
three_qubit_code_density_matrix = ThreeQubitCode(tools.outer_product(tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[0]]), tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[0]])), 2, is_ket=False)

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
        state0.single_qubit_pauli(1, 'Y', overwrite=True)
        self.assertTrue(state0.state[2 ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, state is -1j|010000>
        state0.single_qubit_pauli(1, 'Z', overwrite=True)
        self.assertTrue(state0.state[2 ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubit 0 through 3
        for i in range(N):
            if i != 1:
                state0.single_qubit_pauli(i, 'X', overwrite=True)

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

        state1.single_qubit_pauli(0, 'Y')
        self.assertTrue(np.linalg.norm(state1.state - rho2) <= 1e-10)

        # Test pauli
        state0 = State(psi0, N, is_ket=True)

        # Apply sigma_y on the second qubit to get 1j|010000>
        state0.single_qubit_pauli(1, 'Y')
        self.assertTrue(state0.state[2 ** (N - 2), 0] == 1j)

        # Apply sigma_z on the second qubit, state is -1j|010000>
        state0.single_qubit_pauli(1, 'Z')
        self.assertTrue(state0.state[2 ** (N - 2), 0] == -1j)

        # Apply sigma_x on qubits
        for i in range(N):
            if i != 1:
                state0.single_qubit_pauli(i, 'X')

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

        state1.single_qubit_pauli(0, 'Y')
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
        two_qubit_code_logical_state.single_qubit_pauli(1, 'Y', overwrite=True)
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(two_qubit_code_logical_state.state, 1j*tools.tensor_product([TwoQubitCode.basis[0], TwoQubitCode.basis[1]])))
        self.assertTrue(np.allclose(two_qubit_code_logical_state.single_qubit_pauli(1, 'Z', overwrite=False), -1j*tools.tensor_product([TwoQubitCode.basis[0], TwoQubitCode.basis[1]])))
        two_qubit_code_logical_state.single_qubit_pauli(0, 'X', overwrite=True)
        # Should get out -1j|1L>|1L>
        self.assertTrue(np.allclose(two_qubit_code_logical_state.state, 1j*tools.tensor_product([TwoQubitCode.basis[1], TwoQubitCode.basis[1]])))

        two_qubit_code_logical_state.all_qubit_rotation(np.pi / 2, two_qubit_code_logical_state.X, overwrite=True)
        self.assertTrue(np.allclose(two_qubit_code_logical_state.state, -1j*tools.tensor_product([TwoQubitCode.basis[0], TwoQubitCode.basis[0]])))


        jfs_logical_state.single_qubit_pauli(1, 'Y', overwrite=True)
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(jfs_logical_state.state,
                                    1j * tools.tensor_product([JordanFarhiShor.basis[0], JordanFarhiShor.basis[1]])))
        self.assertTrue(np.allclose(jfs_logical_state.single_qubit_pauli(1, 'Z', overwrite=False),
                                    -1j * tools.tensor_product([JordanFarhiShor.basis[0], JordanFarhiShor.basis[1]])))
        jfs_logical_state.single_qubit_pauli(0, 'X', overwrite=True)
        # Should get out 1j|1L>|1L>
        self.assertTrue(np.allclose(jfs_logical_state.state,
                                    1j * tools.tensor_product([JordanFarhiShor.basis[1], JordanFarhiShor.basis[1]])))

        jfs_logical_state.all_qubit_rotation(np.pi / 2, jfs_logical_state.X, overwrite=True)

        # Should get -1j|0L>|0L>
        self.assertTrue(np.allclose(jfs_logical_state.state, -1j*tools.tensor_product([JordanFarhiShor.basis[0], JordanFarhiShor.basis[0]])))

        three_qubit_code_logical_state.single_qubit_pauli(1, 'Y', overwrite=True)
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(three_qubit_code_logical_state.state,
                                    1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[1]])))
        self.assertTrue(np.allclose(three_qubit_code_logical_state.single_qubit_pauli(1, 'Z', overwrite=False),
                                    -1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[1]])))
        three_qubit_code_logical_state.single_qubit_pauli(0, 'X', overwrite=True)
        # Should get out 1j|1L>|1L>
        self.assertTrue(np.allclose(three_qubit_code_logical_state.state,
                                    1j * tools.tensor_product([ThreeQubitCode.basis[1], ThreeQubitCode.basis[1]])))

        three_qubit_code_logical_state.all_qubit_rotation(np.pi / 2, three_qubit_code_logical_state.X, overwrite=True)

        # Should get -1j|0L>|0L>
        self.assertTrue(np.allclose(three_qubit_code_logical_state.state,
                                    -1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[0]])))

        # Test density matrix single qubit operations
        three_qubit_code_density_matrix.single_qubit_pauli(1, 'Y', overwrite=True)
        # Should get out 1j|0L>|1L>
        self.assertTrue(np.allclose(three_qubit_code_density_matrix.state, tools.outer_product(
            1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[1]]),
            1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[1]]))
                                    ))
        self.assertTrue(np.allclose(three_qubit_code_density_matrix.single_qubit_pauli(1, 'Z', overwrite=False),
                                    tools.outer_product(
                                        -1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[1]]),
                                        -1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[1]]))
                                    ))
        three_qubit_code_density_matrix.single_qubit_pauli(0, 'X', overwrite=True)
        # Should get out 1j|1L>|1L>
        self.assertTrue(np.allclose(three_qubit_code_density_matrix.state, tools.outer_product(
            1j * tools.tensor_product([ThreeQubitCode.basis[1], ThreeQubitCode.basis[1]]),
            1j * tools.tensor_product([ThreeQubitCode.basis[1], ThreeQubitCode.basis[1]]))
                                    ))

        three_qubit_code_density_matrix.all_qubit_rotation(np.pi / 2, three_qubit_code_density_matrix.X, overwrite=True)

        # Should get -1j|0L>|0L>
        self.assertTrue(np.allclose(three_qubit_code_density_matrix.state, tools.outer_product(
            -1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[0]]),
            -1j * tools.tensor_product([ThreeQubitCode.basis[0], ThreeQubitCode.basis[0]]))
                                    ))


if __name__ == '__main__':
    unittest.main()
