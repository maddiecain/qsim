from typing import Tuple
from qsim.tools import operations, tools
import numpy as np
from qsim.state import *


class LindbladNoise(object):
    def __init__(self, povm, weights=None):
        # POVM and weights are lists
        self.povm = povm
        if weights is None:
            self.weights = [1] * povm.shape[-1]

    def all_qubit_channel(self, s: State):
        for i in range(s.N):
            s.state = self.channel(s.state, i)

    def channel(self, s, i):
        # Output placeholder
        a = s * (1 - np.sum(self.weights))
        for j in range(len(self.povm)):
            a = a + self.weights[j] * (operations.single_qubit_operation(s, i, self.povm[j], is_pauli=False) -
                                       1 / 2 * operations.left_multiply(s, i, self.povm[j].conj().T @ self.povm[j]) - 1 / 2 * operations.right_multiply(
                        s, i, self.povm[j].conj().T @ self.povm[j]))
        return a


def random_pauli_noise(s, i, m, theta):
    # Apply the rotation e^{i*theta*(m-hat * pauli vector)} to qubit i
    pauli = m[0] * State.SX + m[1] * State.SY + m[2] * State.SZ
    s.state = operations.single_qubit_rotation(s.state, i, theta, pauli, is_ket=s.is_ket)


def depolarize_single_qubit(s, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
    return s * (1 - p) + p / 3 * (operations.single_qubit_operation(s, i, tools.SIGMA_X_IND, is_pauli=True) +
                                  operations.single_qubit_operation(s, i, tools.SIGMA_Y_IND, is_pauli=True) +
                                  operations.single_qubit_operation(s, i, tools.SIGMA_Z_IND, is_pauli=True))


def pauli_channel_single_qubit(s, i: int, ps: Tuple[float]):
    """ Perform general Pauli channel on the i-th qubit of an input density matrix

        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            ps = a 3-tuple (px, py, pz) indicating the probability
                 of applying each Pauli operator

        Returns:
            (1-px-py-pz) * rho + px * Xi * rho * Xi
                               + py * Yi * rho * Yi
                               + pz * Zi * rho * Zi
    """
    return (1 - sum(ps)) * s + (ps[0] * operations.single_qubit_operation(s, i, tools.SIGMA_X_IND, is_pauli=True)
                                + ps[1] * operations.single_qubit_operation(s, i, tools.SIGMA_Y_IND, is_pauli=True)
                                + ps[2] * operations.single_qubit_operation(s, i, tools.SIGMA_Z_IND, is_pauli=True))


def amplitude_channel_single_qubit(s, i: int, p):
    # Define Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    K1 = np.array([[0, np.sqrt(p)], [0, 0]])

    return operations.single_qubit_operation(s, i, K0, is_pauli=False, is_ket=False) + \
           operations.single_qubit_operation(s, i, K1, is_pauli=False, is_ket=False)


class DepolarizingNoise(LindbladNoise):
    def __init__(self, p):
        super().__init__(np.array([(np.sqrt(1 - 3 * p)) * np.identity(2), np.sqrt(p) * tools.SX, np.sqrt(p) * tools.SY,
                                   np.sqrt(p) * tools.SZ]))
        self.p = p

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return depolarize_single_qubit(s, i, self.p)


class PauliNoise(LindbladNoise):
    def __init__(self, p: Tuple[float]):
        super().__init__(self, np.array(
            [(np.sqrt(1 - np.sum(p))) * np.identity(2), np.sqrt(p[0]) * tools.SX, np.sqrt(p[1]) * tools.SY,
             np.sqrt(p[2]) * tools.SZ]))
        assert len(p) == 3
        self.p = p

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return pauli_channel_single_qubit(s, i, self.p)


class AmplitudeDampingNoise(LindbladNoise):
    def __init__(self, p):
        super().__init__(np.array([[[1, 0], [0, np.sqrt(1 - p)]], [[0, np.sqrt(p)], [0, 0]]]))
        self.p = p

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return amplitude_channel_single_qubit(s, i, self.p)


class ZProjectionNoise(LindbladNoise):
    def __init__(self, p):
        projectors = 1 / np.sqrt(2) * np.array([[[1, 0], [0, 0]], [[0, 0], [0, -1]]])
        super().__init__(projectors, weights=(p, p))
        self.p = p


"""class ZProjectionNoise(LindbladNoise):
    def __init__(self, p):
        projectors = 1/np.sqrt(2)*np.array([[[1, 0], [0, 0]], [[0, 0], [0, -1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 1j], [0, 0]], [[0, 0], [-1j, 0]]])
        super().__init__(np.array([np.sqrt(1-np.sum(p))*np.identity(2), projectors[4]*np.sqrt(p[2]), projectors[5]*np.sqrt(p[2]), projectors[2]*np.sqrt(p[1]), projectors[3]*np.sqrt(p[1]), projectors[0]*np.sqrt(p[0]), projectors[1]*np.sqrt(p[0])]))
        self.p = p"""
