from typing import Tuple
from qsim.tools import operations, tools
import numpy as np
from qsim.state import State


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
    return (1 - sum(ps)) * s + (ps[0] * operations.single_qubit_operation(i, tools.SIGMA_X_IND, is_pauli=True)
                                + ps[1] * operations.single_qubit_operation(i, tools.SIGMA_Y_IND, is_pauli=True)
                                + ps[2] * operations.single_qubit_operation(i, tools.SIGMA_Z_IND, is_pauli=True))


def amplitude_channel_single_qubit(s, i: int, p):
    # Define Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    K1 = np.array([[0, np.sqrt(p)], [0, 0]])

    return operations.single_qubit_operation(s, i, K0, is_pauli=False, is_ket=False) + \
           operations.single_qubit_operation(s, i, K1, is_pauli=False, is_ket=False)



class Noise(object):
    def __init__(self, channel=None):
        self.quantum_channel = channel
        if channel is None:
            self.quantum_channel = self.channel

    def all_qubit_channel(self, s: State):
        for i in range(s.N):
            s.state = self.channel(s.state, i)

    def channel(self, s, i):
        # Do absolutely nothing
        return s


class DepolarizingNoise(Noise):
    def __init__(self, p):
        super().__init__(self.channel)
        self.p = p

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return depolarize_single_qubit(s, i, self.p)


class PauliNoise(Noise):
    def __init__(self, ps: Tuple[float]):
        super().__init__(self.channel)
        assert len(ps) == 3
        self.ps = ps

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return pauli_channel_single_qubit(s, i, self.ps)


class AmplitudeDampingNoise(Noise):
    def __init__(self, p):
        super().__init__(self.channel)
        self.p = p

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return amplitude_channel_single_qubit(s, i, self.p)


