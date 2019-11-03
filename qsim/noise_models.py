from typing import Tuple
from qsim.tools import operations


def multi_qubit_noise(s, p: float, single_qubit_noise):
    """Helper function to generate a system noise channel given a single qubit noise channel
    and noise rate. single_qubit_noise is a function that takes in an entire density matrix, a
    noise rate, and a qubit to act on, and returns the updated density matrix."""
    for i in range(s.shape[0]):
        single_qubit_noise(s, i, p)


def depolarize_single_qubit(s, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix
    
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
    """
    return s * (1 - p) + p / 3 * (operations.single_qubit_operation(s, i, 0, is_pauli=True) +
                                        operations.single_qubit_operation(s, i, 1, is_pauli=True) +
                                        operations.single_qubit_operation(s, i, 2, is_pauli=True))


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
    assert len(ps) == 3

    return (1 - sum(ps)) * s + (ps[0] * operations.single_qubit_operation(i, 1, is_pauli=True)
                                + ps[1] * operations.single_qubit_operation(i, 2, is_pauli=True)
                                + ps[2] * operations.single_qubit_operation(i, 3, is_pauli=True))


def amplitude_channel_single_qubit(s, i: int, p):
    # Define Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1-p)])]
    K1 = np.array([[0, np.sqrt(p)], [0, 0]])

    return operations.single_qubit_operation(s.state, i, K0, is_pauli=False, is_ket=False) + \
           operations.single_qubit_operation(s.state, i, K1, is_pauli=False, is_ket=False)