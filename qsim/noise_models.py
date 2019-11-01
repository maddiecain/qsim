import qsim.tools.operations
from typing import Tuple


def multi_qubit_noise(N: int, p: float, single_qubit_noise):
    """Helper function to generate a system noise channel given a single qubit noise channel
    and noise rate. single_qubit_noise is a function that takes in an entire density matrix, a
    noise rate, and a qubit to act on, and returns the updated density matrix."""
    def noisy_output(rho, rate):
        for i in range(N):
            rho = noise_single_qubit(rho, i, p)
        return rho
    return lambda rho: noisy_output(rho, p)

def depolarize_single_qubit(rho, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix
    
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
    """
    return (1-p)*rho + p/3*(qsim.operations.multiply_single_qubit_mixed(rho, i, 1)
                            + qsim.operations.multiply_single_qubit_mixed(rho, i, 2)
                            + qsim.operations.multiply_single_qubit_mixed(rho, i, 3))


def pauli_channel_single_qubit(rho, i: int, ps: Tuple[float]):
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

    return (1-sum(ps))*rho + (ps[0] * qsim.tools.single_qubit.multiply_single_qubit_mixed(rho, i, 1)
                              + ps[1] * qsim.tools.single_qubit.multiply_single_qubit_mixed(rho, i, 2)
                              + ps[2] * qsim.tools.single_qubit.multiply_single_qubit_mixed(rho, i, 3))
