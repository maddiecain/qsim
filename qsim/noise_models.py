import qsim.tools.single_qubit
from typing import Tuple


class NoiseModel(object):
    def __init__(self, rate, model):
        """Rate is a fraction describing the probability the noise channel is applied
        to a single qubit during a unit time step.

        Model is a function specifying the noise channel to apply to the entire system."""
        self.rate = rate
        self.model = model

    def multi_qubit_noise(self, N, single_qubit_noise):
        """Helper function to generate a system noise channel given a single qubit noise channel
        and noise rate. single_qubit_noise is a function that takes in an entire density matrix, a
        noise rate, and a qubit to act on, and returns the updated density matrix."""
        def noisy_output(rho, rate):
            for i in range(N):
                if rate > np.random.uniform():
                    rho = noise_single_qubit(rho, i, self.rate)
            return rho
        return lambda rho: noisy_output(rho, self.rate)



def depolarize_single_qubit(rho, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix
    
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
    """
    return (1-p)*rho + p/3*(qsim.single_qubit.multiply_single_qubit_mixed(rho, i, 1)
                            + qsim.single_qubit.multiply_single_qubit_mixed(rho, i, 2)
                            + qsim.single_qubit.multiply_single_qubit_mixed(rho, i, 3))


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
