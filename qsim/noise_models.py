from qsim.state import State
from typing import Tuple


def multi_qubit_noise(s: State, p: float, single_qubit_noise):
    """Helper function to generate a system noise channel given a single qubit noise channel
    and noise rate. single_qubit_noise is a function that takes in an entire density matrix, a
    noise rate, and a qubit to act on, and returns the updated density matrix."""
    for i in range(s.N):
        single_qubit_noise(s, i, p)

def depolarize_single_qubit(s: State, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix
    
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
    """
    s.state = (1-p)*s.state + p/3*(s.single_qubit_operation(i, 1, is_pauli=True)
                            + s.single_qubit_operation(i, 2, is_pauli=True)
                            + s.single_qubit_operation(i, 3, is_pauli=True))


def pauli_channel_single_qubit(s: State, i: int, ps: Tuple[float]):
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

    s.state = (1-sum(ps))*s.state + (ps[0] * s.single_qubit_operation(i, 1, is_pauli = True)
                              + ps[1] * s.single_qubit_operation(i, 2, is_pauli=True)
                              + ps[2] * s.single_qubit_operation(i, 3, is_pauli=True))
