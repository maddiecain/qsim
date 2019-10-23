import qsim.ops as qops
from typing import Tuple

def depolarize_one_qubit(rho, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix
    
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
    """
    return (1-p)*rho + p/3*(qops.multiply_single_qubit_mixed(rho, i, 1) 
                          + qops.multiply_single_qubit_mixed(rho, i, 2)
                          + qops.multiply_single_qubit_mixed(rho, i, 3))


def pauli_channel_one_qubit(rho, i: int, ps: Tuple[float]):
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

    return (1-sum(ps))*rho + ( ps[0] * qops.multiply_single_qubit_mixed(rho, i, 1) 
                             + ps[1] * qops.multiply_single_qubit_mixed(rho, i, 2)
                             + ps[2] * qops.multiply_single_qubit_mixed(rho, i, 3))
