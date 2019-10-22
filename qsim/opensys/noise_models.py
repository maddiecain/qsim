import qsim.ops as qops

def depolarize_one_qubit(rho, i: int, p: float):
    """ Perform depolarizing channel on the i-th qubit of an input density matrix
    
        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
    """
    
    assert len(rho.shape) == 2 # rho should be a matrix
    
    return (1-p)*rho + p/3*(qops.multiply_single_qubit_mixed(rho, i, 1) 
                          + qops.multiply_single_qubit_mixed(rho, i, 2)
                          + qops.multiply_single_qubit_mixed(rho, i, 3))

