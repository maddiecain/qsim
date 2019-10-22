from qsim.opensys import noise_models
import qsim.ops as qops
import numpy as np

def test_depolarize_one_qubit():
    SX, SY, SZ = qops.PAULI_MATRICES
    op1 = np.kron(SX, SY)
    p = 0.093
    op2 = noise_models.depolarize_one_qubit(op1, 1, p)
    op3 = noise_models.depolarize_one_qubit(op2, 0, 2*p)
    assert np.linalg.norm(op2 - op1*(1-4*p/3)) <= 1e-10
    assert np.linalg.norm(op3 - op1*(1-4*p/3)*(1-8*p/3)) <= 1e-10
    
