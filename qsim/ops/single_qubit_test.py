from qsim.ops import single_qubit

import numpy as np

def test_multiply_single_qubit():
    N = 6
    # initialize in |000000>
    psi0 = np.zeros(2**N)
    psi0[0] = 1

    # apply sigma_y on the last qubit to get 1j|000001>
    psi1 = single_qubit.multiply_single_qubit(psi0, N-1, 2)
    assert psi1[1] == 1j

    # apply sigma_z on the last qubit
    psi2 = single_qubit.multiply_single_qubit(psi1, N-1, 3)
    assert np.vdot(psi1,psi2) == -1

    # apply sigma_x on qubit 0 through 4
    for i in range(N-1):
        psi1 = single_qubit.multiply_single_qubit(psi1, i, 1)

    # vector norm should still be 1
    assert np.vdot(psi1, psi1) == 1

    # should be 1j|111111>
    assert psi1[-1] == 1j


def test_rotate_single_qubit():
    N = 6
    # initialize in |000000>
    psi0 = np.zeros(2**N)
    psi0[0] = 1

    # rotate by exp(-1i*pi/4*sigma_y) every qubit to get |++++++>
    psi1 = psi0
    for i in range(N):
        psi1 = single_qubit.rotate_single_qubit(psi1, i, np.pi/4, 2)
    assert np.vdot(psi1, np.ones(2**N)/2**(N/2)) == 1

    # apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z)
    # on every qubit to get exp(-1j*N*pi/4)*|000000>
    for i in range(N):
        psi1 = single_qubit.rotate_single_qubit(psi1, i, np.pi/4, 3)
        psi1 = single_qubit.rotate_single_qubit(psi1, i, np.pi/4, 1)

    assert np.abs(np.vdot(psi0, psi1)*np.exp(1j*np.pi/4*N) - 1) <= 1e-10


def test_multiply_single_qubit_mixed():
    psi = np.kron([1,1], [1,0])/2**(1/2)
    rho = np.outer(psi, psi)

    # apply sigma_Y to first qubit
    psi2 = np.kron([1,-1],[1,0])*-1j/2**(1/2)
    rho2 = np.outer(psi2,psi2.conj())

    assert np.linalg.norm(single_qubit.multiply_single_qubit_mixed(
        rho, 0, 2) - rho2) <= 1e-10 


def test_operate_single_qubit_mixed():
    psi = np.kron([1,1], [1,0])/2**(1/2)
    rho = np.outer(psi, psi)

    # apply sigma_Y to first qubit
    psi2 = np.kron([1,-1],[1,0])*-1j/2**(1/2)
    rho2 = np.outer(psi2,psi2.conj())

    assert np.linalg.norm(single_qubit.operate_single_qubit_mixed(
        rho, 0, single_qubit.SY) - rho2) <= 1e-10

