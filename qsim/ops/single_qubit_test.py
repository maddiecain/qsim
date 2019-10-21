from qsim.ops import single_qubit


def test_multiply_single_spin():
    N = 6
    # initialize in |000000>
    psi0 = np.zeros(2**N)
    psi0[0] = 1

    # apply sigma_y on the first spin to get 1j|100000>
    psi1 = single_qubit.multiply_single_spin(psi0, 0, 2)
    assert psi1[1] == 1j

    # apply sigma_z on the first spin
    psi2 = single_qubit.multiply_single_spin(psi1, 0, 3)
    assert np.vdot(psi1,psi2) == -1

    # apply sigma_x on spin 2 through 6
    for i in range(1,N):
        psi1 = single_qubit.multiply_single_spin(psi1, i, 1)

    # vector norm should still be 1
    assert np.vdot(psi1, psi1) == 1

    # should be 1j|111111>
    assert psi1[-1] == 1j



def test_rotate_single_spin():
    N = 6
    # initialize in |000000>
    psi0 = np.zeros(2**N)
    psi0[0] = 1

    # rotate by exp(-1i*pi/4*sigma_y) every spin to get |++++++>
    psi1 = psi0
    for i in range(N):
        psi1 = single_qubit.rotate_single_spin(psi1, i, np.pi/4, 2)
    assert np.vdot(psi1, np.ones(2**N)/2**(N/2)) == 1

    # apply exp(-1i*pi/4*sigma_x)*exp(-1i*pi/4*sigma_z)
    # on every spin to get exp(-1j*N*pi/4)*|000000>
    for i in range(N):
        psi1 = single_qubit.rotate_single_spin(psi1, i, np.pi/4, 3)
        psi1 = single_qubit.rotate_single_spin(psi1, i, np.pi/4, 1)

    assert np.abs(np.vdot(psi0, psi1)*np.exp(1j*np.pi/4*N) - 1) <= 1e-10

