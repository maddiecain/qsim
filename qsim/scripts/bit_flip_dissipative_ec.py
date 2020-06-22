from qsim import master_equation
from qsim.tools import operations
from qsim import tools
from qsim.evolution.lindblad_operators import PauliNoise
from qsim.state import ETThreeQubitCode, ThreeQubitCode
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg

# Initial state is 1/sqrt(2)(|000>+|111>)
N = 2
#psi = ETThreeQubitCode.basis[0]
psi = (ETThreeQubitCode.basis[0] + ETThreeQubitCode.basis[1]) / np.sqrt(2)
psi = tools.outer_product(psi, psi)
psi = tools.tensor_product([psi]*N)
state = ETThreeQubitCode(psi, 1, is_ket=False)

edges = [[0, 1]]#, [1, 2], [0 ,2]]
ham = None

def hamiltonian_commutator(s):
    # Computes [H, s]
    # s is the input state
    # H = Z_L Z_L
    global ham
    return ham @ s - s @ ham

def hamiltonian_commutator_basic(s):
    # Computes [H, s]
    # s is the input state
    # H = Z_L Z_L
    global ham_basic
    return ham_basic @ s - s @ ham_basic


def lindbladian_corr(coefficient=1):
    # Bit flip on the third qubit
    c1 = 1 / 4 * tools.tensor_product([tools.identity(2), tools.X()]) @ (
            tools.identity(ETThreeQubitCode.n) + ETThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[1])
    # Bit flip on the first qubit
    c2 = 1 / 4 * tools.tensor_product([tools.X(), tools.identity(2)]) @ (
            tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ETThreeQubitCode.n) + ETThreeQubitCode.stabilizers[1])
    # Bit flip on the second qubit
    c3 = 1 / 4 * tools.tensor_product([tools.identity(1), tools.X(), tools.identity(1)]) @ (
            tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[1])
    return [c1, c2, c3]

def lindbladian_corr_extra(coefficient=1):
    # Bit flip on the third qubit
    global ham_basic_2local
    c1 = 1 / 4 * scipy.linalg.expm(-1j * 2 * ham_basic_2local/corr_rate) * tools.tensor_product([tools.identity(2), tools.X()]) @ (
            tools.identity(ETThreeQubitCode.n) + ETThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[1])
    # Bit flip on the first qubit
    c2 = 1 / 4 * scipy.linalg.expm(-1j * 2 * ham_basic_2local/corr_rate) * tools.tensor_product([tools.X(), tools.identity(2)]) @ (
            tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ETThreeQubitCode.n) + ETThreeQubitCode.stabilizers[1])
    # Bit flip on the second qubit
    c3 = 1 / 4 * scipy.linalg.expm(-1j * 2 * ham_basic_2local/corr_rate) * tools.tensor_product([tools.identity(1), tools.X(), tools.identity(1)]) @ (
            tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ETThreeQubitCode.n) - ETThreeQubitCode.stabilizers[1])
    return [c1, c2, c3]

def corr_all_qubit_liouvillian(s, povm, weights):
    def corr_liouvillian(s, i):
        a = np.zeros(s.shape)
        for j in range(len(povm)):
            a = a + weights[j] * (operations.single_qubit_operation(s, i, povm[j]) -
                                  1 / 2 * operations.left_multiply(s, i, povm[j].conj().T @ povm[j]) -
                                  1 / 2 * operations.right_multiply(s, i, povm[j].conj().T @ povm[j]))
        return a

    a = np.zeros(s.shape)
    for i in range(int(N)):
        a = a + corr_liouvillian(s, i)
    return a



# print(np.linalg.eig(corr_povm_basic))
lindbladX = PauliNoise([1, 0, 0])
t0 = 0
tf = 100
dt = .1
me = master_equation.MasterEquation(hamiltonian=None, noise_model=None)
#print('res_corr')
"""res_corr = me.run_ode_solver(state.state, t0, tf, dt,
                             func=lambda s, t: -1j * hamiltonian_commutator(
                                 s) + noise_rate * lindbladX.all_qubit_liouvillian(
                                 s)  +  corr_all_qubit_liouvillian(s, corr_povm_basic,
                                                                          [corr_rate_basic] * ETThreeQubitCode.n))
"""
print('res_corr_basic')
print('res_noisy')

"""res_noisy = me.run_ode_solver(state.state, t0, tf, dt,
                              func=lambda s, t: -1j * hamiltonian_commutator_basic(
                                  s) + noise_rate * lindbladX.all_qubit_liouvillian(s))
print('res_ideal')

"""

#fidelity_ideal = np.array([tools.trace(res_ideal[i] @ res_ideal[i]) for i in range(res_ideal.shape[0])])
#fidelity_noisy = np.array([tools.trace(res_ideal[j] @ res_noisy[j]) for j in range(res_ideal.shape[0])])
#fidelity_corr = np.array([tools.trace(res_ideal[k] @ res_corr[k]) for k in range(res_ideal.shape[0])])

ax = plt.figure(figsize=(6, 6))

for c in [.01, .1, 1, 2, 5, 10]:
    coefficient = c
    print(coefficient)
    hc = True
    if hc:
        ham = np.zeros(state.state.shape)
        for i in range(N):
            # Sum of Z_L's
            ham = ham + tools.tensor_product([tools.identity(i * ETThreeQubitCode.n), ETThreeQubitCode.Z,
                                              tools.identity(ETThreeQubitCode.n * (N - i - 1))])
        for j in range(len(edges)):
            # Sum of Z_L Z_L's
            ham = ham + tools.tensor_product(
                [tools.identity(ETThreeQubitCode.n * edges[j][0]), ETThreeQubitCode.Z,
                 tools.identity(ETThreeQubitCode.n * (edges[j][1] - edges[j][0] - 1)), ETThreeQubitCode.Z,
                 tools.identity(ETThreeQubitCode.n * (N - edges[j][1] - 1))])

        ham = scipy.sparse.diags(np.diagonal(ham * coefficient))
        ham_basic = np.zeros(state.state.shape)
        ham_basic_2local = np.zeros(state.state.shape)
        for i in range(N):
            # Sum of Z_L's
            ham_basic = ham_basic + tools.tensor_product([tools.identity(i * ThreeQubitCode.n), ThreeQubitCode.Z,
                                                          tools.identity(ThreeQubitCode.n * (N - i - 1))])
        for j in range(len(edges)):
            # Sum of Z_L Z_L's
            ham_basic = ham_basic + tools.tensor_product(
                [tools.identity(ThreeQubitCode.n * edges[j][0]), ThreeQubitCode.Z,
                 tools.identity(ThreeQubitCode.n * (edges[j][1] - edges[j][0] - 1)), ThreeQubitCode.Z,
                 tools.identity(ThreeQubitCode.n * (N - edges[j][1] - 1))])
            ham_basic_2local = ham_basic_2local + tools.tensor_product(
                [tools.identity(ThreeQubitCode.n * edges[j][0]), ThreeQubitCode.Z,
                 tools.identity(ThreeQubitCode.n * (edges[j][1] - edges[j][0] - 1)), ThreeQubitCode.Z,
                 tools.identity(ThreeQubitCode.n * (N - edges[j][1] - 1))])
        ham_basic = scipy.sparse.diags(np.diagonal(ham_basic * coefficient))
        ham_basic_2local = scipy.sparse.diags(np.diagonal(ham_basic * coefficient))

    if not hc:
        ham = np.zeros(state.state.shape)
        for i in range(N):
            # Sum of Z_L's
            ham = ham + tools.tensor_product([tools.identity(i * ETThreeQubitCode.n), ETThreeQubitCode.X,
                                              tools.identity(ETThreeQubitCode.n * (N - i - 1))])
        for j in range(len(edges)):
            # Sum of Z_L Z_L's
            ham = ham + tools.tensor_product(
                [tools.identity(ETThreeQubitCode.n * edges[j][0]), ETThreeQubitCode.X,
                 tools.identity(ETThreeQubitCode.n * (edges[j][1] - edges[j][0] - 1)), ETThreeQubitCode.X,
                 tools.identity(ETThreeQubitCode.n * (N - edges[j][1] - 1))])

        ham = scipy.sparse.diags(np.diagonal(ham * coefficient))
        ham_basic = np.zeros(state.state.shape)
        for i in range(N):
            # Sum of Z_L's
            ham_basic = ham_basic + tools.tensor_product(
                [tools.identity(i * ThreeQubitCode.n), ThreeQubitCode.X,
                 tools.identity(ThreeQubitCode.n * (N - i - 1))])
        for j in range(len(edges)):
            # Sum of Z_L Z_L's
            ham = ham + tools.tensor_product(
                [tools.identity(ThreeQubitCode.n * edges[j][0]), ThreeQubitCode.X,
                 tools.identity(ThreeQubitCode.n * (edges[j][1] - edges[j][0] - 1)), ThreeQubitCode.X,
                 tools.identity(ThreeQubitCode.n * (N - edges[j][1] - 1))])
        ham_basic = scipy.sparse.diags(np.diagonal(ham_basic * coefficient))
    corr_rate = 10
    corr_povm_extra = lindbladian_corr_extra(coefficient=coefficient)
    noise_rate = .1
    corr_povm = lindbladian_corr()
    corr_rate_basic = corr_rate
    res_ideal = me.run_ode_solver(state.state, t0, tf, dt,
                                  func=lambda s, t: -1j * hamiltonian_commutator(s))
    res_corr_basic = me.run_ode_solver(state.state, t0, tf, dt,
                                       func=lambda s, t: -1j * hamiltonian_commutator_basic(
                                           s) + noise_rate * lindbladX.all_qubit_liouvillian(
                                           s) + corr_all_qubit_liouvillian(s, corr_povm_extra,
                                                                           [corr_rate_basic] * ThreeQubitCode.n))

    fidelity_corr_basic = np.array([tools.trace(res_ideal[l] @ res_corr_basic[l]) for l in range(res_ideal.shape[0])])
    plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_corr_basic, label='Hamiltonian strength $c=$'+str(c))

#print('Noisy', fidelity_noisy)
print('Basic', fidelity_corr_basic)
#print('Hamiltonian', fidelity_corr)

plt.title(
    r'Bit flip code with Pauli $\Gamma_X=$' + str(noise_rate) + ', $\Gamma_{corr}=$' + str(
        corr_rate)+', $H = c(Z_{L, 1}+Z_{L, 2}+Z_{L, 1}Z_{L, 2})$')# '$(X_{L, 1}+X_{L, 2})$')#+
plt.xlabel(r'$t$')
plt.ylabel(r'tr$\left(\rho_{ideal}\rho\right)$')
plt.xlabel(r'$t$')

#plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_ideal, label='Noiseless')
#plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_noisy, label='No correction')
#plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_corr, label='Correction with Hamiltonian')
plt.legend(loc='lower right')
plt.savefig('dec_' + str(noise_rate) + '_' + str(corr_rate) + '_' + str(coefficient) + '_' + str(tf) + '.pdf',
            format='pdf')
plt.show()
