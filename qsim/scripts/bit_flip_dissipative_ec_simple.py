from qsim import master_equation
from qsim.tools import operations
from qsim import tools
from qsim.noise.noise_models import PauliNoise
from qsim.state import ThreeQubitCode
import numpy as np
import matplotlib.pyplot as plt

# New ThreeQubitCode object with two logical qubits
# Initial state is 1/sqrt(2)(|000>+|111>)
N = 1
psi = (ThreeQubitCode.basis[0] + ThreeQubitCode.basis[1]) / np.sqrt(2)
psi = tools.outer_product(psi, psi)
psi = tools.tensor_product([psi]*N)
state = ThreeQubitCode(psi, 1, is_ket=False)

ham_corr = np.zeros((8, 8))
ham_corr[1, 1] = 2
ham_corr[2, 2] = 2
ham_corr[3, 3] = -2
ham_corr[4, 4] = 2
ham_corr[5, 5] = -2
ham_corr[6, 6] = -2
coefficient = 1
ham_corr = ham_corr * coefficient
def hamiltonian_correction(s, coefficient=1):
    global ham_corr
    s1 = np.zeros(s.shape)
    s2 = np.zeros(s.shape)
    for i in range(N):
        # Sum of Z_L's
        s1 = s1 + operations.left_multiply(s, i, ham_corr)
        s2 = s2 + operations.right_multiply(s, i, ham_corr)
    return s1 - s2


def hamiltonian_commutator(s, coefficient=1):
    # Computes [H, s]
    # s is the input state
    # H = Z_L Z_L
    s1 = np.zeros(s.shape)
    s2 = np.zeros(s.shape)
    for i in range(N):
        # Sum of Z_L's
        s1 = s1 + operations.left_multiply(s, i, ThreeQubitCode.Z)
        s2 = s2 + operations.right_multiply(s, i, ThreeQubitCode.Z)
    return coefficient * (s1 - s2)


def lindbladian_corr():
    # Bit flip on the third qubit
    c1 = 1 / 4 * tools.tensor_product([tools.identity(2), tools.X()]) @ (
            tools.identity(ThreeQubitCode.n) + ThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ThreeQubitCode.n) - ThreeQubitCode.stabilizers[1])
    # Bit flip on the first qubit
    c2 = 1 / 4 * tools.tensor_product([tools.X(), tools.identity(2)]) @ (
            tools.identity(ThreeQubitCode.n) - ThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ThreeQubitCode.n) + ThreeQubitCode.stabilizers[1])
    # Bit flip on the second qubit
    c3 = 1 / 4 * tools.tensor_product([tools.identity(1), tools.X(), tools.identity(1)]) @ (
            tools.identity(ThreeQubitCode.n) - ThreeQubitCode.stabilizers[0]) @ (
                 tools.identity(ThreeQubitCode.n) - ThreeQubitCode.stabilizers[1])
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
    for i in range(int(1)):
        a = a + corr_liouvillian(s, i)
    return a


corr_rate = 100
corr_povm_basic = lindbladian_corr()
noise_rate = .1
corr_povm = lindbladian_corr(rate=corr_rate + noise_rate, coefficient=coefficient)
corr_rate_basic = corr_rate
# print(np.linalg.eig(corr_povm_basic))
lindbladX = PauliNoise([1, 0, 0])
t0 = 0
tf = 100
dt = 1
me = master_equation.MasterEquation(hamiltonian=None, noise_model=None)
print('res_corr')
res_corr = me.run_ode_solver(state.state, t0, tf, dt,
                             func=lambda s, t: -1j * hamiltonian_commutator(
                                 s, coefficient=coefficient) + noise_rate * lindbladX.all_qubit_liouvillian(
                                 s) - 1j * hamiltonian_correction(s, coefficient=coefficient) + \
                                               corr_all_qubit_liouvillian(s, corr_povm_basic,
                                                                          [corr_rate_basic] * ThreeQubitCode.n))
print('res_corr_basic')

res_corr_basic = me.run_ode_solver(state.state, t0, tf, dt,
                                   func=lambda s, t: -1j * hamiltonian_commutator(
                                       s, coefficient=coefficient) + noise_rate * lindbladX.all_qubit_liouvillian(
                                       s) + corr_all_qubit_liouvillian(s, corr_povm_basic,
                                                                       [corr_rate_basic] * ThreeQubitCode.n))
print('res_noisy')

res_noisy = me.run_ode_solver(state.state, t0, tf, dt,
                              func=lambda s, t: -1j * hamiltonian_commutator(
                                  s, coefficient=coefficient) + noise_rate * lindbladX.all_qubit_liouvillian(s))
print('res_ideal')

res_ideal = me.run_ode_solver(state.state, t0, tf, dt,
                              func=lambda s, t: -1j * hamiltonian_commutator(s, coefficient=coefficient))

fidelity_ideal = np.array([tools.trace(res_ideal[i] @ res_ideal[i]) for i in range(res_ideal.shape[0])])
fidelity_noisy = np.array([tools.trace(res_ideal[j] @ res_noisy[j]) for j in range(res_ideal.shape[0])])
fidelity_corr = np.array([tools.trace(res_ideal[k] @ res_corr[k]) for k in range(res_ideal.shape[0])])
fidelity_corr_basic = np.array([tools.trace(res_ideal[l] @ res_corr_basic[l]) for l in range(res_ideal.shape[0])])
print('Noisy', fidelity_noisy)
print('Basic', fidelity_corr_basic)
print('Hamiltonian', fidelity_corr)
ax = plt.figure(figsize=(6, 6))
plt.title(
    r'Bit flip code with Pauli $\Gamma_X=$' + str(noise_rate) + ', $\Gamma_{corr}=$' + str(
        corr_rate) + ', $H = $' + str(coefficient) + '$Z_L$')
plt.xlabel(r'$t$')
plt.ylabel(r'tr$\left(\rho_{ideal}\rho\right)$')
plt.xlabel(r'$t$')
plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_ideal, label='Noiseless')
plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_noisy, label='No correction')
plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_corr, label='Correction with Hamiltonian')
plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity_corr_basic, label='Correction no Hamiltonian')
plt.legend(loc='lower left')
plt.savefig('dec_' + str(noise_rate) + '_' + str(corr_rate) + '_' + str(coefficient) + '_' + str(tf) + '.pdf',
            format='pdf')
plt.show()
