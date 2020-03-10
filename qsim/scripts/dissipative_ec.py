import numpy as np
from qsim.noise import noise_models
from qsim import tools
from qsim.state import *
import matplotlib.pyplot as plt
from qsim.hamiltonian import *


def ham_term(*argv):
    def decode(code):
        if code == 'i':
            return np.identity(2)
        if code == 'x':
            return tools.X()
        if code == 'y':
            return tools.Y()
        if code == 'z':
            return tools.Z()

    return tools.tensor_product([decode(arg) for arg in argv])


def h1():
    ham = (tools.identity(5) + ham_term('z', 'z', 'i', 'i', 'i')) @ (
                ham_term('i', 'z', 'z', 'i', 'i') - tools.identity(5)) @ \
          ham_term('i', 'i', 'x', 'i', 'x') + (tools.identity(5) + ham_term('z', 'i', 'z', 'i', 'i')) @ (
                      ham_term('i', 'z', 'z', 'i', 'i') - tools.identity(5)) @ \
          ham_term('i', 'x', 'i', 'x', 'x') + (tools.identity(5) + ham_term('i', 'z', 'z', 'i', 'i')) @ (
                      ham_term('z', 'z', 'i', 'i', 'i') - tools.identity(5)) @ \
          ham_term('x', 'i', 'i', 'x', 'i')
    return ham + ham.conj().T


def h2():
    hd = ham_term('i', 'i', 'i', 'i', 'x') + ham_term('i', 'i', 'i', 'x', 'i') + ham_term('i', 'i', 'i', 'x', 'x') + \
         ham_term('i', 'i', 'i', 'x', 'z') - ham_term('i', 'i', 'i', 'y', 'y') + ham_term('i', 'i', 'i', 'z', 'x') - \
         ham_term('i', 'z', 'z', 'i', 'x') + ham_term('i', 'z', 'z', 'x', 'i') - ham_term('i', 'z', 'z', 'x', 'x') + \
         ham_term('i', 'z', 'z', 'x', 'z') + ham_term('i', 'z', 'z', 'y', 'y') - ham_term('i', 'z', 'z', 'z', 'x') - \
         ham_term('z', 'i', 'z', 'i', 'x') - ham_term('z', 'i', 'z', 'x', 'i') + ham_term('z', 'i', 'z', 'x', 'x') - \
         ham_term('z', 'i', 'z', 'x', 'z') - ham_term('z', 'i', 'z', 'y', 'y') - ham_term('z', 'i', 'z', 'z', 'x') + \
         ham_term('z', 'z', 'i', 'i', 'x') - ham_term('z', 'z', 'i', 'x', 'i') - ham_term('z', 'z', 'i', 'x', 'x') - \
         ham_term('z', 'z', 'i', 'x', 'z') + ham_term('z', 'z', 'i', 'y', 'y') + ham_term('z', 'z', 'i', 'z', 'x')
    hc = ham_term('i', 'i', 'x', 'i', 'i') + ham_term('i', 'x', 'i', 'i', 'i') + ham_term('x', 'i', 'i', 'i', 'i') + \
         ham_term('x', 'z', 'z', 'i', 'i') + ham_term('z', 'x', 'z', 'i', 'i') + ham_term('z', 'z', 'x', 'i', 'i') - \
         ham_term('i', 'i', 'x', 'i', 'z') - ham_term('i', 'x', 'i', 'i', 'z') + ham_term('x', 'i', 'i', 'i', 'z') + \
         ham_term('x', 'z', 'z', 'i', 'z') - ham_term('z', 'x', 'z', 'i', 'z') - ham_term('z', 'z', 'x', 'i', 'z') + \
         ham_term('i', 'i', 'x', 'z', 'i') - ham_term('i', 'x', 'i', 'z', 'i') - ham_term('x', 'i', 'i', 'z', 'i') - \
         ham_term('x', 'z', 'z', 'z', 'i') - ham_term('z', 'x', 'z', 'z', 'i') + ham_term('z', 'z', 'x', 'z', 'i') - \
         ham_term('i', 'i', 'x', 'z', 'z') + ham_term('i', 'x', 'i', 'z', 'z') - ham_term('x', 'i', 'i', 'z', 'z') - \
         ham_term('x', 'z', 'z', 'z', 'z') + ham_term('x', 'z', 'z', 'z', 'z') + ham_term('z', 'x', 'z', 'z', 'z') - \
         ham_term('z', 'z', 'x', 'z', 'z')
    return hd + hc + 1j * (hd @ hc - hc @ hd)


def h3():
    return -1 * ham_term('z', 'i', 'i', 'x', 'i') + ham_term('i', 'z', 'i', 'x', 'i') - ham_term('i', 'z', 'i', 'y', 'i') + \
           ham_term('i', 'i', 'z', 'y', 'i')

def h4():
    ham = (tools.identity(5) + ham_term('z', 'z', 'i', 'i', 'i')) @ (
                ham_term('i', 'z', 'z', 'i', 'i') - tools.identity(5)) @ \
          ham_term('i', 'i', 'i', 'i', 'x') + (tools.identity(5) + ham_term('z', 'i', 'z', 'i', 'i')) @ (
                      ham_term('i', 'z', 'z', 'i', 'i') - tools.identity(5)) @ \
          ham_term('i', 'i', 'i', 'x', 'x') + (tools.identity(5) + ham_term('i', 'z', 'z', 'i', 'i')) @ (
                      ham_term('z', 'z', 'i', 'i', 'i') - tools.identity(5)) @ \
          ham_term('i', 'i', 'i', 'x', 'i')
    return ham + ham.conj().T


hamiltonian = Hamiltonian(h2())
# h2 params: p_len = 5, kappa = 1, n = 100, rate = np.linspace(4, 100, p_len)
# Number of qubits
N = 1
# Numerically integrates the master equation

# Number of time steps
n = 100
dt = 1 / n
p_len = 10
# Hamiltonian strength
kappa = 1  # np.linspace(0, 10, 10)
# Bit flip rate
p_error = 1 * dt
bit_flip = noise_models.PauliNoise((p_error, 0, 0))
# Dissipative noise rate
rate = np.linspace(0, n, p_len)
corrective = noise_models.AmplitudeDampingNoise(1)

noisy_results = np.zeros((p_len, n))
ec_results = np.zeros((p_len, n))
for j in range(p_len):
    ideal = ThreeQubitCodeTwoAncillas.basis[0]
    # Noise with no error correction
    noisy = ThreeQubitCodeTwoAncillas(tools.outer_product(ideal, ideal), N, is_ket=False)
    # Dissipative error correction
    ec = ThreeQubitCodeTwoAncillas(tools.outer_product(ideal, ideal), N, is_ket=False)
    # Ideal state for fidelity calculations
    ideal = ThreeQubitCodeTwoAncillas(tools.outer_product(ideal, ideal), N, is_ket=False)
    corrective.p = rate[j] * dt
    for i in range(n):
        # Evolve density matrix
        # Apply bit flip noise
        hamiltonian.evolve(ec, dt * kappa)
        for l in range(3):
            noisy.state = bit_flip.channel(noisy.state, l)
            ec.state = bit_flip.channel(ec.state, l)
        for l in range(3, 5):
            ec.state = corrective.channel(ec.state, l)
        # Compute fidelity
        noisy_results[j, i] = np.real(tools.trace(noisy.state @ ideal.state, ind = (4, 3)))
        ec_results[j, i] = np.real(tools.trace(ec.state @ ideal.state, ind = (4, 3)))

fig, ax = plt.subplots(1, 1)
fig.suptitle('Dissipative Error Correction on Bit Flip Code', fontsize=14)
ax.set_ylabel('Fidelity')
ax.set_xlabel('Time')
t = np.linspace(0, 1, n)
for l in range(p_len):
    ax.plot(t, ec_results[l], label=rate[l] * dt)
ax.plot(t, noisy_results[0], color='k', label='No Hamiltonian')

plt.legend(loc='upper right', title='Amplitude Damping Rate')

plt.show()
