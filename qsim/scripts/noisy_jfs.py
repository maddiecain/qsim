import numpy as np
from qsim import noise_models
from qsim.qaoa import simulate, variational_parameters
from qsim import tools, operations
from qsim.state import *
import matplotlib.pyplot as plt


def random_pauli_noise(s, scale=.01):
    m = np.random.rand(3)
    m = m/(np.linalg.norm(m))**2
    theta = np.random.normal(scale=.25)
    pauli = m[0]*State.SX+m[1]*State.SY+m[2]*State.SZ
    for p in range(s.N):
        s.state = operations.single_qubit_rotation(s.state, p, theta, pauli, is_ket=s.is_ket)

N = 1
# Initialize in state |1_L>
#ideal = s.single_qubit_operation(0, s.SX)
# Numerically integrate the master equation

ep = 10
penalty = variational_parameters.HamiltonianPenalty()
depolarize = noise_models.DepolarizingNoise(0.01)
amplitude = noise_models.AmplitudeDampingNoise(0.01)
#pauli = noise_models.PauliNoise(0.01)

n = 500
dt = np.pi/2/n
p_len = 10
rate = np.linspace(1, 10, p_len)
depolarize_results = np.zeros((p_len, n))
amplitude_results = np.zeros((p_len, n))


for j in [depolarize, amplitude]:
    for k in range(p_len):
        psi = JordanFarhiShor.basis[0]
        s = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        s_good = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        j.p = rate[k]*dt
        for i in range(n):
            # Evolve density matrix
            j.all_qubit_channel(s)
            s.all_qubit_rotation(dt, s.SX)
            s_good.all_qubit_rotation(dt, s.SX)
            penalty.evolve_penalty(s, dt*ep)
            # Compute fidelity
            if j == depolarize:
                depolarize_results[k,i] = np.real(tools.trace(s.state @ s_good.state))
            if j == amplitude:
                amplitude_results[k,i] = np.real(tools.trace(s.state @  s_good.state))


"""for j in range(q):
    for k in range(p_len):
        psi = JordanFarhiShor.basis[0]
        s = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        s_good = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        for i in range(n):
            # Apply noise
            random_pauli_noise(s, rate[k]*dt)
            # Compute fidelity
            results[k,i] = np.real(tools.trace(s.state @ s_good.state))
"""
# Plots for depolarizing, amplitude damping, and pauli channels
# Error probability versus final fidelity
# No rotation, just penalty
# SX, SY, SZ rotations
# As a function of initial state
# Trotterized version

fig, ax = plt.subplots(1, 2, sharey=True)
fig.suptitle('Bit flip of |0_L> under noise with penalty', fontsize=14)
ax[0].set_title('Depolarizing channel')
ax[1].set_title('Amplitude damping channel')
ax[0].set_ylabel('Overlap with noiseless simulation')
ax[0].set_xlabel('Time')
ax[1].set_xlabel('Time')
t = np.linspace(0, np.pi/2, n)
for l in range(p_len):
    ax[0].plot(t, depolarize_results[l])
    ax[1].plot(t, amplitude_results[l], label = round(rate[l],2))
plt.legend(loc='upper right')

plt.show()
