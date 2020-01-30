import numpy as np
from qsim import noise_models
from qsim import tools
from qsim.state import *
import matplotlib.pyplot as plt
from qsim.qaoa import simulate, variational_parameters
import networkx as nx


# h2 params: p_len = 5, kappa = 1, n = 100, rate = np.linspace(4, 100, p_len)
# Number of qubits

G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1

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
ep = np.linspace(0, 50, p_len)

noisy_results = np.zeros((p_len, 2*n))
ec_results = np.zeros((p_len, 2*n))

# Initiate simulation
sim_penalty = simulate.SimulateQAOA(G, 1, 2, is_ket=False, code=TwoQubitCode)

hp = variational_parameters.HamiltonianPenalty()
hc = variational_parameters.HamiltonianC(sim_penalty.C)
hb = variational_parameters.HamiltonianB()

sim_penalty.variational_params = [hc, hb]
sim_penalty.noise = [noise_models.PauliNoise((.025, 0, 0)), noise_models.PauliNoise((.025, 0, 0))]

# Solve for optimal parameters in the absence of penalty Hamiltonian
res = sim_penalty.find_parameters_brute(n=5)
for j in range(p_len):
    ideal = np.sqrt(1/2)*(TwoQubitCode.basis[0]+TwoQubitCode.basis[0])
    # Noise with no error correction
    noisy = TwoQubitCode(tools.outer_product(ideal, ideal), N, is_ket=False)
    # Dissipative error correction
    ec = TwoQubitCode(tools.outer_product(ideal, ideal), N, is_ket=False)
    # Ideal state for fidelity calculations
    ideal = ThreeQubitCodeTwoAncillas(tools.outer_product(ideal, ideal), N, is_ket=False)
    rate = 1/ep[j] * dt
    for (l, k) in enumerate([hb, hc], 0):
        for i in range(n):
            # Evolve density matrix
            # Apply bit flip noise
            k.evolve(noisy, dt * res[0][l])
            k.evolve(ideal, dt * res[0][l])
            hp.evolve(ec, ep * dt)
            noisy.state = bit_flip.all_qubit_channel(rate)
            ec.state = bit_flip.all_qubit_channel(rate)
            # Compute fidelity
            noisy_results[j, i] = np.real(tools.trace(noisy.state @ ideal.state, ind = (4, 3)))
            ec_results[j, i] = np.real(tools.trace(ec.state @ ideal.state, ind = (4, 3)))

fig, ax = plt.subplots(1, 1)
fig.suptitle('Dissipative Error Correction on Bit Flip Code', fontsize=14)
ax.set_ylabel('Fidelity')
ax.set_xlabel('Normalized Time')
t = np.linspace(0, 1, 2*n)
for l in range(p_len):
    ax.plot(t, ec_results[l], label=rate[l] * dt)
ax.plot(t, noisy_results[0], color='k', label='None')

plt.legend(loc='upper right', title='Dissipative EC Rate')

plt.show()
