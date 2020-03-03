import numpy as np
from qsim.noise import noise_models
from qsim.qaoa import variational_parameters
from qsim import tools
from qsim.state import *
import matplotlib.pyplot as plt
import scipy.optimize


def rx_circuit(s):
    # Apply a X-gate as a rotation over time
    s.all_qubit_rotation(np.pi / 2, s.X)


def ry_circuit(s):
    # Apply a Y-gate as a rotation over time
    s.all_qubit_rotation(np.pi / 2, s.Y)


def plot_rx(k=500):
    def random_angle(scale=.001):
        m = np.random.rand(3)
        m = m / (np.linalg.norm(m))
        theta = np.random.normal(scale=scale)
        return m, theta

    def run_rx(param):
        N = 1
        # Initialize in state |1_L>
        psi = JordanFarhiShor.basis[0]
        s = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        penalty = variational_parameters.HamiltonianBookatzPenalty()
        rx_circuit(s)
        s_ideal = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        rx_circuit(s_ideal)
        for q in range(s.N):
            noise_models.random_pauli_noise(s, i, m, theta)
        # random_pauli_noise(s_ideal, m, theta)
        penalty.evolve_penalty(s, param)
        return -1 * np.real(tools.trace(s.state @ s_ideal.state))

    thetas = []
    alphas = []
    vals = []
    for i in range(k):
        m, theta = random_angle(scale=.05)

        res = scipy.optimize.brute(run_rx, np.array([(0, 2 * np.pi)]), full_output=True)
        # print('m, theta:', m, theta)
        # print('f_val:', np.real(res[1])/-1)
        # print('params:', np.array(res[0]))
        thetas.append(theta)
        alphas.append(res[0])
        vals.append(np.real(res[1] / -1))

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Theta versus fidelity', fontsize=14)

    plt.scatter(thetas, vals)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Theta versus alpha', fontsize=14)

    plt.scatter(thetas, alphas)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Alpha versus fidelity', fontsize=14)

    plt.scatter(alphas, vals)

    plt.show()


def plot_rx_rx(k=20):
    def random_angle(scale=.001):
        m = np.random.rand(3)
        m = m / (np.linalg.norm(m))
        theta = np.random.normal(scale=scale)
        return m, theta

    def run_rx_ry(param):
        N = 1
        # Initialize in state |1_L>
        psi = JordanFarhiShor.basis[0]
        s = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        penalty = variational_parameters.HamiltonianBookatzPenalty()
        rx_circuit(s)
        s_ideal = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        rx_circuit(s_ideal)
        rx_circuit(s_ideal)
        for q in range(s.N):
            noise_models.random_pauli_noise(s, i, m1, theta1)
        penalty.evolve_penalty(s, param[0])
        for q in range(s.N):
            noise_models.random_pauli_noise(s, i, m2, theta2)
        rx_circuit(s)
        penalty.evolve_penalty(s, param[1])

        return -1 * np.real(tools.trace(s.state @ s_ideal.state))

    thetas1 = []
    thetas2 = []
    alphas1 = []
    alphas2 = []
    vals = []
    for i in range(k):
        m1, theta1 = random_angle(scale=.05)
        m2, theta2 = random_angle(scale=.05)

        res = scipy.optimize.brute(run_rx_ry, np.array([(0, np.pi), (0, np.pi)]), full_output=True)
        # print('m, theta:', m, theta)
        # print('f_val:', np.real(res[1])/-1)
        # print('params:', np.array(res[0]))
        thetas1.append(theta1)
        thetas2.append(theta2)
        alphas1.append(np.real(res[0][0]) % 2 * np.pi)
        alphas2.append(np.real(res[0][1]) % 2 * np.pi)
        vals.append(np.real(res[1]) / -1)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Alpha versus vals', fontsize=14)
    ax[0].set_title('Alpha_1 versus vals')
    ax[1].set_title('Alpha_2 versus vals')
    ax[0].scatter(alphas1, vals)
    ax[1].scatter(alphas2, vals)

    plt.show()


def plot_rx_error(k=100):
    def random_angle(scale=.001):
        m = np.random.rand(3)
        m = m / (np.linalg.norm(m))
        theta = np.random.normal(scale=scale)
        return m, theta

    def run_rx(param):
        N = 1
        # Initialize in state |1_L>
        psi = JordanFarhiShor.basis[0]
        s = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        penalty = variational_parameters.HamiltonianBookatzPenalty()
        rx_circuit(s)
        s_ideal = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        rx_circuit(s_ideal)
        for q in range(s.N):
            noise_models.random_pauli_noise(s, q, m, theta)
        # random_pauli_noise(s_ideal, m, theta)
        # random_pauli_noise(s_ideal, m, theta)
        penalty.evolve_penalty(s, param)
        return -1 * np.real(tools.trace(s.state @ s_ideal.state))

    thetas = []
    alphas = []
    vals = []
    for i in range(k):
        m, theta = random_angle(scale=.05)

        res = scipy.optimize.brute(run_rx, [(0, np.pi)], full_output=True)
        # print('m, theta:', m, theta)
        # print('f_val:', np.real(res[1])/-1)
        # print('params:', np.array(res[0]))
        thetas.append(theta)
        alphas.append(res[0][0])
        vals.append(np.real(res[1] / -1))
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Theta versus fidelity', fontsize=14)

    plt.scatter(thetas, vals)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Theta versus alpha', fontsize=14)

    plt.scatter(thetas, alphas)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Alpha versus fidelity', fontsize=14)

    plt.scatter(alphas, vals)

    plt.show()


def plot_2_alpha(k=20):
    def random_angle(scale=.001):
        m = np.random.rand(3)
        m = m / (np.linalg.norm(m))
        theta = np.random.normal(scale=scale)
        return m, theta

    def run_rx_ry(param):
        N = 1
        # Initialize in state |1_L>
        psi = JordanFarhiShor.basis[0]
        s = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        penalty = variational_parameters.HamiltonianPenalty()
        rx_circuit(s)
        s_ideal = JordanFarhiShor(tools.outer_product(psi, psi), N, is_ket=False)
        # Apply circuit
        rx_circuit(s_ideal)
        rx_circuit(s_ideal)
        for q in range(s.N):
            noise_models.random_pauli_noise(s, q, m1, theta1)
        penalty.evolve_penalty(s, param[0])
        for q in range(s.N):
            noise_models.random_pauli_noise(s, q, m1, theta1)
        rx_circuit(s)
        penalty.evolve_penalty(s, param[1])

        return -1 * np.real(tools.trace(s.state @ s_ideal.state))

    alphas1 = np.linspace(0, np.pi, k)
    alphas2 = np.linspace(0, np.pi, k)
    vals = []
    m1, theta1 = random_angle(scale=.05)
    for j in range(k):
        vals.append(run_rx_ry([alphas1[j], alphas2[j]]) / -1)
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Alpha versus vals', fontsize=14)
    ax.set_title('Alpha_1 versus vals')
    ax.scatter(alphas1, vals)

    plt.show()


# plot_2_alpha()
plot_rx_rx()

