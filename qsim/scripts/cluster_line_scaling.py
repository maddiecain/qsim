import sys
import os

sys.path.append(os.getcwd())
os.system("source activate qsim")

import matplotlib.pyplot as plt
import networkx as nx

from qsim.codes import rydberg
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim.graph_algorithms.graph import Graph, line_graph

Gamma = 0
Delta = 1
Omega_r = 1
Omega_g = 1
delta = 100
Omega = 1 / 100


def eit_simulation(graph, noise_model=None, show_graph=False, gamma=1, delta: float = 0):
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    if noise_model == 'continuous':
        laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg)
        laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg)
        rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
        if delta != 0:
            detuning = hamiltonian.HamiltonianEnergyShift(code=rydberg, IS_subspace=True, graph=graph,
                                                          energies=[delta])
        spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2), rates=[gamma],
                                                                      IS_subspace=True, code=rydberg)

        # Initialize adiabatic algorithm
        if delta != 0:
            simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='continuous', code=rydberg, noise=[spontaneous_emission])
        else:
            simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='continuous', code=rydberg, noise=[spontaneous_emission])
        return simulation

    elif noise_model == 'monte_carlo':
        laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg)
        laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg)
        rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
        spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2), rates=[gamma],
                                                                      IS_subspace=True, code=rydberg)

        # Initialize adiabatic algorithm
        if delta != 0:
            detuning = hamiltonian.HamiltonianEnergyShift(code=rydberg, IS_subspace=True, graph=graph, energies=[delta])
            simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2, detuning],
                                       cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                       noise_model='monte_carlo', code=rydberg, noise=[spontaneous_emission])
        else:
            simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='monte_carlo', code=rydberg, noise=[spontaneous_emission])
        return simulation


def run(n, t, gamma, omega):
    graph, mis = line_graph(n, return_mis=True)
    graph = Graph(graph)

    def rydberg_EIT_schedule(t, tf, coefficients=None):
        if coefficients is None:
            coefficients = [1, 1]
        for i in range(len(simulation_eit.hamiltonian)):
            if i == 0:
                # We don't want to update normal detunings
                simulation_eit.hamiltonian[i].energies = [t / tf * coefficients[0]]
            if i == 1:
                simulation_eit.hamiltonian[i].energies = [(tf - t) / tf * coefficients[1]]

        return True

    simulation_eit = eit_simulation(graph, noise_model='monte_carlo', gamma=gamma, delta=0)
    res = simulation_eit.ratio_vs_total_time([t], schedule=lambda t, tf: rydberg_EIT_schedule(t, tf,
                                                                                              coefficients=[omega,
                                                                                                            omega]),
                                             plot=False, verbose=True, method='odeint', iter=1)
    return res



if __name__ == "__main__":
    index = int(sys.argv[1])
    # Convert index to a delta and omega
    # Line graphs 5-15
    # Times 25-250
    g = 10
    o = 10
    t = (index % 10 + 1) * 25
    number_of_nodes = int((index - index % 10) / 10) + 5
    print('running', t, number_of_nodes, flush=True)
    res = run(number_of_nodes, t, g, o)
    print(t, g, o, res, flush=True)
