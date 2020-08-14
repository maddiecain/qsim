import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.codes import rydberg, qubit
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
import numpy as np
import time
import matplotlib.cm as cm
from qsim import tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult

from qsim.tools.tools import outer_product
from qsim.codes import qubit
from qsim.evolution.hamiltonian import HamiltonianMIS, HamiltonianDriver, HamiltonianMaxCut
from qsim.graph_algorithms.graph import Graph
from qsim.codes.quantum_state import State
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.lindblad_master_equation import LindbladMasterEquation

import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import IS_projector, Graph, line_graph, degree_fails_graph


def adiabatic_spectrum_line(plot=True, num=50, n=3):
    def adiabatic_hamiltonian(t, tf):
        graph, mis = line_graph(n=n)
        ham = np.zeros((2 ** n, 2 ** n))
        coefficients = adiabatic_schedule(t, tf)
        rydberg_energy = 50
        rydberg_hamiltonian = hamiltonian.HamiltonianMIS(graph, energy=rydberg_energy, code=qubit)
        # TODO: generalize this!
        if n == 3:
            ham = ham + coefficients[1] * tools.tensor_product([qubit.Z, np.identity(2), np.identity(2)])
            ham = ham + coefficients[1] * tools.tensor_product([np.identity(2), qubit.Z, np.identity(2)])
            ham = ham + coefficients[1] * tools.tensor_product([np.identity(2), np.identity(2), qubit.Z])
            ham = ham + coefficients[0] * tools.tensor_product([qubit.X, np.identity(2), np.identity(2)])
            ham = ham + coefficients[0] * tools.tensor_product([np.identity(2), qubit.X, np.identity(2)])
            ham = ham + coefficients[0] * tools.tensor_product([np.identity(2), np.identity(2), qubit.X])
            ham = ham + np.diag(rydberg_hamiltonian.hamiltonian.T[0])
        elif n == 2:
            ham = ham + coefficients[1] * tools.tensor_product([qubit.Z, np.identity(2)])
            ham = ham + coefficients[1] * tools.tensor_product([np.identity(2), qubit.Z])
            ham = ham + coefficients[0] * tools.tensor_product([qubit.X, np.identity(2)])
            ham = ham + coefficients[0] * tools.tensor_product([np.identity(2), qubit.X])
            ham = ham + np.diag(rydberg_hamiltonian.hamiltonian.T[0])
        return np.linalg.eig(ham)

    times = np.linspace(0, 1, num=num)
    eigvals = np.zeros((len(times), 2 ** n))
    for i in range(len(times)):
        eigval, eigvec = adiabatic_hamiltonian(times[i], 1)
        eigvals[i] = eigval.real
        if plot:
            for j in range(2 ** n):
                plt.scatter(times[i], eigvals[i, j], c='k', s=4)
    if plot:
        plt.show()


def delta_vs_T():
    graph, mis = degree_fails_graph(return_mis=True)
    graph = Graph(graph)
    graph.mis_size = mis
    times = np.concatenate([np.arange(7, 11, 1), np.arange(20, 110, 10)])
    detunings = np.arange(3, 27, 3)
    print(times, detunings)
    cost_function = []
    for i in range(len(times)):
        cost_function_detuning = []
        for j in range(len(detunings)):
            simulation = adiabatic_simulation(graph, noise_model='continuous', delta=detunings[j])
            results, info = simulation.run(times[i], schedule=lambda t, tf: simulation.rydberg_MIS_schedule(t, tf,
                                                                                                            coefficients=[
                                                                                                                np.sqrt(
                                                                                                                    detunings[
                                                                                                                        j]),
                                                                                                                1]),
                                           method='RK23')
            cost = simulation.cost_hamiltonian.cost_function(results[-1]) / simulation.graph.mis_size
            print(times[i], detunings[j], cost)
            cost_function_detuning.append(cost)
        cost_function.append(cost_function_detuning)

    plt.imshow(cost_function, vmin=0, vmax=1, interpolation=None, extent=[0, max(detunings), max(times), 0],
               origin='upper')
    plt.xticks(detunings)
    plt.yticks(times)
    plt.colorbar()
    plt.xlabel(r'Detuning $\delta$')
    plt.ylabel(r'Annealing time $T$')
    plt.show()


def omega_vs_T(graph=None, mis=None, show_graph=False, n=3):
    schedule = lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]
    if graph is None:
        graph, mis = line_graph(n=n)
    graph = Graph(graph)
    if show_graph:
        nx.draw(graph)
        plt.show()
    times = np.arange(20, 100, 10)
    omegas = np.arange(5, 8, .25)
    print(times, omegas)
    cost_function = []
    for i in range(len(times)):
        cost_function_detuning = []
        for j in range(len(omegas)):
            simulation = adiabatic_simulation(graph, noise_model='continuous', delta=omegas[j])
            results = master_equation.run_ode_solver(psi, 0, times[i], num_from_time(times[i]),
                                                     schedule=lambda t: schedule(t, times[i]))
            cost = rydberg_hamiltonian_cost.cost_function(results[-1], is_ket=False) / mis
            print(times[i], omegas[j], cost)
            cost_function_detuning.append(cost)
        cost_function.append(cost_function_detuning)

    plt.imshow(cost_function, vmin=0, vmax=1, interpolation=None, extent=[0, max(omegas), max(times), 0],
               origin='upper')
    plt.xticks(np.linspace(0, max(omegas), 10))
    plt.yticks(np.linspace(0, max(times), 10))
    plt.colorbar()
    plt.xlabel(r'Rabi frequency $\Omega$')
    plt.ylabel(r'Annealing time $T$')
    plt.show()


def ARvstime_EIT(tf=10, graph=None, mis=None, show_graph=False, n=3):
    schedule = lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]
    if graph is None:
        graph, mis = line_graph(n=n)
    graph = Graph(graph)
    if show_graph:
        nx.draw(graph)
        plt.show()
    rabi1 = 3
    rabi2 = 3
    # Generate the driving
    laser1 = hamiltonian.HamiltonianDriver(transition=(0, 1), energy=rabi1, code=rydberg, IS_subspace=True, graph=graph)
    laser2 = hamiltonian.HamiltonianDriver(transition=(1, 2), energy=rabi2, code=rydberg, IS_subspace=True, graph=graph)
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, code=rydberg, detuning=1, energy=0,
                                                          IS_subspace=True)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg, IS_subspace=True, graph=graph)

    # Initialize master equation
    master_equation = LindbladMasterEquation(
        hamiltonians=[laser2, laser1],
        jump_operators=[spontaneous_emission])
    # Begin with all qubits in the ground codes
    psi = np.zeros((rydberg_hamiltonian_cost.hamiltonian.shape[0], 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    results = master_equation.run_ode_solver(psi, 0, tf, num_from_time(tf), schedule=lambda t: schedule(t, tf))
    cost_function = [rydberg_hamiltonian_cost.cost_function(results[i], is_ket=False) / mis for i in
                     range(results.shape[0])]
    print(cost_function[-1])
    plt.scatter(np.linspace(0, tf, num_from_time(tf)), cost_function, c='teal', label='approximation ratio')
    plt.legend()
    plt.xlabel(r'Approximation ratio')
    plt.ylabel(r'Time $t$')
    plt.show()


experiment_params = False
if experiment_params:
    Gamma = 0  # .885
    Delta = 2 * np.pi * 1.5
    Omega_r = np.pi * 5.5
    Omega_g = np.pi * 15
    delta = 2 * np.pi * 100
    Omega = 1.29
else:
    Gamma = 0
    Delta = 1
    Omega_r = 1
    Omega_g = 1
    delta = 100
    Omega = 1 / 100


def adiabatic_simulation(graph, noise_model=None, show_graph=False, delta=3, approximate=False):
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    if noise_model == 'continuous':
        if not approximate:
            laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg)
            laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg)
            laser_detuning = hamiltonian.HamiltonianEnergyShift(index=1, IS_subspace=True, energies=[delta],
                                                                graph=graph, code=rydberg)
            detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
            rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
            spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2),
                                                                          IS_subspace=True, code=rydberg, rates=[Gamma])

            # Initialize adiabatic algorithm
            simulation = SimulateAdiabatic(graph,
                                           hamiltonian=[laser1, laser2, laser_detuning, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='continuous', code=rydberg, noise=[spontaneous_emission])
            return simulation
        else:
            laser = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, energies=[Omega])
            detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
            spontaneous_emission1 = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 1),
                                                                           rates=[(Omega_g / delta) ** 2 * Gamma],
                                                                           IS_subspace=True)
            spontaneous_emission2 = lindblad_operators.SpontaneousEmission(graph=graph, transition=(0, 1),
                                                                           rates=[(Omega_r / delta) ** 2 * Gamma],
                                                                           IS_subspace=True)
            rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)

            simulation = SimulateAdiabatic(graph,
                                           hamiltonian=[laser, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='continuous', noise=[spontaneous_emission1,
                                                                            spontaneous_emission2])
            return simulation

    elif noise_model is None:
        laser = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
        detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
        rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)

        # Initialize adiabatic algorithm
        simulation = SimulateAdiabatic(graph, hamiltonian=[laser, detuning], cost_hamiltonian=rydberg_hamiltonian_cost,
                                       IS_subspace=True)
        return simulation

    elif noise_model == 'monte_carlo':
        if not approximate:
            laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg)
            laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg)
            laser_detuning = hamiltonian.HamiltonianEnergyShift(index=1, IS_subspace=True, energies=[delta],
                                                                graph=graph, code=rydberg)
            detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
            rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
            spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2), rates=[Gamma],
                                                                          IS_subspace=True, code=rydberg)

            # Initialize adiabatic algorithm
            simulation = SimulateAdiabatic(graph,
                                           hamiltonian=[laser1, laser2, laser_detuning, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='monte_carlo', code=rydberg, noise=[spontaneous_emission])
            return simulation
        else:
            laser1 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph)
            detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
            spontaneous_emission1 = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 1),
                                                                           rates=[(Omega_g / delta) ** 2 * Gamma],
                                                                           IS_subspace=True)
            spontaneous_emission2 = lindblad_operators.SpontaneousEmission(graph=graph, transition=(0, 1),
                                                                           rates=[(Omega_r / delta) ** 2 * Gamma],
                                                                           IS_subspace=True)
            rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)

            simulation = SimulateAdiabatic(graph,
                                           hamiltonian=[laser1, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='monte_carlo', noise=[spontaneous_emission1,
                                                                             spontaneous_emission2])
            return simulation


def eit_simulation(graph, noise_model=None, show_graph=False, gamma=3.8, delta=0, approximate=False, Omega_g=1,
                   Omega_r=1):
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    if noise_model == 'continuous':
        if not approximate:
            laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg,
                                                   energies=[Omega_g])
            laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg,
                                                   energies=[Omega_r])
            rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
            detuning = hamiltonian.HamiltonianEnergyShift(code=rydberg, IS_subspace=True, graph=graph, energies=[delta])
            spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2), rates=[gamma],
                                                                          IS_subspace=True, code=rydberg)

            # Initialize adiabatic algorithm
            simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='continuous', code=rydberg, noise=[spontaneous_emission])
            return simulation
        else:
            laser = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph,
                                                  energies=[Omega_g * Omega_r / delta])
            detuning = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
            spontaneous_emission1 = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 1),
                                                                           rates=[(Omega_g / delta) ** 2 * gamma],
                                                                           IS_subspace=True)
            spontaneous_emission2 = lindblad_operators.SpontaneousEmission(graph=graph, transition=(0, 1),
                                                                           rates=[(Omega_r / delta) ** 2 * gamma],
                                                                           IS_subspace=True)
            rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)

            simulation = SimulateAdiabatic(graph,
                                           hamiltonian=[laser, detuning],
                                           cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                           noise_model='continuous', noise=[spontaneous_emission1,
                                                                            spontaneous_emission2])
            return simulation

    elif noise_model == 'monte_carlo':
        laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg)
        laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg)
        rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
        detuning = hamiltonian.HamiltonianEnergyShift(code=rydberg, IS_subspace=True, graph=graph, energies=[delta])

        spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2), rates=[gamma],
                                                                      IS_subspace=True, code=rydberg)

        # Initialize adiabatic algorithm
        simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2, detuning],
                                       cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                       noise_model='monte_carlo', code=rydberg, noise=[spontaneous_emission])
        return simulation


a = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])

graph = nx.from_numpy_array(np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]))


# ARvstime_adiabatic_noise(tf=10, n=2, IS_subspace=False)
# graph, mis = line_graph(2, return_mis=True)


# simulation.ratio_vs_time(25, schedule=lambda t, tf: simulation.rydberg_MIS_schedule(t, tf, coefficients=[np.sqrt(15)]*2), method='odeint', plot=True)
# t1 = time.time()
# print(t1-t)
# simulation.ratio_vs_total_time([10, 20, 50, 75, 100, 200, 300, 400], schedule=simulation.rydberg_MIS_schedule, plot=True)#simulation = adiabatic_simulation(graph, noise_model='monte_carlo')
# [laser1, laser2, laser_detuning1, laser_detuning2, detuning]
def rydberg_linear_MIS_schedule(t, tf):
    if simulation.noise_model is not None:
        for i in range(len(simulation.hamiltonian)):
            if i == 0:
                omega_g = 7.5
                if t < .25:
                    simulation.hamiltonian[i].energies = [t / .25 * omega_g]
                elif .25 <= t <= tf - .25:
                    simulation.hamiltonian[i].energies = [omega_g]
                else:
                    simulation.hamiltonian[i].energies = [(tf - t) / .25 * omega_g]
            elif i == 1:
                omega_r = 2.75
                if t < .25:
                    simulation.hamiltonian[i].energies = [t / .25 * omega_r]
                elif .25 <= t <= tf - .25:
                    simulation.hamiltonian[i].energies = [omega_r]
                else:
                    simulation.hamiltonian[i].energies = [(tf - t) / .25 * omega_r]
            elif i == 3:
                # This is Delta
                Delta = 1.5
                if t < .25:
                    simulation.hamiltonian[i].energies = [Delta]
                elif .25 <= t <= tf - .25:
                    simulation.hamiltonian[i].energies = [-Delta]
                else:
                    simulation.hamiltonian[i].energies = [-2 * Delta * (t - .25) / (tf - .5) + Delta]
    else:
        for i in range(len(simulation.hamiltonian)):
            if i == 0:
                omega = 4 * np.pi
                if t < 2.5:
                    simulation.hamiltonian[i].energies = [t / 2.5 * omega]
                elif 2.5 <= t <= tf - 2.5:
                    simulation.hamiltonian[i].energies = [omega]
                else:
                    simulation.hamiltonian[i].energies = [(tf - t) / 2.5 * omega]
            elif i == 1:
                # This is Delta
                Delta = 1.5 * 2 * np.pi
                if t < 2.5:
                    simulation.hamiltonian[i].energies = [Delta]
                elif 2.5 <= t <= tf - 2.5:
                    simulation.hamiltonian[i].energies = [-2 * Delta * (t - 2.5) / (tf - 5) + Delta]
                else:
                    simulation.hamiltonian[i].energies = [-Delta]
    return True


def experiment_rydberg_MIS_schedule(t, tf, coefficients=None):
    if coefficients is None:
        coefficients = [1, 1, 1]
    if len(simulation.hamiltonian) == 2:
        for i in range(len(simulation.hamiltonian)):
            if i == 0:
                # We don't want to update normal detunings
                simulation.hamiltonian[i].energies = [coefficients[0] * np.sin(np.pi * t / tf) ** 2]
            if i == 1:
                simulation.hamiltonian[i].energies = [coefficients[1] * (2 * t / tf - 1)]
    else:
        for i in range(len(simulation.hamiltonian)):
            if i == 0:
                # We don't want to update normal detunings
                simulation.hamiltonian[i].energies = [coefficients[0] * np.sin(np.pi * t / tf)]
            if i == 1:
                simulation.hamiltonian[i].energies = [coefficients[1] * np.sin(np.pi * t / tf)]
            if i == 3:
                simulation.hamiltonian[i].energies = [coefficients[2] * (2 * t / tf - 1)]
    return True


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


def eit_steady_state(graph, show_graph=False, gamma=1):
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    laser1 = hamiltonian.HamiltonianDriver(transition=(1, 2), IS_subspace=True, graph=graph, code=rydberg)
    laser2 = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph, code=rydberg)
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=rydberg)
    spontaneous_emission = lindblad_operators.SpontaneousEmission(graph=graph, transition=(1, 2), rates=[gamma],
                                                                  IS_subspace=True, code=rydberg)

    # Initialize adiabatic algorithm
    simulation = SimulateAdiabatic(graph, hamiltonian=[laser1, laser2],
                                   cost_hamiltonian=rydberg_hamiltonian_cost, IS_subspace=True,
                                   noise_model='continuous', code=rydberg, noise=[spontaneous_emission])

    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2], jump_operators=[spontaneous_emission])
    initial_state = State(
        tools.outer_product(np.array([[0, 0, 0, 0, 0, 0, 0, 1]]).T, np.array([[0, 0, 0, 0, 0, 0, 0, 1]]).T),
        code=rydberg, IS_subspace=True, is_ket=False)

    print(master_equation.steady_state(initial_state))

    return simulation


#
def time_performance():
    #graph, mis = line_graph(5, return_mis=True)
    graph, mis = degree_fails_graph(return_mis=True)
    graph = Graph(graph)
    ratios_d = [100]
    ratios_r = [100]
    for d in ratios_d:
        for r in ratios_r:
            print(d, r)

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

            simulation_eit = eit_simulation(graph, noise_model='continuous', gamma=1, delta=d, Omega_g=r, Omega_r=r)
            simulation_eit.ratio_vs_total_time([50],
                                         schedule=lambda t, tf: rydberg_EIT_schedule(t, tf,
                                                                                     coefficients=[
                                                                                         r, r]),
                                         plot=True, verbose=True, method='RK23')


time_performance()

graph, mis = line_graph(1, return_mis=True)
graph = Graph(graph)
graph.mis_size = mis
# eit_steady_state(graph)
# delta_vs_T()
# graph = nx.random_regular_graph(3, 8)
# .draw(graph)
# plt.show()

# graph.mis_size = mis

# coefficients=[Omega_g, Omega_r, Delta] [Omega, Delta]
simulation = adiabatic_simulation(graph, noise_model='continuous', delta=delta, approximate=False)
# res, info = simulation.run(20, schedule=lambda t, tf: True, verbose=True, method='odeint', iter=10)
# print(res)
# res, info = simulation.ratio_vs_time(100, schedule=lambda t, tf:
# experiment_rydberg_MIS_schedule(t, tf, coefficients=[Omega_g, Omega_r, Delta]), verbose=True, method='RK45', plot=True)
# experiment_rydberg_MIS_schedule(t, tf, coefficients=[1, 1, 1])
# simulation.noise_model = None
# simulation.ratio_vs_time(10, schedule=lambda t, tf: experiment_rydberg_MIS_schedule(t, tf, coefficients=[15 * np.pi, 5.5 * np.pi, 2 * np.pi * 1.5]),
#                         plot=True, method='odeint', verbose=True)
# simulation.ratio_vs_total_time(np.arange(5, 90, 5), schedule=lambda t, tf: simulation.rydberg_MIS_schedule(t, tf,
#                                                                                                           coefficients=[
#                                                                                                               Omega,
#                                                                                                               Delta]),
#                               plot=True, method='odeint', verbose=True)
graph, mis = line_graph(1, return_mis=True)
graph = Graph(graph)
simulation_eit = eit_simulation(graph, noise_model='continuous', gamma=1, delta=100, Omega_g=1, Omega_r=1)

print('here')
performance, info = simulation_eit.ratio_vs_time(50,
                                                 schedule=lambda t, tf: rydberg_EIT_schedule(t, tf,
                                                                                             coefficients=[
                                                                                                 10, 10]),
                                                 plot=True, verbose=True, method='RK45')


def normalized_time_plot():
    gamma = 1
    gammas = np.arange(.01, .3, .01)
    omegas = np.arange(1, 11, 2)
    tf = 10
    for g in gammas:
        simulation_eit.noise[0].rates = [g]
        for (i, o) in zip(range(len(omegas)), omegas):
            time_scale = o ** 2 / g
            # simulation_eit.hamiltonian[0].energies = [omega]
            # simulation_eit.hamiltonian[1].energies = [omega]
            # dark = 1 / 2 * np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
            print(g, o)
            performance, info = simulation_eit.ratio_vs_time(100 / time_scale,
                                                             schedule=lambda t, tf: rydberg_EIT_schedule(t, tf,
                                                                                                         coefficients=[
                                                                                                             o, o]),
                                                             plot=False, verbose=True, method='RK45')
            plt.plot(info.t * time_scale, performance, c=cm.viridis(o / 10))
    plt.get_cmap('viridis')
    plt.ylabel(r'approximation ratio $\alpha$')
    plt.xlabel(r'time ($\gamma/\Omega^2$)')

    plt.show()


def time_scales_plot():
    gamma = 3.8 * 2
    omegas = np.arange(1, 30)
    # gammas = np.arange(1, 10)/5
    times = list(np.arange(1, 15))
    results = np.zeros((len(omegas), len(times)))

    for (i, o) in zip(range(len(omegas)), omegas):
        omega = np.sqrt(gamma * o)
        # simulation_eit.hamiltonian[0].energies = [omega]
        # simulation_eit.hamiltonian[1].energies = [omega]
        # dark = 1 / 2 * np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
        performance = np.zeros(len(times))
        j = 0
        for t in times:
            print(t)
            res, info = simulation_eit.run(t, schedule=lambda t, tf: rydberg_EIT_schedule(t, tf,
                                                                                          coefficients=[omega, omega]))
            performance[j] = simulation_eit.cost_hamiltonian.cost_function(res[-1])  # tools.fidelity(dark, res[-1])
            j += 1
        # performance = simulation_eit.ratio_vs_total_time(times,
        #                               schedule=lambda t, tf: rydberg_EIT_schedule(t, tf, coefficients=[om, om]),
        #                               plot=False, verbose=True, method='RK45')
        print(performance)
        results[i, ...] = performance
    plt.imshow(results.T, vmin=0, vmax=1, interpolation=None, extent=[0, max(omegas), 0, max(times)],
               origin='lower', aspect='auto')
    plt.colorbar()

    plt.ylabel(r'Total annealing time $T$')
    plt.xlabel(r'$\Omega^2/\gamma$')
    plt.show()


# time_scales_plot()
# simulation_eit.ratio_vs_total_time(list(np.arange(10, 150, 5)),
#                                   schedule=lambda t, tf: rydberg_EIT_schedule(t, tf, coefficients=[3.8, 3.8]),
#                                   plot=False, verbose=True, method='RK45')

"""for i in [10]:
    print('Line size', i)
    graph, mis = line_graph(n=i, return_mis=True)
    graph = Graph(graph)
    graph.mis_size = mis
    simulation = adiabatic_simulation(graph, noise_model='monte_carlo', delta=2 * np.pi * 100, approximate=True)
    simulation.ratio_vs_time(10, schedule=lambda t, tf:
    experiment_rydberg_MIS_schedule(t, tf, coefficients=[Omega, Delta]),
                                          plot=True, verbose=True, method='odeint', iter=10)
"""
"""for i in [2]:
    print('Line size', i)
    graph, mis = line_graph(n=i, return_mis=True)
    graph = Graph(graph)
    graph.mis_size = mis
    simulation_eit = eit_simulation(graph, noise_model='continuous')
    t = time.time()
    simulation_eit.distribution_vs_total_time([5, 10, 15, 20],
                                       schedule=lambda t, tf: rydberg_EIT_schedule(t, tf, coefficients=[3.8, 3.8]),
                                       plot=True, verbose=True, method='odeint', iter=2)
simulation_eit = eit_simulation(graph, noise_model='continuous')
"""
"""simulation.distribution_vs_total_time(list(np.arange(10, 50, 5)),
                                      schedule=lambda t, tf: experiment_rydberg_MIS_schedule(t, tf,
                                                                                             coefficients=[15 * np.pi,
                                                                                                           5.5 * np.pi,
                                                                                                           2 * np.pi * 1.5]),
                                      # schedule=lambda t, tf:simulation.rydberg_MIS_schedule(t, tf, coefficients=[
                                      # 1.2, 2 * np.pi * 1.5]),
                                      plot=True, verbose=True, method='odeint')"""


# simulation_eit = eit_simulation(graph, noise_model='continuous')
# simulation_eit.ratio_vs_total_time(list(np.arange(10, 500, 5)),
#                                   schedule=lambda t, tf: rydberg_EIT_schedule(t, tf, coefficients=[.129, .129]),
#                                   plot=True, verbose=True, method='odeint')
# ARvstime_adiabatic_noise_IS_subspace(n=2, tf=10)
# graph, mis = degree_fails_graph()
# ARvstime_EIT(tf=100, n=2)#graph=graph, mis=mis)
# delta_vs_T(graph=graph, mis=mis)
# omega_vs_T(graph=graph, mis=mis)
# ARvsAT_adiabatic_noise(n=3, IS_subspace=False)
# ARvsAT_adiabatic()
# ARvstime_degree(tf=5)
# ARvstime_line(tf=100, n=4, detailed=True, num=500)
# ARvsAT_adiabatic(n=2)
# ARvsAT_adiabatic_noise(n=2)
# adiabatic_spectrum(n=2)


def hybrid_plot():
    ratios = [.01, .1, 1, 10, 100]
    times = [10, 25, 50, 75, 100]
    res = [[[7.903540710954717e-08, 8.778173466479557e-07, 4.382469969110428e-06, 1.0635930271668862e-05,
             1.9637742280750244e-05, 0, 0],
            [0.000763378017786215, 0.00811810792269105, 0.036388511119153796, 0.07872042928726915, 0.12895874960259915,
             0, 0],
            [0.7098223848838034, 0.847755519874062, 0.9032081874439326, 0.9292207709512127, 0.9440242337287895, 0, 0],
            [0.6405671248646899, 0.7361725476177058, 0.8155461472548492, 0.860201596153856, 0.889037552912785, 0, 0],
            [0.5399513472882004, 0.6059622475169303, 0, 0, 0, 0, 0]],  # jobs 24-26
           [[7.807944201446433e-08, 8.576474300740871e-07, 4.251805394153309e-06, 1.0291536401906625e-05,
             1.897729122483051e-05, 0, 0],
            [0.0007557306572328421, 0.007941728260169187, 0.03545132402798893, 0.0767744376393966, 0.12612810215862263,
             0, 0],
            [0.7092460352522333, 0.8477054421385614, 0.9030660881688419, 0.9289206609694314, 0.9440503423204963, 0, 0],
            [0.6405896040745865, 0.736155655820771, 0.8154582289131395, 0.8601114584532107, 0.8888961101898137, 0, 0],
            [0.5399513472882004, 0.6059622475169303, 0, 0, 0, 0, 0]],  # assume close to 9-13, 24-26
           [[3.5861456762152794e-08, 3.104024825956016e-07, 1.1555643586592528e-06, 2.7867581433449106e-06,
             4.724646848770968e-06, 0, 0],
            [0.00034158777488903203, 0.0024483605027884694, 0.010040825674918938, 0.0223308850944577,
             0.03883398220936498, 0, 0],
            [0.6544369306493069, 0.8583326925458685, 0.9043063604291482, 0.924071782151208, 0.9372100501252504, 0, 0],
            [0.6428601490391608, 0.7345569576683322, 0.8140807484312567, 0.8592771592061664, 0.8883259599822697, 0, 0],
            [0.5351204802168216, 0, 0, 0, 0, 0, 0]],  # jobs 9-13
           [[4.880428157618146e-10, 2.808736687337327e-09, 1.1369990337705131e-08, 2.5551335658756286e-08,
             4.5663331121928976e-08],
            [6.466336912485638e-06, 3.896215491959554e-05, 0.00015476367897887048, 0.0003468247646219208,
             0.0006129305260793942, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.9350780987220587, 0.944114954169243],
            [0.5311983723663157, 0, 0, 0, 0, 0, 0]],  # first five jobs 14-18
           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]]  # first five jobs 19-23
    plt.imshow(res)
    plt.show()
