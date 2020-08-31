from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.codes import rydberg, qubit
from qsim.codes.quantum_state import State
import numpy as np
from qsim import tools
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph, line_graph, ring_graph
from scipy.sparse import csr_matrix

experiment_params = True
if experiment_params:
    Gamma = 8.85 / 10
    Delta = 2 * np.pi * 15 / 10
    Omega_r = 2 * np.pi * 55 / 2 / 10
    Omega_g = 2 * np.pi * 150 / 2 / 10
    delta = 2 * np.pi * 1000 / 2 / 10


class EffectiveOperatorSpontaneousEmission(object):
    def __init__(self, transition: tuple, rates: list, code=None, IS_subspace=True, graph=None):
        # jump_operators and weights are numpy arrays
        assert IS_subspace
        if code is None:
            code = qubit
        self.code = code
        if rates is None:
            rates = [1, 1]
        self.rates = rates
        self.transition = transition
        self.IS_subspace = IS_subspace
        self.graph = graph
        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            if code is not qubit:
                IS, nary_to_index, num_IS = graph.independent_sets_code(self.code)
            else:
                # We have already solved for this information
                IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets
            self.jump_operators = []
            # For each atom, consider the states spontaneous emission can generate transitions between
            # Over-allocate space
            for j in range(graph.n):
                rows_1 = np.zeros(num_IS, dtype=int)
                columns_1 = np.zeros(num_IS, dtype=int)
                entries_1 = np.zeros(num_IS, dtype=int)
                rows_2 = np.zeros(num_IS, dtype=int)
                columns_2 = np.zeros(num_IS, dtype=int)
                entries_2 = np.zeros(num_IS, dtype=int)
                num_terms_1 = 0
                num_terms_2 = 0
                for i in IS:
                    if IS[i][2][j] == self.transition[0]:
                        # Flip spin at this location
                        # Get binary representation
                        temp = IS[i][2].copy()
                        temp[j] = self.transition[1]
                        flipped_temp = tools.nary_to_int(temp, base=code.d)
                        if flipped_temp in nary_to_index:
                            # This is a valid spin flip
                            rows_1[num_terms_1] = nary_to_index[flipped_temp]
                            columns_1[num_terms_1] = i
                            entries_1[num_terms_1] = 1
                            num_terms_1 += 1
                    elif IS[i][2][j] == self.transition[1]:
                        rows_2[num_terms_2] = i
                        columns_2[num_terms_2] = i
                        entries_2[num_terms_2] = 1
                        num_terms_2 += 1
                # Cut off the excess in the arrays
                columns_1 = columns_1[:num_terms_1]
                rows_1 = rows_1[:num_terms_1]
                entries_1 = entries_1[:num_terms_1]
                columns_2 = columns_2[:num_terms_2]
                rows_2 = rows_2[:num_terms_2]
                entries_2 = entries_2[:num_terms_2]
                # Now, append the jump operator
                self.jump_operators.append((csr_matrix((entries_1, (rows_1, columns_1)), shape=(num_IS, num_IS)),
                                            csr_matrix((entries_2, (rows_2, columns_2)), shape=(num_IS, num_IS))))

    def liouvillian(self, state: State):
        a = np.zeros(state.shape)
        for i in range(self.graph.n):
            jump_op = self.rates[0] * self.jump_operators[i][0] + self.rates[1] * self.jump_operators[i][1]
            a = a + jump_op @ state @ jump_op.T - 1 / 2 * jump_op.T @ jump_op @ state - \
                1 / 2 * state @ jump_op.T @ jump_op
        return a


def adiabatic_simulation(graph, show_graph=False, approximate=False):
    if show_graph:
        nx.draw(graph)
        plt.show()
    if approximate:
        laser = hamiltonian.HamiltonianDriver(transition=(0, 1), IS_subspace=True, graph=graph,
                                              energies=(Omega_g * Omega_r / delta,))
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
    else:
        # Generate the driving and Rydberg Hamiltonians
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


def run(times, n=2, approximate=False):
    graph = line_graph(n)
    simulation = adiabatic_simulation(graph, approximate=approximate)

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

    # Omega_g * Omega_r / delta
    res = simulation.performance_vs_time(2, schedule=lambda t, tf:
    experiment_rydberg_MIS_schedule(t, tf, coefficients=[Omega_g, Omega_r, Delta]),
                                               plot=True, verbose=False, method=['trotterize', 'odeint'],
                                         metric=['approximation_ratio', 'optimum_overlap'])

    return res


res = run([2, 4], approximate=True, n=3)
print(res, flush=True)
