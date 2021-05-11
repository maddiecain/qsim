import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.codes import rydberg, qubit
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.codes.quantum_state import State
import numpy as np
from qsim import tools
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph, line_graph, ring_graph, degree_fails_graph
from scipy.sparse import csr_matrix


class StrongSpontaneousEmission(object):
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
                IS, nary_to_index, num_IS = graph.independent_sets_qudit(self.code)
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

    def global_liouvillian(self, state: State):
        a = np.zeros(state.shape)
        for i in range(self.graph.n):
            jump_op = self.rates[0] * self.jump_operators[i][0] + self.rates[1] * self.jump_operators[i][1]
            a = a + jump_op @ state @ jump_op.T - 1 / 2 * jump_op.T @ jump_op @ state - \
                1 / 2 * state @ jump_op.T @ jump_op
        return a


def effective_operator_comparison(graph=None, mis=None, tf=10, show_graph=False, n=3, gamma=500):
    # Generate annealing schedule
    def schedule1(t, tf):
        return [[t / tf, (tf - t) / tf, 1], [1]]

    if graph is None:
        graph, mis = line_graph(n=n)
    graph = Graph(graph)
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianDriver(transition=(0, 1), energies=[rabi1], code=rydberg, IS_subspace=True,
                                           graph=graph)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianDriver(transition=(1, 2), energies=[rabi2], code=rydberg, IS_subspace=True,
                                           graph=graph)
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianRydberg(graph, code=rydberg, detuning=1, energy=0,
                                                              IS_subspace=True)
    # Initialize spontaneous emission
    spontaneous_emission_rate = gamma
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg, IS_subspace=True, graph=graph)
    strong_spontaneous_emission_rate = (1, 1)
    strong_spontaneous_emission = lindblad_operators.StrongSpontaneousEmission(transition=(0, 2),
                                                                               rates=strong_spontaneous_emission_rate,
                                                                               code=rydberg, IS_subspace=True,
                                                                               graph=graph)

    def schedule2(t, tf):
        return [[], [
            (2 * t / tf / np.sqrt(spontaneous_emission_rate), 2 * (tf - t) / tf / np.sqrt(spontaneous_emission_rate))]]

    # Initialize master equation
    master_equation = LindbladMasterEquation(
        hamiltonians=[laser2, laser1],
        jump_operators=[spontaneous_emission])
    # Begin with all qubits in the ground codes
    psi = np.zeros((rydberg_hamiltonian_cost.hamiltonian.shape[0], 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)

    # Integrate the master equation
    results1 = master_equation.run_ode_solver(psi, 0, tf, num_from_time(tf), schedule=lambda t: schedule1(t, tf))
    cost_function = [rydberg_hamiltonian_cost.cost_function(results1[i], is_ket=False) / mis for i in
                     range(results1.shape[0])]
    # Initialize master equation
    master_equation = LindbladMasterEquation(
        hamiltonians=[],
        jump_operators=[strong_spontaneous_emission])

    # Integrate the master equation
    results2 = master_equation.run_ode_solver(psi, 0, tf, num_from_time(tf), schedule=lambda t: schedule2(t, tf))
    cost_function_strong = [rydberg_hamiltonian_cost.cost_function(results2[i], is_ket=False) / mis for i in
                            range(results2.shape[0])]
    print(results2[-1])
    times = np.linspace(0, tf, num_from_time(tf))
    # Compute the fidelity of the results
    fidelities = [tools.fidelity(results1[i], results2[i]) for i in range(results1.shape[0])]
    plt.plot(times, fidelities, color='r', label='Fidelity')
    plt.plot(times, cost_function, color='teal', label='Rydberg EIT approximation ratio')
    plt.plot(times, cost_function_strong, color='y', linestyle=':', label='Effective operator approximation ratio')

    plt.hlines(1, 0, max(times), linestyles=':', colors='k')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.03)
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()


def plot_detuning_vs_T():
    adiabaitic = open('degreefails_adiabatic.txt', mode='r')
    times = np.arange(1, 20, 1)
    detunings = np.arange(3, 27, 3)
    costs = []
    for line in adiabaitic:
        costs.append(float(line.split(' ')[2]))
    costs = np.asarray(costs)
    costs = np.reshape(costs, (len(times), len(detunings)))
    print(np.max(costs))
    plt.imshow(costs, vmin=0, vmax=1, interpolation=None, extent=[0, max(detunings), max(times), 0], origin='upper')
    plt.xticks(np.linspace(0, max(detunings), 10))
    plt.yticks(np.linspace(0, max(times), 10))
    plt.colorbar()
    plt.xlabel(r'Detuning $\delta$')
    plt.ylabel(r'Annealing time $T$')
    plt.show()


def transition_matrix(graph, transition):
    IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets

    rows = np.zeros(graph.n * num_IS, dtype=int)
    columns = np.zeros(graph.n * num_IS, dtype=int)
    entries = np.zeros(graph.n * num_IS, dtype=float)
    num_terms = 0
    for i in IS:
        for j in range(len(IS[i][2])):
            if IS[i][2][j] == transition[0]:
                # Flip spin at this location
                # Get binary representation
                temp = IS[i][2].copy()
                temp[j] = transition[1]
                flipped_temp = tools.nary_to_int(temp)
                if flipped_temp in nary_to_index:
                    # This is a valid spin flip
                    rows[num_terms] = nary_to_index[flipped_temp]
                    columns[num_terms] = i
                    entries[num_terms] = 1
                    num_terms += 1
    # Cut off the excess in the arrays
    columns = columns[:num_terms]
    rows = rows[:num_terms]
    entries = entries[:num_terms]
    return csr_matrix((entries, (rows, columns)), shape=(num_IS, num_IS))


def effective_markov_chain(graph=None, n=3, initial_state=None, omega_g=1, omega_r=1, gamma=1, tf=1):
    def schedule(t, tf):
        return [4 * (tf - t) / tf * omega_r ** 2 / gamma, 4 * t / tf * omega_g ** 2 / gamma]

    # Generate a graph
    if graph is None:
        graph, mis = line(n)
    graph = Graph(graph)
    # Generate the transition matrix
    # For each IS, look at spin flips generated by the laser
    # Over-allocate space
    g_to_r = transition_matrix(graph, (1, 0))
    r_to_g = transition_matrix(graph, (0, 1))
    if initial_state is None:
        initial_state = np.zeros((g_to_r.shape[0], 1), dtype=np.float64)
        initial_state[-1, -1] = 1
    t0 = 0
    num = tf * 5
    state = initial_state
    times = np.linspace(t0, tf, num=num)
    dt = times[1] - times[0]
    cost = np.zeros(len(times))
    for i in range(len(times)):
        state_transitions = np.sum(dt * r_to_g * schedule(times[i], tf)[0] + dt * g_to_r * schedule(times[i], tf)[1],
                                   axis=0)
        state = np.multiply((1 - state_transitions.T), state) + (
                    dt * r_to_g * schedule(times[i], tf)[0] + dt * g_to_r * schedule(times[i], tf)[1]) @ state
        cost[i] = state[0]
        # Normalize s
        # print((np.sum(np.abs(s))))
        # s = s/(np.sum(np.abs(s)))
    plt.plot(times, cost)
    plt.show()
    print(state)


# effective_markov_chain(n=1, tf=5000, gamma=100)
# plot_detuning_vs_T()
effective_operator_comparison(n=1, tf=500, gamma=100)
