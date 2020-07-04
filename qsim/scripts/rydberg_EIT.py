import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.state import rydberg_EIT, state_tools, qubit
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.schrodinger_equation import SchrodingerEquation
import numpy as np
from qsim import tools
import matplotlib.pyplot as plt
from qsim.graph_algorithms import graph as g
import scipy.linalg as sp
from qsim.graph_algorithms.graph import IS_projector


def cosine_schedule(t, tf):
    return [[1 - np.cos(t / tf * np.pi / 2), np.cos(t / tf * np.pi / 2), 1], [1]]


def step_schedule(t, tf):
    if t < tf / 2:
        return [[0, 0, 0], [0]]
    else:
        return [[1, 0, 0], [1]]


# Generate a simple graph
def degree_graph():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 4, 1), (0, 5, 1), (4, 5, 1), (1, 4, 1), (1, 3, 1), (2, 4, 1)])
    return graph, 3


def triangle():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 2, 1), (1, 2, 1)])
    return graph, 1


def line(n=3):
    graph = nx.Graph()
    if n == 1:
        graph.add_node(0)
    graph.add_weighted_edges_from(
        [(i, i + 1, 1) for i in range(n - 1)])
    return [graph, np.ceil(n / 2)]


def ARvstime_degree(tf=10, schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]], show_graph=False, detailed=True):
    graph, mis = degree_graph()
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 50
    cost_function = []
    times = np.linspace(0, tf, num=num)
    # Integrate the master equation
    results = master_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
    cost_function = [rydberg.cost_function(results[i], is_ket=False) / mis for i in range(results.shape[0])]

    # Add initial conditions
    plt.plot(times, cost_function, color='teal', label='approximation ratio')

    plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()


def ARvstime_line(tf=10, schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]], n=3, show_graph=False, detailed=True,
                  detuning=0, num=50):
    graph, mis = line(n=n)
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 1
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT, detuning=detuning)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule

    def rydberg_subspace(state):
        if n == 3:
            r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
            one_r = tools.tensor_product([r, not_r, not_r])
            two_r = tools.tensor_product([r, r, not_r])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_r, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_r, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_r, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 0, 1], two_r, is_ket=False)
            return res
        if n == 2:
            r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
            one_r = tools.tensor_product([r, not_r])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1], one_r, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0], one_r, is_ket=False)
            return res

    def excited_subspace(state):
        if n == 3:
            e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            not_e = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
            one_e = tools.tensor_product([e, not_e, not_e])
            two_e = tools.tensor_product([e, e, not_e])
            three_e = tools.tensor_product([e, e, e])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [0, 2, 1], two_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_e, is_ket=False)

            return res
        if n == 2:
            e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            not_e = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
            one_e = tools.tensor_product([e, not_e])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1], one_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0], one_e, is_ket=False)
            return res

    def ground_subspace(state):
        if n == 3:
            g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            not_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
            one_g = tools.tensor_product([g, not_g, not_g])
            two_g = tools.tensor_product([g, g, not_g])
            three_g = tools.tensor_product([g, g, g])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], two_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_g, is_ket=False)

            return res
        if n == 2:
            g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            not_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
            one_g = tools.tensor_product([g, not_g])
            two_g = tools.tensor_product([g, g])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1], one_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0], one_g, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [0, 1], two_g, is_ket=False)

            return res

    def ge(state):
        assert n == 2
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_g = tools.tensor_product([g, e])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1], one_g, is_ket=False)
        return res

    def gg(state):
        assert n == 2
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_g = tools.tensor_product([g, g])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1], one_g, is_ket=False)
        return res

    def gr(state):
        assert n == 2
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        one_g = tools.tensor_product([r, g])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1], one_g, is_ket=False)
        return res

    def er(state):
        assert n == 2
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        one_g = tools.tensor_product([e, r])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1], one_g, is_ket=False)
        return res

    def ee(state):
        assert n == 2
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        one_g = tools.tensor_product([e, e])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1], one_g, is_ket=False)
        return res

    def mis_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        mis_proj = tools.tensor_product([r, not_r, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], mis_proj, is_ket=False)

    def is_subspace(state):
        return g.IS_projector(graph, rydberg_EIT) * state

    def ggg(state):
        assert n == 3
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([g, g, g])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def gge(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([g, g, e])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def geg(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([g, e, g])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def gee(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([g, e, e])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def ege(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([e, g, e])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def eee(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        op = tools.tensor_product([e, e, e])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def ggr(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([g, g, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def ger(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([g, e, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def egr(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([e, g, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def erg(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([e, r, g])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def grg(state):
        assert n == 3
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        op = tools.tensor_product([g, r, g])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def eer(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        op = tools.tensor_product([e, e, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def ere(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        op = tools.tensor_product([e, r, e])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def rgr(state):
        assert n == 3
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([r, g, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    def rer(state):
        assert n == 3
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        op = tools.tensor_product([r, e, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], op, is_ket=False)

    times = np.linspace(0, tf, num=num)
    if n == 3:
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
        cost_function = [rydberg.cost_function(results[i], is_ket=False) / mis for i in range(results.shape[0])]
        tools.is_valid_state(results[-1])
        if detailed:
            is_pop = [np.real(np.trace(is_subspace(results[i]))) for i in range(results.shape[0])]
            res_ggg = np.array([np.real(np.trace(ggg(results[i]))) for i in range(results.shape[0])])
            res_gge = np.array([np.real(np.trace(gge(results[i]))) for i in range(results.shape[0])])
            res_geg = np.array([np.real(np.trace(geg(results[i]))) for i in range(results.shape[0])])
            res_gee = np.array([np.real(np.trace(gee(results[i]))) for i in range(results.shape[0])])
            res_ege = np.array([np.real(np.trace(ege(results[i]))) for i in range(results.shape[0])])
            res_ere = np.array([np.real(np.trace(ere(results[i]))) for i in range(results.shape[0])])
            res_eee = np.array([np.real(np.trace(eee(results[i]))) for i in range(results.shape[0])])
            res_eer = np.array([np.real(np.trace(eer(results[i]))) for i in range(results.shape[0])])
            res_ggr = np.array([np.real(np.trace(ggr(results[i]))) for i in range(results.shape[0])])
            res_ger = np.array([np.real(np.trace(ger(results[i]))) for i in range(results.shape[0])])
            res_egr = np.array([np.real(np.trace(egr(results[i]))) for i in range(results.shape[0])])
            res_grg = np.array([np.real(np.trace(grg(results[i]))) for i in range(results.shape[0])])
            res_erg = np.array([np.real(np.trace(erg(results[i]))) for i in range(results.shape[0])])
            res_rgr = np.array([np.real(np.trace(rgr(results[i]))) for i in range(results.shape[0])])
            res_rer = np.array([np.real(np.trace(rer(results[i]))) for i in range(results.shape[0])])
            plt.plot(times, is_pop, color='k', label='is overlap', linestyle='--')
            plt.plot(times, cost_function, color='teal', label='approximation ratio')
            plt.plot(times, res_ggg, label=r'$|ggg\rangle$', linestyle='--')
            # plt.plot(times, res_gge, label=r'$|gge\rangle, |egg\rangle$')
            plt.plot(times, res_geg, label=r'$|geg\rangle$', linestyle='--')
            plt.plot(times, res_gee, label=r'$|gee\rangle, |eeg\rangle$', linestyle='--')
            # plt.plot(times, res_ege, label=r'$|ege\rangle$')
            # plt.plot(times, res_eee, label=r'$|eee\rangle$')
            plt.plot(times, res_ere, label=r'$|ere\rangle$', linestyle='--')
            plt.plot(times, res_grg, label=r'$|grg\rangle$', linestyle='--')
            plt.plot(times, res_eer, label=r'$|eer\rangle, |ree\rangle$', linestyle='--')
            plt.plot(times, res_ggr, label=r'$|ggr\rangle, |rgg\rangle$', linestyle='--')
            plt.plot(times, res_ger, label=r'$|ger\rangle, |reg\rangle$', linestyle='--')
            # plt.plot(times, res_egr, label=r'$|egr\rangle, |rge\rangle$')
            # plt.plot(times, res_eer, label=r'$|eer\rangle, |ree\rangle$')
            # plt.plot(times, res_egr, label=r'$|egr\rangle, |rge\rangle$')
            # plt.plot(times, res_erg, label=r'$|erg\rangle, |gre\rangle$')
            plt.plot(times, res_rgr, label=r'$|rgr\rangle$', linestyle='--')
            plt.plot(times, res_rer, label=r'$|rer\rangle$', linestyle='--')

        else:
            # Add initial conditions
            is_pop = [np.real(np.trace(is_subspace(results[i]))) for i in range(results.shape[0])]
            ground_pop = [np.real(np.trace(ground_subspace(results[i]))) for i in range(results.shape[0])]
            excited_pop = [np.real(np.trace(excited_subspace(results[i]))) for i in range(results.shape[0])]
            rydberg_pop = [np.real(np.trace(rydberg_subspace(results[i]))) for i in range(results.shape[0])]
            mis_pop = [np.real(np.trace(mis_subspace(results[i]))) for i in range(results.shape[0])]
            plt.plot(times, cost_function, color='teal', label='approximation ratio')
            plt.plot(times, excited_pop, color='y', linestyle='--', label=r'$P(n_e)\geq 1$')
            plt.plot(times, rydberg_pop, color='g', linestyle='--', label=r'$P(n_r)\geq 1$')
            plt.plot(times, ground_pop, color='m', linestyle='--', label=r'$P(n_g)\geq 1$')
            plt.plot(times, mis_pop, color='r', linestyle='--', label=r'mis overlap')
            plt.plot(times, is_pop, color='k', linestyle='--', label=r'is overlap')

    if n == 2:
        res_er = []
        res_gg = []
        res_gr = []
        res_ge = []
        res_ee = []

        cost_function = []
        is_pop = []
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
        print(np.around(results[-1], decimals=3), np.argwhere(results[-1] != 0))
        cost_function = [rydberg.cost_function(results[i], is_ket=False) / mis for i in range(results.shape[0])]
        res_er.append(np.real(np.trace(er(results[-1]))))
        res_gg.append(np.real(np.trace(gg(results[-1]))))
        res_ee.append(np.real(np.trace(ee(results[-1]))))
        res_ge.append(np.real(np.trace(ge(results[-1]))))
        res_gr.append(np.real(np.trace(gr(results[-1]))))

        is_pop.append(np.real(np.trace(is_subspace(results[-1]))))
        # Add initial conditions
        is_pop = [np.real(np.trace(is_subspace(results[i]))) for i in range(results.shape[0])]
        res_gg = [np.real(np.trace(gg(results[i]))) for i in range(results.shape[0])]
        res_gr = [np.real(np.trace(gr(results[i]))) for i in range(results.shape[0])]
        res_ee = [np.real(np.trace(ee(results[i]))) for i in range(results.shape[0])]
        res_ge = [np.real(np.trace(ge(results[i]))) for i in range(results.shape[0])]
        res_er = [np.real(np.trace(er(results[i]))) for i in range(results.shape[0])]

        plt.plot(times, cost_function, color='teal', label='approximation ratio')
        plt.plot(times, res_gg, color='y', linestyle='--', label=r'$|gg\rangle$')
        plt.plot(times, res_gr, color='g', linestyle='--', label=r'$|gr\rangle, |rg\rangle$')
        plt.plot(times, res_ee, color='m', linestyle='--', label=r'$|ee\rangle$')
        plt.plot(times, res_ge, color='r', linestyle='--', label=r'$|ge\rangle, |eg\rangle$')
        plt.plot(times, res_er, linestyle='--', label=r'$|er\rangle, |re\rangle$')
        plt.plot(times, is_pop, color='k', linestyle='--', label=r'is overlap')
    print(np.trace(results[-1]@results[-1]))

    plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()


def ARvstime_triangle(tf=10, schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]):
    graph, mis = triangle()
    # nx.draw(graph)
    # plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = .01

    def rydberg_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        one_r = tools.tensor_product([r, not_r, not_r])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_r, is_ket=False)

        return res

    def excited_subspace(state):
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        not_e = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_e = tools.tensor_product([e, not_e, not_e])
        two_e = tools.tensor_product([e, e, not_e])
        three_e = tools.tensor_product([e, e, e])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 2, 1], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_e, is_ket=False)

        return res

    def ground_subspace(state):
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        not_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        one_g = tools.tensor_product([g, not_g, not_g])
        two_g = tools.tensor_product([g, g, not_g])
        three_g = tools.tensor_product([g, g, g])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 2, 0], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 0, 1], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_g, is_ket=False)

        return res

    def mis_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        mis = tools.tensor_product([r, not_r, not_r])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], mis, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], mis, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], mis, is_ket=False)
        return res

    def is_subspace(state):
        return g.IS_projector(graph, rydberg_EIT) * state

    times = np.linspace(0, tf, num)
    excited_pop = []
    ground_pop = []
    rydberg_pop = []
    cost_function = []
    mis_pop = []
    is_pop = []
    # Integrate the master equation
    results = master_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
    cost_function = [rydberg.cost_function(results[i], is_ket=False) / mis for i in range(results.shape[0])]
    excited_pop.append(np.real(np.trace(excited_subspace(results[-1]))))
    rydberg_pop.append(np.real(np.trace(rydberg_subspace(results[-1]))))
    mis_pop.append(np.real(np.trace(mis_subspace(results[-1]))))
    is_pop.append(np.real(np.trace(is_subspace(results[-1]))))
    ground_pop.append(np.real(np.trace(ground_subspace(results[-1]))))
    # Add initial conditions
    is_pop = [np.real(np.trace(is_subspace(results[i]))) for i in range(results.shape[0])]
    ground_pop = [np.real(np.trace(ground_subspace(results[i]))) for i in range(results.shape[0])]
    excited_pop = [np.real(np.trace(excited_subspace(results[i]))) for i in range(results.shape[0])]
    rydberg_pop = [np.real(np.trace(rydberg_subspace(results[i]))) for i in range(results.shape[0])]
    mis_pop = [np.real(np.trace(mis_subspace(results[i]))) for i in range(results.shape[0])]
    plt.plot(times, cost_function, color='teal', label='approximation ratio')
    plt.plot(times, excited_pop, color='y', linestyle='--', label=r'$P(n_e)\geq 1$')
    plt.plot(times, rydberg_pop, color='g', linestyle='--', label=r'$P(n_r)\geq 1$')
    plt.plot(times, ground_pop, color='m', linestyle='--', label=r'$P(n_g)\geq 1$')
    plt.plot(times, mis_pop, color='r', linestyle='--', label=r'mis overlap')
    plt.plot(times, is_pop, color='k', linestyle='--', label=r'is overlap')
    # plt.hlines(1/40, 0, max(times), linestyle='--')
    # plt.hlines(1/5, 0, max(times))
    # print(cost_function[-1], excited_pop[-1], ground_pop[-1])

    plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()


def ARvsAT_line(schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]], show_graph=False, n=3):
    graph, mis = line(n=n)
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 1000

    def rydberg_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        one_r = tools.tensor_product([r, not_r, not_r])
        two_r = tools.tensor_product([r, r, not_r])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_r, is_ket=False)
        # res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 0, 1], two_r, is_ket=False)
        # res = res + rydberg_EIT.left_multiply(state, [1, 2, 0], two_r, is_ket=False)

        return res

    def excited_subspace(state):
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        not_e = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_e = tools.tensor_product([e, not_e, not_e])
        two_e = tools.tensor_product([e, e, not_e])
        three_e = tools.tensor_product([e, e, e])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 2, 1], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_e, is_ket=False)

        return res

    def ground_subspace(state):
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        not_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        one_g = tools.tensor_product([g, not_g, not_g])
        two_g = tools.tensor_product([g, g, not_g])
        three_g = tools.tensor_product([g, g, g])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_g, is_ket=False)

        return res

    def single_excitation_subspace(state):
        g = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        not_g = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_g = tools.tensor_product([g, not_g, not_g])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], one_g, is_ket=False)

    def mis_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        mis_proj = tools.tensor_product([r, not_r, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], mis_proj, is_ket=False)

    def is_subspace(state):
        return g.IS_projector(graph, rydberg_EIT) * state

    times = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 50, 100, 200, 500, 1000]
    excited_pop = []
    ground_pop = []
    cost_function = []
    mis_pop = []
    rydberg_pop = []
    for tf in times:
        print(tf)
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
        cost_function.append(rydberg.cost_function(results[-1], is_ket=False) / mis)
        excited_pop.append(np.real(np.trace(excited_subspace(results[-1]))))
        mis_pop.append(np.real(np.trace(mis_subspace(results[-1]))))
        ground_pop.append(np.real(np.trace(ground_subspace(results[-1]))))
        rydberg_pop.append(np.real(np.trace(rydberg_subspace(results[-1]))))
    # Add initial conditions
    cost_function = [rydberg.cost_function(results[0], is_ket=False) / mis] + cost_function
    excited_pop = [np.real(np.trace(excited_subspace(results[0])))] + excited_pop
    ground_pop = [np.real(np.trace(ground_subspace(results[0])))] + ground_pop
    rydberg_pop = [np.real(np.trace(rydberg_subspace(results[0])))] + rydberg_pop
    mis_pop = [np.real(np.trace(mis_subspace(results[0])))] + mis_pop
    times = [0] + times
    plt.scatter(times, cost_function, color='teal', label='approximation ratio')
    plt.scatter(times, excited_pop, color='y', label=r'$P(n_e)\geq 1$')
    plt.scatter(times, ground_pop, color='m', label=r'$P(n_g)\geq 1$')
    plt.scatter(times, rydberg_pop, color='g', label=r'rydberg population')

    plt.scatter(times, mis_pop, color='r', label=r'mis overlap')
    plt.plot(times, cost_function, color='teal')
    plt.plot(times, excited_pop, color='y', linestyle='--')
    plt.plot(times, ground_pop, color='m', linestyle='--')
    plt.plot(times, rydberg_pop, color='g', linestyle='--')

    plt.plot(times, mis_pop, color='r', linestyle='--')
    # plt.hlines(1 / 40, 0, max(times), linestyle='--')
    # plt.hlines(1 / 5, 0, max(times))

    plt.legend()
    plt.xlabel(r'Total annealing time $T$')
    plt.ylabel(r'Final approximation ratio')
    plt.show()


def ARvsAT_triangle(schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]], show_graph=False):
    graph, mis = triangle()
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 50

    def rydberg_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        one_r = tools.tensor_product([r, not_r, not_r])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_r, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_r, is_ket=False)

        return res

    def excited_subspace(state):
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        not_e = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_e = tools.tensor_product([e, not_e, not_e])
        two_e = tools.tensor_product([e, e, not_e])
        three_e = tools.tensor_product([e, e, e])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 2, 1], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_e, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_e, is_ket=False)

        return res

    def ground_subspace(state):
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        not_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        one_g = tools.tensor_product([g, not_g, not_g])
        two_g = tools.tensor_product([g, g, not_g])
        three_g = tools.tensor_product([g, g, g])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], one_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_g, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_g, is_ket=False)

        return res

    def single_excitation_subspace(state):
        g = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        not_g = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        one_g = tools.tensor_product([g, not_g, not_g])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], one_g, is_ket=False)

    def mis_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        mis_proj = tools.tensor_product([not_r, not_r, r])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0, 1, 2], mis_proj, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [0, 2, 1], mis_proj, is_ket=False)
        res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], mis_proj, is_ket=False)
        return res

    def is_subspace(state):
        return g.IS_projector(graph, rydberg_EIT) * state

    times = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 35, 50, 100, 200, 500, 1000]
    excited_pop = []
    ground_pop = []
    cost_function = []
    mis_pop = []
    rydberg_pop = []
    for tf in times:
        print(tf)
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
        cost_function.append(rydberg.cost_function(results[-1], is_ket=False) / mis)
        excited_pop.append(np.real(np.trace(excited_subspace(results[-1]))))
        mis_pop.append(np.real(np.trace(mis_subspace(results[-1]))))
        ground_pop.append(np.real(np.trace(ground_subspace(results[-1]))))
        rydberg_pop.append(np.real(np.trace(rydberg_subspace(results[-1]))))
    # Add initial conditions
    cost_function = [rydberg.cost_function(results[0], is_ket=False) / mis] + cost_function
    excited_pop = [np.real(np.trace(excited_subspace(results[0])))] + excited_pop
    ground_pop = [np.real(np.trace(ground_subspace(results[0])))] + ground_pop
    rydberg_pop = [np.real(np.trace(rydberg_subspace(results[0])))] + rydberg_pop
    mis_pop = [np.real(np.trace(mis_subspace(results[0])))] + mis_pop
    times = [0] + times
    plt.scatter(times, cost_function, color='teal', label='approximation ratio')
    plt.scatter(times, excited_pop, color='y', label=r'$P(n_e)\geq 1$')
    plt.scatter(times, ground_pop, color='m', label=r'$P(n_g)\geq 1$')
    plt.scatter(times, rydberg_pop, color='g', label=r'rydberg population')

    plt.scatter(times, mis_pop, color='r', label=r'mis overlap')
    plt.plot(times, cost_function, color='teal')
    plt.plot(times, excited_pop, color='y', linestyle='--')
    plt.plot(times, ground_pop, color='m', linestyle='--')
    plt.plot(times, rydberg_pop, color='g', linestyle='--')

    plt.plot(times, mis_pop, color='r', linestyle='--')
    # plt.hlines(1 / 40, 0, max(times), linestyle='--')
    # plt.hlines(1 / 5, 0, max(times))

    plt.legend()
    plt.xlabel(r'Total annealing time $T$')
    plt.ylabel(r'Final approximation ratio')
    plt.show()


def lindblad_eigenvalues(state=None, k=5, n=3, coefficients=None):
    graph, mis = line(n=n)
    shape = (rydberg_EIT.d ** n, rydberg_EIT.d ** n)
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])

    eigvals, eigvecs = master_equation.eig(state=state, k=k, shape=shape, coefficients=coefficients)
    # Reshape eigenvectors
    eigvecs = np.reshape(eigvecs, [shape[0], shape[1], eigvecs.shape[-1]])
    eigvecs = np.moveaxis(eigvecs, -1, 0)
    print(eigvals)
    steady_state = np.argwhere(np.isclose(eigvals.real, 0)).T[0]
    #near_steady_state = np.argwhere(np.isclose(eigvals.real, -8.00554902e-04)).T[0]
    #print('near steady state:', near_steady_state)
    steady_state_eigvecs = eigvecs[steady_state, :, :]
    #near_steady_state_eigvecs = eigvecs[near_steady_state, :, :]
    # Renormalize eigenstates
    for i in range(steady_state_eigvecs.shape[0]):
        steady_state_eigvecs[i, :, :] = steady_state_eigvecs[i, :, :] / np.trace(steady_state_eigvecs[i, :, :])
        tools.is_valid_state(steady_state_eigvecs[i, :, :], verbose=False)
    #for j in range(near_steady_state_eigvecs.shape[0]):
    #    near_steady_state_eigvecs[j, :, :] = near_steady_state_eigvecs[j, :, :] / np.trace(near_steady_state_eigvecs[j, :, :])
    #print('near steady state eigvecs', near_steady_state_eigvecs)
    steady_state_eigvals = eigvals[steady_state]
    print(steady_state_eigvals, steady_state_eigvecs)
    print(np.around(steady_state_eigvecs, decimals=4))
    cost_function = [rydberg.cost_function(steady_state_eigvecs[i], is_ket=False) / mis for i in range(len(steady_state_eigvals))] * 2
    steady_state_eigvals_cc = steady_state_eigvals.conj()
    steady_state_eigvals_real = np.concatenate((steady_state_eigvals_cc.real, steady_state_eigvals_cc.real))
    steady_state_eigvals_complex = np.concatenate((steady_state_eigvals_cc.imag, steady_state_eigvals_cc.imag))
    eigvals_cc = eigvals.conj()
    eigvals_real = np.concatenate((eigvals_cc.real, eigvals_cc.real))
    eigvals_complex = np.concatenate((eigvals_cc.imag, eigvals_cc.imag))
    plt.hlines(0, xmin=-1, xmax=.1)
    plt.vlines(0, ymin=-100, ymax=100)
    plt.scatter(eigvals_real, eigvals_complex, c='k', s=4)
    plt.scatter(steady_state_eigvals_real, steady_state_eigvals_complex, cmap='spring', c=cost_function)
    plt.colorbar()
    plt.show()


def lindblad_steady_state(k=1, coefficients=None):
    # Generate the driving and Rydberg Hamiltonians
    graph, mis = line(3)
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode
    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-3, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 50
    eigvals, eigvecs = master_equation.steady_state(psi, k=k, coefficients=coefficients)
    eigvecs = np.reshape(eigvecs, [psi.shape[0], psi.shape[1], eigvecs.shape[-1]])
    eigvecs[:, :, 0] = eigvecs[:, :, 0] / np.trace(eigvecs[:, :, 0])
    print(eigvecs[:, :, 0], np.argwhere(eigvecs[:, :, 0] > 1e-07))
    # print(tools.is_valid_state(eigvecs[:,:,0]))
    print(rydberg.cost_function(eigvecs[:, :, 0], is_ket=False))


def STIRAP(schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]):
    graph, mis = line(n=1)
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 50

    def rydberg_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0], r, is_ket=False)
        return res

    def excited_subspace(state):
        e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0], e, is_ket=False)
        return res

    def ground_subspace(state):
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        res = np.zeros_like(state)
        res = res + rydberg_EIT.left_multiply(state, [0], g, is_ket=False)
        return res

    times = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 50]  # , 500, 1000]
    excited_pop = []
    ground_pop = []
    rydberg_pop = []
    for final in times:
        print(final)
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, tf, num,
                                                 schedule=lambda t: [[t / final, (final - t) / final, 1], [1]])
        excited_pop.append(np.real(np.trace(excited_subspace(results[-1]))))
        ground_pop.append(np.real(np.trace(ground_subspace(results[-1]))))
        rydberg_pop.append(np.real(np.trace(rydberg_subspace(results[-1]))))
    # Add initial conditions
    excited_pop = [np.real(np.trace(excited_subspace(results[0])))] + excited_pop
    ground_pop = [np.real(np.trace(ground_subspace(results[0])))] + ground_pop
    rydberg_pop = [np.real(np.trace(rydberg_subspace(results[0])))] + rydberg_pop
    times = [0] + times
    plt.scatter(times, excited_pop, color='y', label=r'$|e\rangle$')
    plt.scatter(times, ground_pop, color='m', label=r'$|g\rangle$')
    plt.scatter(times, rydberg_pop, color='g', label=r'$|r\rangle$')
    plt.plot(times, excited_pop, color='y', linestyle='--')
    plt.plot(times, ground_pop, color='m', linestyle='--')
    plt.plot(times, rydberg_pop, color='g', linestyle='--')

    plt.legend()
    plt.xlabel(r'Total annealing time $T$')
    plt.ylabel(r'Final overlap')
    plt.show()


def SSvstime_line(schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]], show_graph=False, n=3, detuning=0):
    graph, mis = line(n=n)
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=rydberg_EIT, detuning=detuning)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 50

    def lv(state, schedule, t, tf):
        res = np.zeros(state.shape)
        coefficients = schedule(t, tf)
        for i in range(len(master_equation.hamiltonians)):
            res = res - 1j * coefficients[0][i] * (master_equation.hamiltonians[i].left_multiply(state, is_ket=False) -
                                                   master_equation.hamiltonians[i].right_multiply(state, is_ket=False))

        for i in range(len(master_equation.jump_operators)):
            res = res + coefficients[1][i] * master_equation.jump_operators[i].liouvillian(state, is_ket=False)
        return np.trace(sp.sqrtm(res.conj().T @ res)).real

    final_times = [1, 3, 5, 7, 9, 11, 13]  # , 15, 17, 19, 23, 27, 33, 37, 43, 46, 48, 50, 100]#, 200, 500]
    ss = []
    times = []
    for final in final_times:
        print(final)
        t = np.linspace(0, final, num=num)
        times.append(t)
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, final, num, schedule=lambda t: schedule(t, final))
        ss.append([lv(results[ti], schedule, t[ti], final) for ti in range(len(t))])
    # Add initial conditions
    for i in range(len(ss)):
        plt.plot(times[i], ss[i])
    # plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Tr$(\sqrt{\mathcal{L}[\rho]^2})$')
    plt.show()


def SSvstime_STIRAP(schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]):
    graph, mis = line(n=1)
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(1, 2), energy=rabi1, code=rydberg_EIT)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi2, code=rydberg_EIT)
    # Initialize spontaneous emission
    spontaneous_emission_rate = 1
    spontaneous_emission = lindblad_operators.SpontaneousEmission(transition=(1, 2), rate=spontaneous_emission_rate,
                                                                  code=rydberg_EIT)

    # Initialize master equation
    master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2],
                                             jump_operators=[spontaneous_emission])
    from scipy.integrate import ode

    # Begin with all qubits in the ground state
    psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    psi = tools.outer_product(psi, psi)
    # Generate annealing schedule
    num = 50

    def lv(state, schedule, t, tf):
        res = np.zeros(state.shape)
        coefficients = schedule(t, tf)
        for i in range(len(master_equation.hamiltonians)):
            res = res - 1j * coefficients[0][i] * (master_equation.hamiltonians[i].left_multiply(state, is_ket=False) -
                                                   master_equation.hamiltonians[i].right_multiply(state, is_ket=False))

        for i in range(len(master_equation.jump_operators)):
            res = res + coefficients[1][i] * master_equation.jump_operators[i].liouvillian(state, is_ket=False)
        return np.trace(sp.sqrtm(res.conj().T @ res)).real

    final_times = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 33, 37, 43, 46, 48, 50, 100]  # , 200, 500]
    ss = []
    times = []
    for final in final_times:
        print(final)
        t = np.linspace(0, final, num=num)
        times.append(t)
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, final, num, schedule=lambda t_in: schedule(t_in, final))
        ss.append([lv(results[ti], schedule, t[ti], final) for ti in range(len(t))])

    # Add initial conditions
    for i in range(len(ss)):
        plt.plot(times[i], ss[i])
    # plt.legend()
    plt.xlabel(r'annealing time $t$')
    plt.ylabel(r'Tr$(\sqrt{\mathcal{L}[\rho]^2})$')
    plt.show()


def adiabatic_schedule(t, tf, delta=1, omega=1):
    return [omega * np.sin(np.pi * t / tf) ** 2, delta * (2 * t / tf - 1), 1]


def ARvsAT_adiabatic(schedule=adiabatic_schedule, show_graph=False, n=3):
    graph, mis = line(n=n)
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi1, code=qubit)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(pauli='Z', transition=(0, 1), energy=rabi2, code=qubit)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=qubit)

    # Initialize Schrodinger equation
    schrodinger_equation = SchrodingerEquation(hamiltonians=[laser1, laser2, rydberg])

    # Begin with all qubits in the ground state
    psi = np.zeros((qubit.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    # Generate annealing schedule
    num = 50

    def rydberg_subspace(state):
        r = np.array([[1, 0], [0, 0]])
        not_r = np.array([[0, 0], [0, 1]])
        one_r = tools.tensor_product([r, not_r, not_r])
        two_r = tools.tensor_product([r, not_r, r])

        res = np.zeros_like(state)
        res = res + qubit.left_multiply(state, [0, 1, 2], one_r, is_ket=True)
        res = res + qubit.left_multiply(state, [1, 0, 2], one_r, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], one_r, is_ket=True)
        res = res + qubit.left_multiply(state, [0, 1, 2], two_r, is_ket=True)
        return (state.T.conj() @ res)[0, 0]

    def ground_subspace(state):
        g = np.array([[0, 0], [0, 1]])
        not_g = np.array([[1, 0], [0, 0]])
        one_g = tools.tensor_product([g, not_g, not_g])
        two_g = tools.tensor_product([g, g, not_g])
        three_g = tools.tensor_product([g, g, g])

        res = np.zeros_like(state)
        res = res + qubit.left_multiply(state, [0, 1, 2], one_g, is_ket=True)
        res = res + qubit.left_multiply(state, [1, 0, 2], one_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], one_g, is_ket=True)
        res = res + qubit.left_multiply(state, [0, 1, 2], two_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 0, 1], two_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], two_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], three_g, is_ket=True)
        return (state.conj().T @ res)[0, 0]

    def mis_subspace(state):
        r = np.array([[1, 0], [0, 0]])
        not_r = np.array([[0, 0], [0, 1]])
        mis_proj = tools.tensor_product([r, not_r, r])
        res = np.zeros_like(state)
        res = res + qubit.left_multiply(state, [0, 1, 2], mis_proj, is_ket=True)
        return (state.conj().T @ res)[0, 0]

    def is_subspace(state):
        return g.IS_projector(graph, qubit) * state

    times = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10,  11, 13, 15, 17, 19, 23, 27, 35]  # , 50, 100, 200, 500, 1000]
    ground_pop = []
    cost_function = []
    mis_pop = []
    rydberg_pop = []
    for tf in times:
        print(tf)
        # Integrate the master equation
        results = schrodinger_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
        cost_function.append(rydberg.cost_function(results[-1], is_ket=True) / mis)
        mis_pop.append(np.real(mis_subspace(results[-1])))
        ground_pop.append(np.real(ground_subspace(results[-1])))
        rydberg_pop.append(np.real(rydberg_subspace(results[-1])))
    # Add initial conditions
    cost_function = [rydberg.cost_function(results[0], is_ket=True) / mis] + cost_function
    ground_pop = [np.real(ground_subspace(results[0]))] + ground_pop
    rydberg_pop = [np.real(rydberg_subspace(results[0]))] + rydberg_pop
    mis_pop = [np.real(mis_subspace(results[0]))] + mis_pop
    times = [0] + times
    plt.scatter(times, cost_function, color='teal', label='approximation ratio')
    plt.scatter(times, ground_pop, color='m', label=r'$P(n_g)\geq 1$')
    plt.scatter(times, rydberg_pop, color='g', label=r'rydberg population')

    plt.scatter(times, mis_pop, color='r', label=r'mis overlap')
    plt.plot(times, cost_function, color='teal')
    plt.plot(times, ground_pop, color='m', linestyle='--')
    plt.plot(times, rydberg_pop, color='g', linestyle='--')

    plt.plot(times, mis_pop, color='r', linestyle='--')

    plt.legend()
    plt.xlabel(r'Total annealing time $T$')
    plt.ylabel(r'Final approximation ratio')
    plt.show()


def ARvstime_adiabatic(tf=10, schedule=adiabatic_schedule, show_graph=False, n=3):
    graph, mis = line(n=n)
    if show_graph:
        nx.draw(graph)
        plt.show()
    # Generate the driving and Rydberg Hamiltonians
    rabi1 = 1
    laser1 = hamiltonian.HamiltonianLaser(transition=(0, 1), energy=rabi1, code=qubit)
    rabi2 = 1
    laser2 = hamiltonian.HamiltonianLaser(pauli='Z', transition=(0, 1), energy=rabi2, code=qubit)
    rydberg_energy = 50
    rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=qubit)

    # Initialize Schrodinger equation
    schrodinger_equation = SchrodingerEquation(hamiltonians=[laser1, laser2, rydberg])

    # Begin with all qubits in the ground state
    psi = np.zeros((qubit.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
    psi[-1, -1] = 1
    # Generate annealing schedule
    num = 50

    def rydberg_subspace(state):
        r = np.array([[1, 0], [0, 0]])
        not_r = np.array([[0, 0], [0, 1]])
        one_r = tools.tensor_product([r, not_r, not_r])
        two_r = tools.tensor_product([r, not_r, r])

        res = np.zeros_like(state)
        res = res + qubit.left_multiply(state, [0, 1, 2], one_r, is_ket=True)
        res = res + qubit.left_multiply(state, [1, 0, 2], one_r, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], one_r, is_ket=True)
        res = res + qubit.left_multiply(state, [0, 1, 2], two_r, is_ket=True)
        return (state.T.conj() @ res)[0, 0]

    def ground_subspace(state):
        g = np.array([[0, 0], [0, 1]])
        not_g = np.array([[1, 0], [0, 0]])
        one_g = tools.tensor_product([g, not_g, not_g])
        two_g = tools.tensor_product([g, g, not_g])
        three_g = tools.tensor_product([g, g, g])

        res = np.zeros_like(state)
        res = res + qubit.left_multiply(state, [0, 1, 2], one_g, is_ket=True)
        res = res + qubit.left_multiply(state, [1, 0, 2], one_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], one_g, is_ket=True)
        res = res + qubit.left_multiply(state, [0, 1, 2], two_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 0, 1], two_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], two_g, is_ket=True)
        res = res + qubit.left_multiply(state, [2, 1, 0], three_g, is_ket=True)
        return (state.conj().T @ res)[0, 0]

    def mis_subspace(state):
        r = np.array([[1, 0], [0, 0]])
        not_r = np.array([[0, 0], [0, 1]])
        mis_proj = tools.tensor_product([r, not_r, r])
        res = np.zeros_like(state)
        res = res + qubit.left_multiply(state, [0, 1, 2], mis_proj, is_ket=True)
        return (state.conj().T @ res)[0, 0]

    def is_subspace(state):
        return g.IS_projector(graph, qubit) * state

    # Integrate the master equation
    results = schrodinger_equation.run_ode_solver(psi, 0, tf, num, schedule=lambda t: schedule(t, tf))
    cost_function = [rydberg.cost_function(results[i], is_ket=True) / mis for i in range(results.shape[0])]
    mis_pop = [np.real(mis_subspace(results[i])) for i in range(results.shape[0])]
    ground_pop = [np.real(ground_subspace(results[i])) for i in range(results.shape[0])]
    rydberg_pop = [np.real(rydberg_subspace(results[i]))for i in range(results.shape[0])]
    times = np.linspace(0, tf, num)

    plt.plot(times, cost_function, color='teal', label='approximation ratio')
    plt.plot(times, ground_pop, color='m', linestyle='--', label=r'$P(n_g)\geq 1$')
    plt.plot(times, rydberg_pop, color='g', linestyle='--', label=r'rydberg population')
    plt.plot(times, mis_pop, color='r', linestyle='--', label=r'mis overlap')

    plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()


#ARvsAT_line(n=3)
#ARvsAT_adiabatic()
#ARvstime_degree(tf=5)
ARvstime_line(tf=1000, n=3, detailed=True, num=5000)
#SSvstime_line(n=3)
#SSvstime_STIRAP()
#ARvstime_line(tf=100, n=2)

#STIRAP()
#lindblad_eigenvalues(k=6)
psi = np.zeros((rydberg_EIT.d ** 2, 1), dtype=np.complex128)
psi[-1, -1] = 1
psi = tools.outer_product(psi, psi)


lindblad_eigenvalues(state=None, k=20, coefficients=[[.3, .7, 1], [1]], n=2)



def adiabatic_spectrum():
    def adiabatic_hamiltonian(t, tf):
        graph, mis = line(n=3)
        ham = np.zeros((2 ** 3, 2 ** 3))
        coefficients = adiabatic_schedule(t, tf)
        print(coefficients)
        rydberg_energy = 50
        rydberg = hamiltonian.HamiltonianRydberg(graph, energy=rydberg_energy, code=qubit)
        ham = ham + coefficients[1] * tools.tensor_product([qubit.Z, np.identity(2), np.identity(2)])
        ham = ham + coefficients[1] * tools.tensor_product([np.identity(2), qubit.Z, np.identity(2)])
        ham = ham + coefficients[1] * tools.tensor_product([np.identity(2), np.identity(2), qubit.Z])
        ham = ham + coefficients[0] * tools.tensor_product([qubit.X, np.identity(2), np.identity(2)])
        ham = ham + coefficients[0] * tools.tensor_product([np.identity(2), qubit.X, np.identity(2)])
        ham = ham + coefficients[0] * tools.tensor_product([np.identity(2), np.identity(2), qubit.X])
        ham = ham + np.diag(rydberg.hamiltonian.T[0])
        return np.linalg.eig(ham)

    times = np.arange(0, 1, .01)
    eigvals = np.zeros((len(times), 8))
    for i in range(len(times)):
        eigval, eigvec = adiabatic_hamiltonian(times[i], 1)
        eigvals[i] = eigval.real
        for j in range(8):
            plt.scatter(times[i], eigvals[i, j], c='k', s=4)
    print(eigval, eigvec.T)
    plt.show()

