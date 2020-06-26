import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.state import rydberg_EIT, state_tools
from qsim.lindblad_master_equation import LindbladMasterEquation
import numpy as np
from qsim import tools
import matplotlib.pyplot as plt
from matplotlib import cm

from qsim.graph_algorithms.graph import IS_projector


# Generate a simple graph
def triangular_prism():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 2, 1), (2, 3, 1), (0, 4, 1), (1, 4, 1), (3, 4, 1), (1, 5, 1), (2, 5, 1),
         (3, 5, 1)])
    return graph, 2


def triangle():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 2, 1), (1, 2, 1)])
    return graph, 1

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
    return graph, np.ceil(n/2)


graph, mis = line(n=3)
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
master_equation = LindbladMasterEquation(hamiltonians=[laser1, laser2, rydberg], jump_operators=[spontaneous_emission])
from scipy.integrate import ode

# Begin with all qubits in the ground state
psi = np.zeros((rydberg_EIT.d ** graph.number_of_nodes(), 1), dtype=np.complex128)
psi[-1,-1] = 1
psi = tools.outer_product(psi, psi)
# Generate annealing schedule
dt = .01


def ARvstime(tf=10, schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]):
    def rydberg_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        #res = np.zeros_like(state)
        #for i in range(state_tools.num_qubits(state, rydberg_EIT)):
        #    res = res + rydberg_EIT.left_multiply(state, [i], r, is_ket=False)
        return r @ state

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
        res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], two_e, is_ket=False)
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

    times = np.arange(0, tf, dt)
    excited_pop = []
    ground_pop = []
    rydberg_pop = []
    cost_function = []
    mis_pop = []
    # Integrate the master equation
    results = master_equation.run_ode_solver(psi, 0, tf, dt, schedule=lambda t: schedule(t, tf))
    for i in range(results.shape[0]):
        if not (tools.is_valid_state(results[i], verbose=False)):
            tools.is_valid_state(results[i], verbose=True)
    cost_function = [rydberg.cost_function(results[i] / mis, is_ket=False) for i in range(results.shape[0])]
    #excited_pop.append(np.real(np.trace(excited_subspace(results[-1]))))
    #rydberg_pop.append(np.real(np.trace(rydberg_subspace(results[-1]))))
    #mis_pop.append(np.real(np.trace(mis_subspace(results[-1]))))
    #ground_pop.append(np.real(np.trace(ground_subspace(results[-1]))))
    # Add initial conditions
    ground_pop = [np.real(np.trace(ground_subspace(results[i]))) for i in range(results.shape[0])]
    excited_pop = [np.real(np.trace(excited_subspace(results[i]))) for i in range(results.shape[0])]
    #rydberg_pop = [np.real(np.trace(rydberg_subspace(results[i]))) for i in range(results.shape[0])]
    #mis_pop = [np.real(np.trace(mis_subspace(results[i]))) for i in range(results.shape[0])]
    plt.plot(times, cost_function, color='teal', label='approximation ratio')
    #plt.plot(times, excited_pop, color='y', linestyle='--', label=r'$P(n_e)\geq 1$')
    #plt.plot(times, rydberg_pop, color='m', linestyle='--', label=r'$P(n_r)\geq 1$')
    #plt.plot(times, ground_pop, color='r', linestyle='--', label=r'$P(n_g)\geq 1$')
    ##plt.plot(times, mis_pop, color='r', linestyle='--', label=r'mis overlap')
    # plt.hlines(1/40, 0, max(times), linestyle='--')
    # plt.hlines(1/5, 0, max(times))
    #print(cost_function[-1], excited_pop[-1], ground_pop[-1])

    plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()


def ARvsAT(schedule=lambda t, tf: [[t / tf, (tf - t) / tf, 1], [1]]):
    def rydberg_subspace(state, n=3):
        if n == 3:
            r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            res = np.zeros_like(state)
            rrr = tools.tensor_product([r, r, r])
            #for i in range(state_tools.num_qubits(state, rydberg_EIT)):
            #    res = res + rydberg_EIT.left_multiply(state, [i], r, is_ket=False)
            return rrr @ res
        elif n == 2:
            r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            res = np.zeros_like(state)
            rrr = tools.tensor_product([r, r, r])


    def excited_subspace(state, n=3):
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
            res = res + rydberg_EIT.left_multiply(state, [1, 0, 2], two_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], two_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [2, 1, 0], three_e, is_ket=False)

            return res
        elif n==2:
            e = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            not_e = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
            one_e = tools.tensor_product([e, not_e])
            two_e = tools.tensor_product([e, e])

            res = np.zeros_like(state)
            res = res + rydberg_EIT.left_multiply(state, [0, 1], one_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [1, 0], one_e, is_ket=False)
            res = res + rydberg_EIT.left_multiply(state, [0, 1], two_e, is_ket=False)

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

    def mis_subspace(state):
        r = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        not_r = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        mis_proj = tools.tensor_product([r, not_r, r])
        return rydberg_EIT.left_multiply(state, [0, 1, 2], mis_proj, is_ket=False)

    times = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27]
    excited_pop = []
    ground_pop = []
    cost_function = []
    mis_pop = []
    rydberg_pop = []
    for tf in times:
        print(tf)
        # Integrate the master equation
        results = master_equation.run_ode_solver(psi, 0, tf, dt, schedule=lambda t: schedule(t, tf))
        #cost_function.append(rydberg.cost_function(results[-1] / mis, is_ket=False))
        #excited_pop.append(np.real(np.trace(excited_subspace(results[-1]))))
        #mis_pop.append(np.real(np.trace(mis_subspace(results[-1]))))
        #ground_pop.append(np.real(np.trace(ground_subspace(results[-1]))))
        #rydberg_pop.append(np.real(np.trace(rydberg_subspace(results[-1]))))
    # Add initial conditions
    cost_function = [rydberg.cost_function(results[0] / mis, is_ket=False)] + cost_function
    #excited_pop = [np.real(np.trace(excited_subspace(results[0])))] + excited_pop
    #ground_pop = [np.real(np.trace(ground_subspace(results[0])))] + ground_pop
    #rydberg_pop = [np.real(np.trace(rydberg_subspace(results[0])))] + rydberg_pop
    #mis_pop = [np.real(np.trace(mis_subspace(results[0])))] + mis_pop
    times = [0] + times
    for i in range(results.shape[0]):
        print(tools.is_valid_state(results[i], verbose=False))
    plt.scatter(times, cost_function, color='teal', label='approximation ratio')
    #plt.scatter(times, excited_pop, color='y', label=r'$P(n_e)\geq 1$')
    #plt.scatter(times, ground_pop, color='m', label=r'$P(n_g)\geq 1$')
    #plt.scatter(times, rydberg_pop, color='teal', label=r'rydberg population')

    #plt.scatter(times, mis_pop, color='r', label=r'mis overlap')
    plt.plot(times, cost_function, color='teal')
    #plt.plot(times, excited_pop, color='y', linestyle='--')
    #plt.plot(times, ground_pop, color='m', linestyle='--')
    #plt.plot(times, rydberg_pop, color='teal', linestyle='--')

    #plt.plot(times, mis_pop, color='r', linestyle='--')
    #plt.hlines(1 / 40, 0, max(times), linestyle='--')
    #plt.hlines(1 / 5, 0, max(times))

    plt.legend()
    plt.xlabel(r'Total annealing time $T$')
    plt.ylabel(r'Final approximation ratio')
    plt.show()


def lindblad_eigenvalues(k=5):
    eigvals, eigvecs = master_equation.eig(psi, k=k)
    # Reshape eigenvectors
    eigvecs = np.reshape(eigvecs, [psi.shape[0], psi.shape[1], eigvecs.shape[-1]])
    cost_function = [rydberg.cost_function(eigvecs[:, i]) for i in range(len(eigvals))] * 2
    eigvals_cc = eigvals.conj()
    eigvals_real = np.concatenate((eigvals.real, eigvals_cc.real))
    eigvals_complex = np.concatenate((eigvals.imag, eigvals_cc.imag))
    plt.scatter(eigvals_real, eigvals_complex, cmap='spring', c=cost_function)
    plt.hlines(0, xmin=-20, xmax=5)
    plt.vlines(0, ymin=-20, ymax=20)
    plt.colorbar()
    plt.show()


def lindblad_steady_state(k=1, coefficients=None):
    eigvals, eigvecs = master_equation.steady_state(psi, k=k, coefficients=coefficients)
    print(eigvals)
    eigvecs = np.reshape(eigvecs, [psi.shape[0], psi.shape[1], eigvecs.shape[-1]])
    eigvecs[:, :, 0] = eigvecs[:,:,0]/np.trace(eigvecs[:,:,0])
    print(eigvecs[:,:,0])
    print(tools.is_valid_state(eigvecs[:,:,0]))
    print(rydberg.cost_function(eigvecs[:,:,0], is_ket=False))

def cosine_schedule(t, tf):
    return [[1-np.cos(t / tf * np.pi/2), np.cos(t / tf * np.pi/2), 1], [1]]


def step_schedule(t, tf):
    if t < tf/2:
        return [[0,0,0],[0]]
    else:
        return [[1,0,0],[1]]


def ham_output(state, coefficients=None):
    res = np.zeros(state.shape)
    for i in range(len(master_equation.hamiltonians)):
        print(i)
        res = res - 1j * coefficients[0][i] * (master_equation.hamiltonians[i].left_multiply(state, is_ket=False) -
                                               master_equation.hamiltonians[i].right_multiply(state, is_ket=False))
        print(master_equation.hamiltonians[i].left_multiply(state, is_ket=False), master_equation.hamiltonians[i].right_multiply(state, is_ket=False))
    for i in range(len(master_equation.jump_operators)):
        res = res + coefficients[1][i] * master_equation.jump_operators[i].liouvillian(state, is_ket=False)
    return res

#print('hi')
#print(ham_output(psi, coefficients=[[1,2],[0]]))
ARvstime(tf=10)
#ARvsAT()
#lindblad_eigenvalues(k=6)
#print(lindblad_steady_state(k=1, coefficients=[[1, 0, 1], [1]]))