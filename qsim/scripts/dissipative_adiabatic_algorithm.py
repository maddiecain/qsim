import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.state import qubit, state_tools
from qsim.lindblad_master_equation import LindbladMasterEquation
import numpy as np
from qsim import tools
import matplotlib.pyplot as plt
from matplotlib import cm


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


def line(n=3):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(i, i + 1, 1) for i in range(n - 1)])
    return graph, 2

def two_triangle():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1), (0, 2, 1), (2, 3, 1), (1, 3, 1), (0, 3, 1)])
    return graph, 2


graph, mis = two_triangle()
"""nx.draw(graph)
plt.show()"""
# Generate the driving and Rydberg Hamiltonians

# Initialize spontaneous emission
IS_penalty = 20
mis_dissipation = lindblad_operators.MISDissipation(graph, IS_penalty=IS_penalty, code=qubit)
hadamard_dissipation = lindblad_operators.Hadamard_Dissipation(code=qubit)
rydberg = hamiltonian.HamiltonianRydberg(graph, energy=IS_penalty, code=qubit)
# Initialize master equation
master_equation = LindbladMasterEquation(jump_operators=[mis_dissipation])

# Begin with all qubits in the ground state
psi = tools.equal_superposition(graph.number_of_nodes())
psi = tools.outer_product(psi, psi)

dt = 0.1


def ARvstime(tf=10, schedule=lambda t, tf: [[], [t / tf, (tf - t) / tf]]):
    def ground_overlap(state):
        g = np.array([[0,0], [0,1]])
        op = tools.tensor_product([g]*graph.number_of_nodes())
        return np.real(np.trace(op @ state))
    t_cutoff = 5
    def step_schedule(t, tf):
        if t<t_cutoff:
            return [[], [t / tf, (tf - t) / tf]]
        else:
            return [[], [1, 0]]
    times = np.arange(0, tf, dt)
    # Integrate the master equation
    results = master_equation.run_ode_solver(psi, 0, tf, dt, schedule=lambda t: schedule(t, tf))
    cost_function = [rydberg.cost_function(results[i] / mis, is_ket=False) for i in range(results.shape[0])]
    tools.is_valid_state(results[-1])
    plt.plot(times, cost_function, color='teal', label='approximation ratio')

    print(results[-1], cost_function[-1])
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel(r'Annealing time $t$')
    plt.ylabel(r'Approximation ratio')
    plt.show()

def lindblad_steady_state():
    eigvals, eigvecs = master_equation.steady_state(psi)
    print(eigvals, eigvecs)
    eigvecs = np.reshape(eigvecs, psi.shape)
    eigvecs = eigvecs/np.trace(eigvecs)
    print(tools.is_valid_state(eigvecs))
    print(rydberg.cost_function(eigvecs, is_ket=False)/mis)

ARvstime()
#psi = lindblad_steady_state()