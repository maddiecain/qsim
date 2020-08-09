import networkx as nx
from qsim.evolution import lindblad_operators, hamiltonian
from qsim.codes import qubit, quantum_state
from qsim.lindblad_master_equation import LindbladMasterEquation
import numpy as np
from qsim import tools
from qsim.codes.quantum_state import State
import matplotlib.pyplot as plt
from matplotlib import cm




class MISDissipation(object):
    def __init__(self, graph: nx.Graph, IS_penalty=1, code=None):
        self.graph = graph
        if code is None:
            code = qubit
        self.code = code
        self.N = self.graph.number_of_nodes()
        self.IS_penalty = IS_penalty
        # Construct jump operators for edges and nodes
        # node =  (np.identity(self.code.d * self.code.n) - self.code.Z) / 2
        # edge = self.IS_penalty * tools.tensor_product([self.code.U, self.code.U])
        node = self.code.X @ (np.identity(self.code.d * self.code.n) - self.code.Z) / 2
        edge = self.IS_penalty * (
                tools.tensor_product([self.code.X, np.identity(self.code.d * self.code.n)]) + tools.tensor_product(
            [np.identity(self.code.d * self.code.n), self.code.X])) @ tools.tensor_product(
            [self.code.U, self.code.U]) / 2
        self.jump_operators = [node, edge]

    def liouvillian(self, state, apply_to=None, is_ket=True):
        if apply_to is None:
            # Apply to all physical qubits
            apply_to_nodes = list(self.graph.nodes)
            apply_to_edges = list(self.graph.edges)
            apply_to = (apply_to_nodes, apply_to_edges)
        a = np.zeros(state.shape)
        for j in range(2):
            for i in range(len(apply_to[j])):
                a = a + self.code.multiply(state, apply_to[j][i], self.jump_operators[j], is_ket=is_ket) - \
                    1 / 2 * self.code.left_multiply(state, apply_to[j][i],
                                                    self.jump_operators[j].T @ self.jump_operators[j], is_ket=is_ket) - \
                    1 / 2 * self.code.right_multiply(state, apply_to[j][i],
                                                     self.jump_operators[j].T @ self.jump_operators[j], is_ket=is_ket)
        return a


class Hadamard_Dissipation(object):
    def __init__(self, code=None):
        if code is None:
            code = qubit
        self.code = code
        self.jump_operators = [self.code.Z @ (np.identity(self.code.d * self.code.n) - self.code.X) / 2]

    def liouvillian(self, state: State, apply_to=None):
        if apply_to is None:
            # Apply to all physical qubits
            apply_to = list(range(state.number_physical_qudits))
        out = State(np.zeros(state.shape), is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)
        for i in range(len(apply_to)):
            out = out + self.code.multiply(state, [apply_to[i]], self.jump_operators[0]) - \
                1 / 2 * self.code.left_multiply(state, [apply_to[i]],
                                                self.jump_operators[0].T @ self.jump_operators[0]) - \
                1 / 2 * self.code.right_multiply(state, [apply_to[i]],
                                                 self.jump_operators[0].T @ self.jump_operators[0])
        return out

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
hadamard_dissipation = Hadamard_Dissipation(code=qubit)
rydberg = hamiltonian.HamiltonianMIS(graph, energies=[IS_penalty], code=qubit)
# Initialize master equation
master_equation = LindbladMasterEquation(jump_operators=[mis_dissipation])

# Begin with all qubits in the ground codes
psi = tools.equal_superposition(graph.number_of_nodes())
psi = State(tools.outer_product(psi, psi))

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
    cost_function = [rydberg.cost_function(results[i] / mis) for i in range(results.shape[0])]
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
    print(rydberg.cost_function(eigvecs)/mis)

#ARvstime()
#psi = lindblad_steady_state()