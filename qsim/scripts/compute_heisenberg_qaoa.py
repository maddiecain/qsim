from qsim.graph_algorithms.graph import line_graph, ring_graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qsim.tools.tools import equal_superposition
from qsim.evolution import hamiltonian
from qsim.codes.quantum_state import State
from qsim.graph_algorithms import qaoa
from qsim.graph_algorithms.graph import Graph


def heisenberg_maxcut_graphs(p, graph):
    # Should be a tuple of (coefficient, term)
    # Term is a dictionary with dict[node] = (x, y, z)
    edge_to_subgraph = {}
    if p > 0:
        for edge in graph.edges:
            # Take each edge, and form its subgraph
            subgraph = nx.Graph()
            subgraph.add_edge(edge[0], edge[1])
            nodes = [edge[0], edge[1]]
            for i in range(p):
                for j in range(len(subgraph.nodes)):
                    for e in graph.edges:
                        if e not in subgraph.edges:
                            if nodes[j] == e[0]:
                                subgraph.add_edge(e[0], e[1])
                                if e[1] not in nodes:
                                    nodes.append(e[1])
                            elif nodes[j] == e[1]:
                                subgraph.add_edge(e[0], e[1])
                                if e[0] not in nodes:
                                    nodes.append(e[0])

            edge_to_subgraph[(edge[0], edge[1])] = subgraph
    return edge_to_subgraph


def compute_QAOA_cost(subgraphs, params):
    assert len(params) % 2 == 0
    p = len(params) // 2
    # Compute the cost function for each subgraph
    cf = 0
    for edge in subgraphs:
        subgraph = subgraphs[edge]
        subgraph_cost = subgraph.copy()
        subgraph = nx.relabel_nodes(subgraph,
                                    {node: i for (node, i) in zip(subgraph.nodes, range(len(subgraph.nodes)))})
        state = State(equal_superposition(len(subgraph.nodes)), IS_subspace=False, graph=subgraph)
        cost = hamiltonian.HamiltonianMaxCut(Graph(subgraph), cost_function=True)
        driver = hamiltonian.HamiltonianDriver()
        for e in subgraph_cost.edges:
            if e != edge:
                subgraph_cost[e[0]][e[1]]['weight'] = 0
            else:
                subgraph_cost[e[0]][e[1]]['weight'] = 1
        subgraph_cost = nx.relabel_nodes(subgraph_cost,
                                    {node: i for (node, i) in zip(subgraph_cost.nodes, range(len(subgraph_cost.nodes)))})

        cost_function = hamiltonian.HamiltonianMaxCut(Graph(subgraph_cost), cost_function=True)
        qa = qaoa.SimulateQAOA(graph=graph, hamiltonian=[], cost_hamiltonian=cost_function)
        hamiltonians = [cost, driver] * p
        qa.hamiltonian = hamiltonians
        result = qa.run(params, initial_state=state)
        cf += result
    return cf


def simulate_QAOA(graph, params):
    assert len(params) % 2 == 0
    p = len(params) // 2
    # Compute the cost function for each subgraph
    state = State(equal_superposition(len(graph.nodes)), IS_subspace=False, graph=graph)
    cost = hamiltonian.HamiltonianMaxCut(graph, cost_function=True)
    driver = hamiltonian.HamiltonianDriver()
    qa = qaoa.SimulateQAOA(graph=graph, hamiltonian=[], cost_hamiltonian=cost)
    hamiltonians = [cost, driver] * p
    qa.hamiltonian = hamiltonians
    result = qa.run(params, initial_state=state)
    return result

graph = nx.random_regular_graph(3, 180)
nx.draw(graph, with_labels=True)
plt.show()
graph = Graph(graph, IS=False)
subgraphs = heisenberg_maxcut_graphs(1, graph)

# Benchmark
import time
t = time.time_ns()
# Fast algorithm for computing the cost function
cf = compute_QAOA_cost(subgraphs, [2, 1])
print(cf)
print('time in seconds', (time.time_ns()-t)/1e9)
# Normal simulation
cf = simulate_QAOA(graph, [2, 2])
print(cf)