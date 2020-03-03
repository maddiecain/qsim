import networkx as nx
import numpy as np
from qsim.noise import noise_models
from qsim.qaoa import simulate, variational_parameters

# Construct a known graph
"""G = nx.Graph()

G.add_edge(0, 1, weight=1)
G.add_edge(0, 2, weight=1)
G.add_edge(2, 3, weight=1)
G.add_edge(0, 4, weight=1)
G.add_edge(1, 4, weight=1)
G.add_edge(3, 4, weight=1)
G.add_edge(1, 5, weight=1)
G.add_edge(2, 5, weight=1)
G.add_edge(3, 5, weight=1)
"""
G = nx.random_regular_graph(2, 4)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
# nx.draw_networkx(G)
# Uncomment to visualize graph
#plot.draw_graph(G)

p = 1
sim = simulate.SimulateQAOA(G, p, 2, is_ket=False)
# Set the default variational operators
sim.variational_params = [variational_parameters.HamiltonianC(sim.C),
                             variational_parameters.HamiltonianB()]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False)
# Set the default variational operators with X^\otimesN
sim_penalty.variational_params = [variational_parameters.HamiltonianC(sim.C),
                                     variational_parameters.HamiltonianB(),
                                     variational_parameters.HamiltonianPauli('X')]
sim.noise = [noise_models.PauliNoise((.025, 0, 0)), noise_models.PauliNoise((.025, 0, 0))]
sim_penalty.noise = [noise_models.PauliNoise((.025, 0, 0)), noise_models.PauliNoise((.025, 0, 0)), noise_models.LindbladNoise()]

sim_penalty.find_parameters_brute(n=20)
#sim.find_parameters_brute(n=5)
