import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plot
from qsim import noise_models
from qsim.qaoa import simulate, variational_parameters
from qsim import tools

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
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
# nx.draw_networkx(G)
# Uncomment to visualize graph
# plot.draw_graph(G)

p = 1
sim = simulate.SimulateQAOA(G, p, 2, is_ket=False)
# Set the default variational operators
sim.variational_params = [variational_parameters.HamiltonianC(sim.C),
                             variational_parameters.HamiltonianB()]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False)
# Set the default variational operators with X^\otimesN
sim_penalty.variational_params = [variational_parameters.HamiltonianC(sim.C),
                                     variational_parameters.HamiltonianB(),
                                     variational_parameters.HamiltonianPauli(tools.SIGMA_X_IND)]
sim.noise = [noise_models.AmplitudeDampingNoise(.05), noise_models.AmplitudeDampingNoise(.05)]
sim_penalty.noise = [noise_models.AmplitudeDampingNoise(.05), noise_models.Noise(), noise_models.AmplitudeDampingNoise(.05)]

sim_penalty.find_parameters_brute(n=20)
sim.find_parameters_brute(n=20)

# in the case of p1=1, m=2, plot the cost function for all beta, gamma
n_sample=200
alpharange=np.linspace(-1*np.pi, np.pi, n_sample)
cost_function_values=[sim_penalty.run([0.78539567, 1.17825878, alpha]) for alpha in alpharange]
plt.plot(alpharange, cost_function_values)
plt.show()
