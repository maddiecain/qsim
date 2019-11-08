import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plot
from qsim import noise_models
from qsim.qaoa import simulate, variational_parameters
from qsim import tools
from qsim.state import *

# Construct a simple graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
# nx.draw_networkx(G)
# Uncomment to visualize graph
# plot.draw_graph(G)

p = 1
sim_code = simulate.SimulateQAOA(G, p, 2, is_ket=False, code=TwoQubitCode)
sim = simulate.SimulateQAOA(G, p, 2, is_ket=False)

# Set the default variational operators
sim.variational_params = [variational_parameters.HamiltonianC(sim.C),
                          variational_parameters.HamiltonianB()]
sim_code.variational_params = [variational_parameters.HamiltonianC(sim_code.C),
                               variational_parameters.HamiltonianB()]

sim.noise = [noise_models.AmplitudeDampingNoise(.05), noise_models.AmplitudeDampingNoise(.05)]
sim_code.noise = [noise_models.AmplitudeDampingNoise(.025), noise_models.AmplitudeDampingNoise(.025)]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, code=TwoQubitCode)
# Set the default variational operators with X^\otimesN
sim_penalty.variational_params = [variational_parameters.HamiltonianC(sim_penalty.C),
                                  variational_parameters.HamiltonianB(),
                                  variational_parameters.HamiltonianPauli(tools.SIGMA_X_IND)]
sim_penalty.noise = [noise_models.AmplitudeDampingNoise(.025), noise_models.Noise(),
                     noise_models.AmplitudeDampingNoise(.025)]

sim_code.find_parameters_brute(n=20)
sim.find_parameters_brute(n=20)
sim_penalty.find_parameters_brute(n=20)

