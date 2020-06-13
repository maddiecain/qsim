import networkx as nx
from qsim.noise import noise_models
from qsim.qaoa import simulate
from qsim import hamiltonian
from qsim.state.state import TwoQubitCode

# Construct a regular graph with v nodes and degree d
v = 2
d = 1
G = nx.random_regular_graph(d, v)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1

# Uncomment to visualize graph
# plot.draw_graph(G)

p = 1
sim_code = simulate.SimulateQAOA(G, p, 2, is_ket=False, code=TwoQubitCode)

# Set the default variational operators
sim_code.hamiltonian = [hamiltonian.HamiltonianC(G, code=TwoQubitCode),
                               hamiltonian.HamiltonianB()]

sim_code.noise = [noise_models.PauliNoise((.025, 0, 0)), noise_models.PauliNoise((.025, 0, 0))]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, code=TwoQubitCode)

# Set the default variational parameters and noise
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, code=TwoQubitCode),
                                  hamiltonian.HamiltonianB(),
                                  hamiltonian.HamiltonianBookatzPenalty()]
sim_penalty.noise = [noise_models.PauliNoise((.025, 0, 0)), noise_models.PauliNoise((.025, 0, 0)),
                     noise_models.LindbladNoise()]

# Find optimal parameters via brute force search
sim_code.find_parameters_brute(n=3)
sim_penalty.find_parameters_brute(n=3)
