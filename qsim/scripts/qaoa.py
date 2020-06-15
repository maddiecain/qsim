import networkx as nx
from qsim.dissipation import lindblad_operators
from qsim.qaoa import simulate
from qsim import hamiltonian

# Construct a known graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
# nx.draw_networkx(G)
# Uncomment to visualize graph
# plot.draw_graph(G)
p = 1
sim = simulate.SimulateQAOA(G, p, 2, is_ket=False, mis=True)
# Set the default variational operators
sim.hamiltonian = [hamiltonian.HamiltonianC(G, mis=True),
                   hamiltonian.HamiltonianB()]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, mis=True)
# Set the default variational operators with a penalty Hamiltonian
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, mis=True),  hamiltonian.HamiltonianBookatzPenalty(),
                           hamiltonian.HamiltonianB()]
sim.noise = [lindblad_operators.PauliNoise((.025, 0, 0)), lindblad_operators.PauliNoise((.025, 0, 0))]
sim_penalty.noise = [lindblad_operators.PauliNoise((.025, 0, 0)), lindblad_operators.LindbladNoise(), lindblad_operators.PauliNoise((.025, 0, 0))]

# You should get the same thing
sim.find_parameters_brute(n=20)
sim_penalty.find_parameters_brute(n=20)
