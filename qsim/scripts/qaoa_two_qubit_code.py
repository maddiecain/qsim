import networkx as nx
from qsim.dissipation import lindblad_operators
from qsim.qaoa import simulate
from qsim import hamiltonian
from qsim.state import TwoQubitCode

# Construct a regular graph with 2 nodes and degree 1
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1

# Uncomment to visualize graph
# plot.draw_graph(G)

p = 1
sim_code = simulate.SimulateQAOA(G, p, 2, is_ket=False, code=TwoQubitCode, mis=True)

# Set the default variational operators
sim_code.hamiltonian = [hamiltonian.HamiltonianC(G, code=TwoQubitCode),
                               hamiltonian.HamiltonianB()]

sim_code.noise = [lindblad_operators.PauliNoise((.025, 0, 0)), lindblad_operators.PauliNoise((.025, 0, 0))]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, code=TwoQubitCode)

# Set the default variational parameters and dissipation
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, code=TwoQubitCode),
                                  hamiltonian.HamiltonianB(),
                                  hamiltonian.HamiltonianBookatzPenalty()]
sim_penalty.noise = [lindblad_operators.PauliNoise((.025, 0, 0)), lindblad_operators.PauliNoise((.025, 0, 0)),
                     lindblad_operators.LindbladNoise()]

# Find optimal parameters via brute force search
sim_code.find_parameters_brute(n=10)
sim_penalty.find_parameters_brute(n=10)
