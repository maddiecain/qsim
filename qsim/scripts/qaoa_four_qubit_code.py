import networkx as nx
import plot
from qsim.dissipation import lindblad_operators
from qsim.qaoa import simulate
from qsim import hamiltonian
from qsim.state import JordanFarhiShor

# Construct a simple graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1

# Uncomment to visualize graph
#plot.draw_graph(G)

p = 1
sim_code = simulate.SimulateQAOA(G, p, 2, is_ket=False, code=JordanFarhiShor, mis=True)

# Set the default variational operators
sim_code.hamiltonian = [hamiltonian.HamiltonianC(G, code=JordanFarhiShor),
                               hamiltonian.HamiltonianB()]

sim_code.noise = [lindblad_operators.PauliNoise((.025, 0, 0)), lindblad_operators.PauliNoise((.025, 0, 0))]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, code=JordanFarhiShor)

# Set the default variational parameters and dissipation
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, code=JordanFarhiShor), hamiltonian.HamiltonianBookatzPenalty(),
                                  hamiltonian.HamiltonianB()]
sim_penalty.noise = [lindblad_operators.PauliNoise((.025, 0, 0)), lindblad_operators.PauliNoise((.025, 0, 0)),
                     lindblad_operators.LindbladNoise()]

sim_code.find_parameters_brute(n=10)
#sim_penalty.find_parameters_brute(n=5)
