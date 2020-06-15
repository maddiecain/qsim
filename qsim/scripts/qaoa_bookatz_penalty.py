import networkx as nx
from qsim.dissipation import quantum_channels
from qsim.qaoa import simulate
from qsim import hamiltonian
from qsim.state import TwoQubitCode, JordanFarhiShor

# Construct a known graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
# nx.draw_networkx(G)
# Uncomment to visualize graph
# plot.draw_graph(G)
p = 1
print('No logical encoding:')
sim = simulate.SimulateQAOA(G, p, 2, is_ket=False, mis=True)
# Set the default variational operators
sim.hamiltonian = [hamiltonian.HamiltonianC(G, mis=True),
                   hamiltonian.HamiltonianB()]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, mis=True)
# Set the default variational operators with a penalty Hamiltonian
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, mis=True),  hamiltonian.HamiltonianBookatzPenalty(),
                           hamiltonian.HamiltonianB()]
sim.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0))]
sim_penalty.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.QuantumChannel(), quantum_channels.PauliChannel((.025, 0, 0))]

# You should get the same thing
sim.find_parameters_brute(n=20)
sim_penalty.find_parameters_brute(n=20)
print('Two qubit code:')
sim_code = simulate.SimulateQAOA(G, p, 2, is_ket=False, code=TwoQubitCode, mis=True)

# Set the default variational operators
sim_code.hamiltonian = [hamiltonian.HamiltonianC(G, code=TwoQubitCode),
                               hamiltonian.HamiltonianB()]

sim_code.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0))]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, code=TwoQubitCode)

# Set the default variational parameters and dissipation
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, code=TwoQubitCode),
                                  hamiltonian.HamiltonianB(),
                                  hamiltonian.HamiltonianBookatzPenalty()]
sim_penalty.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0)),
                     quantum_channels.QuantumChannel()]

# Find optimal parameters via brute force search
sim_code.find_parameters_brute(n=10)
sim_penalty.find_parameters_brute(n=10)
print('Jordan-Farhi-Shor code:')
sim_code = simulate.SimulateQAOA(G, p, 2, is_ket=False, code=JordanFarhiShor, mis=True)

# Set the default variational operators
sim_code.hamiltonian = [hamiltonian.HamiltonianC(G, code=JordanFarhiShor),
                               hamiltonian.HamiltonianB()]

sim_code.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0))]

sim_penalty = simulate.SimulateQAOA(G, p, 3, is_ket=False, code=JordanFarhiShor)

# Set the default variational parameters and dissipation
sim_penalty.hamiltonian = [hamiltonian.HamiltonianC(G, code=JordanFarhiShor), hamiltonian.HamiltonianBookatzPenalty(),
                                  hamiltonian.HamiltonianB()]
sim_penalty.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0)),
                     quantum_channels.QuantumChannel()]
sim_code.find_parameters_brute(n=10)
#sim_penalty.find_parameters_brute(n=5)
