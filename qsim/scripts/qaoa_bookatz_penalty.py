import networkx as nx
from qsim.evolution import quantum_channels, hamiltonian
from qsim.graph_algorithms import qaoa
from qsim.state import two_qubit_code, jordan_farhi_shor
# Construct a known graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
# nx.draw_networkx(G)
# Uncomment to visualize graph
# plot.draw_graph(G)
p = 1
print('No logical encoding:')
hc_qubit = hamiltonian.HamiltonianC(G, mis=True)
sim = qaoa.SimulateQAOA(G, p, 2, is_ket=False, C=hc_qubit)
# Set the default variational operators
sim.hamiltonian = [hc_qubit, hamiltonian.HamiltonianB()]

sim_penalty = qaoa.SimulateQAOA(G, p, 3, is_ket=False, C=hc_qubit)
# Set the default variational operators with a penalty Hamiltonian
sim_penalty.hamiltonian = [hc_qubit, hamiltonian.HamiltonianBookatzPenalty(),
                           hamiltonian.HamiltonianB()]
sim.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0))]
sim_penalty.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.QuantumChannel(), quantum_channels.PauliChannel((.025, 0, 0))]

# You should get the same thing
sim.find_parameters_brute(n=20)
sim_penalty.find_parameters_brute(n=20)
print('Two qubit code:')
hc_two_qubit_code = hamiltonian.HamiltonianC(G, code=two_qubit_code)
sim_code = qaoa.SimulateQAOA(G, p, 2, is_ket=False, code=two_qubit_code, C =hc_two_qubit_code)

# Set the default variational operators
sim_code.hamiltonian = [hc_two_qubit_code, hamiltonian.HamiltonianB(code=two_qubit_code)]

sim_code.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0))]

sim_penalty = qaoa.SimulateQAOA(G, p, 3, is_ket=False, code=two_qubit_code)

# Set the default variational parameters and evolution
sim_penalty.hamiltonian = [hc_two_qubit_code, hamiltonian.HamiltonianB(code=two_qubit_code),
                           hamiltonian.HamiltonianBookatzPenalty(code=two_qubit_code)]
sim_penalty.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0)),
                     quantum_channels.QuantumChannel()]

# Find optimal parameters via brute force search
sim_code.find_parameters_brute(n=10)
sim_penalty.find_parameters_brute(n=10)
print('Jordan-Farhi-Shor code:')
hc_jordan_farhi_shor = hamiltonian.HamiltonianC(G, code=jordan_farhi_shor)
sim_code = qaoa.SimulateQAOA(G, p, 2, is_ket=False, code=jordan_farhi_shor, C = hc_jordan_farhi_shor)

# Set the default variational operators
sim_code.hamiltonian = [hc_jordan_farhi_shor, hamiltonian.HamiltonianB(code=jordan_farhi_shor)]

sim_code.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0))]

sim_penalty = qaoa.SimulateQAOA(G, p, 3, is_ket=False, code=jordan_farhi_shor)

# Set the default variational parameters and evolution
sim_penalty.hamiltonian = [hc_jordan_farhi_shor, hamiltonian.HamiltonianBookatzPenalty(),
                           hamiltonian.HamiltonianB()]
sim_penalty.noise = [quantum_channels.PauliChannel((.025, 0, 0)), quantum_channels.PauliChannel((.025, 0, 0)),
                     quantum_channels.QuantumChannel()]
sim_code.find_parameters_brute(n=10)
#sim_penalty.find_parameters_brute(n=5)
