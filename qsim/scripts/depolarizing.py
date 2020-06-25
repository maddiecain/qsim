import numpy as np
import networkx as nx
from qsim.state import jordan_farhi_shor
from qsim.evolution import quantum_channels
from qsim.evolution.hamiltonian import HamiltonianBookatzPenalty
from qsim.graph_algorithms.qaoa import SimulateQAOA
from qsim import lindblad_master_equation
from qsim.tools import tools
import matplotlib.pyplot as plt

"""Plot the fidelity of a noiseless state with a Jordan-Farhi-Shor state under an energy penalty Hamiltonian,
under the presence of infinite-bandwidth depolarizing noise."""

"""def penalty(Ep, N):
    # Energy penalty on one qubit
    hp = Ep * np.identity(2 ** JordanFarhiShor.n) - tools.outer_product(JordanFarhiShor.basis[0],
                                                                          JordanFarhiShor.basis[0]) + \
           tools.outer_product(JordanFarhiShor.basis[1], JordanFarhiShor.basis[1])
    h = Ep * np.identity(2 ** JordanFarhiShor.n) - tools.outer_product(JordanFarhiShor.basis[0],
                                                                          JordanFarhiShor.basis[0]) + \
           tools.outer_product(JordanFarhiShor.basis[1], JordanFarhiShor.basis[1])
    for i in range(N-1):
        h = tools.tensor_product([h, hp])
    return h"""


# Construct a simple graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
fig = plt.figure()
qaoa = SimulateQAOA(G, 1, 2, is_ket=False, code=jordan_farhi_shor)
psi0 = tools.equal_superposition(qaoa.N, basis=jordan_farhi_shor.logical_basis)
psi0 = tools.outer_product(psi0, psi0)
s = psi0
#h = lambda t: tools.identity(graph_algorithms.N * JordanFarhiShor.n) * graph_algorithms.C + penalty(1, graph_algorithms.N)
m = lindblad_master_equation.MasterEquation(hamiltonians = [HamiltonianBookatzPenalty()])
res = m.run_ode_solver(s, 0, 1)
for Ep in np.linspace(1, 100, 10):
    times = np.linspace(0, 1, num=10)
    m_error = lindblad_master_equation.MasterEquation(hamiltonians = [HamiltonianBookatzPenalty()], noise_models = [quantum_channels.DepolarizingChannel(.05)])
    res_error = m_error.run_ode_solver(s, 0, 1)
    fidelity = np.trace(np.transpose(res.conj(), axes = [0, 2, 1])@res_error, axis1=1, axis2=2)

    plt.title('|+> in JFS code under energy penalty and depolarizing evolution')
    plt.xlabel('Time')
    plt.ylabel('Fidelity with noiseless state with evolution probability 0.05')
    print(str(Ep))
    plt.plot(times, fidelity, label = 'Ep:'+str(Ep))
fidelity = np.trace(np.transpose(res.conj(), axes = [0, 2, 1])@res, axis1=1, axis2=2)
plt.plot(times, fidelity, label='Ep:' + str(Ep))
plt.legend()

plt.show()
