import numpy as np
import networkx as nx
from qsim.dissipation import lindblad_operators
from qsim.qaoa.simulate import SimulateQAOA
from qsim import master_equation
from qsim.tools import tools
import matplotlib.pyplot as plt

def penalty(Ep, N):
    # Energy penalty on one qubit
    hp = Ep * np.identity(2 ** JordanFarhiShor.n) - tools.outer_product(JordanFarhiShor.basis[0],
                                                                          JordanFarhiShor.basis[0]) + \
           tools.outer_product(JordanFarhiShor.basis[1], JordanFarhiShor.basis[1])
    h = Ep * np.identity(2 ** JordanFarhiShor.n) - tools.outer_product(JordanFarhiShor.basis[0],
                                                                          JordanFarhiShor.basis[0]) + \
           tools.outer_product(JordanFarhiShor.basis[1], JordanFarhiShor.basis[1])
    for i in range(N-1):
        h = tools.tensor_product([h, hp])
    return h


# Construct a simple graph
G = nx.random_regular_graph(1, 2)
for e in G.edges:
    G[e[0]][e[1]]['weight'] = 1
fig = plt.figure()
qaoa = SimulateQAOA(G, 1, 2, is_ket=False, code=JordanFarhiShor)
psi0 = tools.equal_superposition(qaoa.N, basis=JordanFarhiShor.basis)
psi0 = tools.outer_product(psi0, psi0)
s = JordanFarhiShor(psi0, qaoa.N, is_ket=qaoa.is_ket)
h = lambda t: tools.identity(qaoa.N * JordanFarhiShor.n) * qaoa.C + penalty(1, qaoa.N)
m = master_equation.MasterEquation(h, lindblad_operators.LindbladNoise())
res = m.run_ode_solver(s.state, 0, 1)
for Ep in np.linspace(1, 100, 10):
    times = np.linspace(0, 1, num=10)
    m_error = master_equation.MasterEquation(h, lindblad_operators.DepolarizingNoise(.05))
    res_error = m_error.run_ode_solver(s.state, 0, 1)
    fidelity = np.trace(np.transpose(res.conj(), axes = [0, 2, 1])@res_error, axis1=1, axis2=2)

    plt.title('|+> in JFS code under energy penalty and depolarizing dissipation')
    plt.xlabel('Time')
    plt.ylabel('Fidelity with noiseless state with dissipation probability 0.05')
    print(str(Ep))
    plt.plot(times, fidelity, label = 'Ep:'+str(Ep))
fidelity = np.trace(np.transpose(res.conj(), axes = [0, 2, 1])@res, axis1=1, axis2=2)
plt.plot(times, fidelity, label='Ep:' + str(Ep))
plt.legend()

plt.show()
