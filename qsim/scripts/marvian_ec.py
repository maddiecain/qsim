import numpy as np
import networkx as nx
from qsim.noise import schrodinger_equation
from qsim.scripts import plot
from matplotlib import cm
from qsim.state import *
from qsim.tools import tools, operations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as sp
from matplotlib import rc
import seaborn

"""Model the Hamiltonians H_B and H_C under the presence of a penalty Hamiltonian H_P.
"""
# Make graphs pretty
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
np.set_printoptions(threshold=np.inf)


def Xi(i, N):
    # Returns the Hamiltonian corresponding to X-
    # i is ith physical qubit, assumed to be a B-type qubit. An L-type qubit is assumed to be the next physical qubit
    # N is the total number of physical qubits
    assert i % 3 == 1
    return tools.tensor_product([tools.identity(i - 1), tools.X(), tools.X(), tools.identity(N - i - 1)])


def Zi(i, N):
    assert i % 3 == 1
    # i is ith physical qubit, assumed to be a B-type qubit. An R-type qubit is assumed to be the next physical qubit
    # N is the total number of physical qubits
    return tools.tensor_product([tools.identity(i), tools.Z(), tools.Z(), tools.identity(N - i - 2)])


def XiXj(i, j, N):
    assert (i < j)
    assert i % 3 == 1
    assert j % 3 == 1
    # i and j are physical qubit indices, assumed to be B-type qubits
    # N is the total number of physical qubits
    return tools.tensor_product(
        [tools.identity(i), tools.X(), tools.identity(j - i - 1), tools.X(), tools.identity(N - j - 1)])


def ZiZj(i, j, N):
    assert i % 3 == 1
    assert j % 3 == 1
    # i and j are physical qubit indices, assumed to be B-type qubits
    # N is the total number of physical qubits
    return tools.tensor_product(
        [tools.identity(i), tools.Z(), tools.identity(j - i - 1), tools.Z(), tools.identity(N - j - 1)])


def Hp(N: list, Ep=1):
    # Two by two geometry (can be generalized in the future)
    [Nx, Ny] = N
    N = int(Nx * Ny * 3)
    hp = np.zeros([2 ** N, 2 ** N])
    for i in range(int(Nx * Ny)):
        # Add gauge interactions within a single logical qubit
        hp = hp + tools.tensor_product(
            [tools.identity(i * 3), tools.Z(), tools.Z(), tools.identity(N - i * 3 - 2)]) + tools.tensor_product(
            [tools.identity(i * 3 + 1), tools.X(), tools.X(), tools.identity(N - i * 3 - 3)])
    # Between rows
    for j in range(Ny):
        for k in range(Nx):
            # Need to deal with edge effects
            # Add gauge interactions within a single logical qubit
            if j != Ny - 1:
                # Along the same row
                hp = hp + tools.tensor_product(
                    [tools.identity(j * Nx * 3 + k * 3), tools.X(), tools.identity(2), tools.X(),
                     tools.identity(N - (j * Nx * 3 + k * 3) - 4)]) + \
                     tools.tensor_product(
                         [tools.identity(j * Nx * 3 + k * 3 + 2), tools.Z(), tools.identity(2), tools.Z(),
                          tools.identity(N - (j * Nx * 3 + k * 3 + 2) - 4)])
                # Along the same column
            if k != Nx - 1:
                hp = hp + tools.tensor_product(
                    [tools.identity(j * Nx * 3 + k * 3), tools.X(), tools.identity(3 * Nx), tools.X(),
                     tools.identity(N - (j * Nx * 3 + k * 3) - 3 * Nx - 2)]) + \
                     tools.tensor_product(
                         [tools.identity(j * Nx * 3 + k * 3 + 2), tools.Z(), tools.identity(3 * Nx), tools.Z(),
                          tools.identity(N - (j * Nx * 3 + k * 3 + 2) - 3 * Nx - 2)])
    return -1 * hp * Ep



G = nx.Graph()
G.add_nodes_from([0, 1])
G.add_edges_from([(0, 1)])
# Uncomment to draw graph
# plot.draw_graph(G)

# G.add_nodes_from([0, 1, 2, 3])
# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
# Assume logical qubit layout is a Ny x Nx grid
Nx = 1
Ny = 2
N = [Nx, Ny]

hp = Hp(N)

# Pick a ground state of the penalty Hamiltonian
# Hfield(0.5)
# basis = [np.array([[1, 1, 0, 0, 0, 0, 1, 1]]).T / 2, np.array([[0, 0, 1, 1, 1, 1, 0, 0]]).T / 2]
# psi0 = (basis[0] + basis[1]) / np.sqrt(2)
eigval, eigvec = np.linalg.eig(hp)
eigvec = eigvec.T
print(sorted(eigval))
gss = eigvec[np.isclose(eigval, np.min(eigval))]
gss = np.linalg.qr(gss.T)[0].T
proj = np.zeros([2 ** (Nx * Ny * 3), 2 ** (Nx * Ny * 3)])
for i in range(gss.shape[0]):
    gssi = np.array([gss[i]]).T
    proj = proj + tools.outer_product(gssi, gssi)
print('Is projector?', tools.is_projector(proj))
gs = np.array([eigvec[np.where(eigval == np.min(eigval))[0][0]]])
gs = gs.T / np.linalg.norm(gs)
psi0 = tools.outer_product(gs, gs)

def Hb(N):
    assert N % 3 == 0
    hb = np.zeros([2 ** N, 2 ** N])
    for i in range(N // 3):
        hb = hb + Xi(3 * i + 1, N)
    return hb


def Hc(N, G):
    # MIS Hamiltonian
    assert N % 3 == 0
    hc = np.zeros([2 ** N, 2 ** N])
    for i in range(N // 3):
        hc = hc + Zi(3 * i + 1, N)
    # Two body interaction for each edge
    # Gotta figure out the coefficient rescaling!
    for (i, j) in G.edges():
        # Assumes nodes are zero indexed
        gij = (proj @ tools.tensor_product([tools.identity(3 * i+2), tools.SZ, tools.identity(3*j-3*i-1), tools.SZ, tools.identity(N-3*j-3*i-3)]) @ proj)#/proj
        gij = np.max(np.linalg.eig(gij)[0])/np.max(np.linalg.eig(proj)[0])
        hc = hc + 1/gij * ZiZj(3 * i + 1, 3 * j + 1, N)
    return hc


def Hfield_static(vec, N):
    # N is the number of physical qubits
    # Generate random B-field for each atom
    # B vectors
    hfield = np.zeros([2 ** N, 2 ** N])
    for i in range(N):
        hfield = hfield + tools.tensor_product(
            [tools.identity(i), tools.SX * vec[i][0] + tools.SY * vec[i][1] + tools.SZ * vec[i][2],
             tools.identity(N - i - 1)])
    return hfield



# Example hamiltonian
hb = Hb(Nx * Ny * 3)
hc = Hc(Nx * Ny * 3, G)
# B field orientation
print('B field:')
vec = np.random.randn(Nx * Ny * 3, 3)
vec = np.array([[0.96008644, -0.51543102, -0.13664101],
                [0.40128861, -0.44023774, 1.0866038],
                [-1.51436961, -0.56834696, 0.29103908],
                [2.31072175, 0.12211733, 0.81743674],
                [-0.95217354, -1.14535345, -0.52514697],
                [0.03665265, -0.45178826, -0.29247424]])
print(vec)
hfield = Hfield_static(vec, Nx * Ny * 3)
# print(hp)
# print('Here:')
# print(hp)
# print(hfield)
# Frequencies
freq = np.random.randn(1) * 5
freq = [-0.31849525]
print(freq)
# freq = np.random.randn(Nx * Ny * 3, 1) * .01
# print(vec, freq)
t0 = 0
tf = 50
dt = 0.005


def Hfield(t):
    # TODO: speed this up
    global hfield
    global freq
    return hfield * np.cos(freq)



s = State(psi0, Nx * Ny * 3, is_ket=False)
print('Running clean state')
eq = schrodinger_equation.SchrodingerEquation(lambda t: hc)#tools.identity(Nx * Ny * 3))
res = eq.run_ode_solver(s.state, t0, tf, dt=dt)


def final_fidelity(Ep, fieldmag, freq, error='tracenorm', gs_proj=True):
    global res
    global proj
    vec = np.random.randn(Nx * Ny * 3, 3)
    vec = vec / np.linalg.norm(vec) * fieldmag
    hfield = Hfield_static(vec, Nx * Ny * 3)

    # Numerically integrates the Schrodinger equation
    eq_error = schrodinger_equation.SchrodingerEquation(lambda t:  Ep * hp + hfield * np.cos(freq * t))
    print('Running error state with Ep = ' + str(Ep))
    print('Frequency = ', str(freq))
    print('Field Magnitude =', str(fieldmag))
    res_error = eq_error.run_ode_solver(s.state, t0, tf, dt=dt)
    # fidelity = np.array([tools.trace(res[i] @ res_error[i]) for i in range(res.shape[0])])
    if error == 'tracenorm':
        if not gs_proj:
            fidelity = tools.trace(res[-1] @ res_error[-1])
        if gs_proj:
            fidelity = tools.trace(proj @ res_error[-1])
    if error == 'fidelity':
        if not gs_proj:
            fidelity = tools.fidelity(res[-1], res_error[-1])
        if gs_proj:
            fidelity = tools.fidelity(proj, res_error[-1])
    print('Fidelity = ', np.real(fidelity))
    return np.real(fidelity)


def full_fidelity(Ep, fieldmag, freq, error='tracenorm', gs_proj=True):
    global res
    global proj
    #vec = np.random.randn(Nx * Ny * 3, 3)
    #vec = vec / np.linalg.norm(vec) * fieldmag
    hfield = Hfield_static(vec, Nx * Ny * 3)

    # Numerically integrates the Schrodinger equation
    eq_error = schrodinger_equation.SchrodingerEquation(lambda t:  Ep * hp + hfield * np.cos(freq * t))
    print('Running error state with Ep = ' + str(Ep))
    print('Frequency = ', str(freq))
    print('Field Magnitude =', str(fieldmag))
    res_error = eq_error.run_ode_solver(s.state, t0, tf, dt=dt)
    # fidelity = np.array([tools.trace(res[i] @ res_error[i]) for i in range(res.shape[0])])
    if error == 'tracenorm':
        if not gs_proj:
            fidelity = np.array([tools.trace(res[i] @ res_error[i]) for i in range(res.shape[0])])
        if gs_proj:
            fidelity = np.array([tools.trace(proj @ res_error[i]) for i in range(res.shape[0])])
    if error == 'fidelity':
        if not gs_proj:
            fidelity = np.array([tools.fidelity(res[i], res_error[i]) for i in range(res.shape[0])])
        if gs_proj:
            fidelity = np.array([tools.fidelity(proj, res_error[i]) for i in range(res.shape[0])])
    return fidelity


def timeplot(Ep, fieldmag, freq, error='tracenorm', gs_proj=True):
    # Plots fidelity versus time
    plt.plot(np.linspace(t0, tf, int((tf - t0) / dt)), full_fidelity(Ep, fieldmag, freq, error='tracenorm', gs_proj=True))
    plt.show()


timeplot(10, 1, freq[0], gs_proj = True)
#timeplot(10, 1, np.pi/2, gs_proj = True)
#timeplot(50, 1, np.pi/2, gs_proj = True)

# Ep = [0, 10, 50, 100]
Ep = [50, 100]
# colors = ['b', 'g', 'r', 'c']
fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
fieldmag = np.linspace(0, 10, 10)
freq = np.linspace(0, np.pi, 10)
for ep in Ep:
    fidelities = np.zeros([fieldmag.shape[0], freq.shape[0]])
    for j in range(fieldmag.shape[0]):
        for k in range(freq.shape[0]):
            # Average over 10 points
            l = 2
            fs = np.zeros(l)
            for i in range(l):
                fs[i] = final_fidelity(ep, fieldmag[j], freq[k])
            print(fs)
            fidelities[k][j] = np.mean(fs)
            print(fidelities[k][j])

    fieldmag, freq = np.meshgrid(fieldmag, freq)
    print(fieldmag.shape, freq.shape, fidelities.shape)
    #urf = ax.scatter(fieldmag, freq, fidelities, cmap=cm.coolwarm, s=20)
    surf = plt.imshow(fidelities, vmin=0, vmax=1, interpolation='none')
    plt.title(r'tr$\left(\Pi_0\rho_{field}\right)$ at $t = $' + str(tf) + r' with $E_p = $' + str(ep) + ' and random inhomogeneous magnetic field')
    # plt.xlabel(r'$t$')
    ax.set_xlabel(r'$|\vec{B}|$')
    ax.set_ylabel(r'$\omega$')
    # plt.ylabel(r'tr$\left(\rho_{ideal}\rho_{field}\right)$')
    #ax.set_zlabel(r'tr$\left(\Pi_0\rho_{field}\right)$ at $t = $' + str(tf))
    # plt.ylabel(r'tr$\left(\sqrt{\sqrt{\Pi_0} \rho_{field} \sqrt{\Pi_0} }\right)^2$')
    # plt.ylabel(r'tr$\left(\sqrt{\sqrt{\rho_{ideal}} \rho_{field} \sqrt{\rho_{ideal}} }\right)^2$')
    # plt.text(0, 0, r'$H = H_P$' + '\n' + r' $|\bar{B}| \simeq 1/2$')
    # plt.plot(times, fidelity, label=r'$E_p$:' + str(Ep))
    #plt.legend()
    fig.colorbar(surf)#, shrink=0.5, aspect=5)
    plt.savefig('plot_ep'+str(ep))

plt.show()
