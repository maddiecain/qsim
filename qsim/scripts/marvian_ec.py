import numpy as np
import networkx as nx
from qsim.noise import schrodinger_equation
from qsim.scripts import plot
from matplotlib import cm
from qsim.state import *
from qsim.tools import tools, operations
import matplotlib.pyplot as plt

"""Model the Hamiltonians H_B and H_C under the presence of a penalty Hamiltonian H_P.
"""

np.set_printoptions(threshold=np.inf)


class MarvianCode(object):
    def __init__(self, Nx, Ny, G=None):
        self.Nx = Nx
        self.Ny = Ny
        self.grid = [Nx, Ny]
        self.N = 3 * Nx * Ny
        self.hp = self.Hp()

        def codespace():
            # Returns a projector into the code space as well as an orthonormal basis for the code space
            eigval, eigvec = np.linalg.eig(self.hp)
            eigvec = eigvec.T
            gss = eigvec[np.isclose(eigval, np.min(eigval))]
            gss = np.linalg.qr(gss.T)[0].T
            p = np.zeros([2 ** self.N, 2 ** self.N])
            for i in range(gss.shape[0]):
                gssi = np.array([gss[i]]).T
                p = p + tools.outer_product(gssi, gssi)
            return p, gss

        self.proj, self.onb = codespace()
        if G is None:
            # G should default to a grid
            G = nx.Graph()
            for k in range(Nx*Ny):
                G.add_node(k)
            # Along rows
            for i in range(Nx-1):
                for j in range(Ny):
                    G.add_edge(j * Ny + i, j * Ny + i+1)
            # Along columns
            for i in range(Ny-1):
                for j in range(Nx):
                    G.add_edge(i * Nx + j,  (i+1) * Nx + j)
        self.G = G
        self.hb = self.Hb()
        self.hc = self.Hc()

    def X(self, i):
        # Returns the Hamiltonian corresponding to X-
        # i is ith physical qubit, assumed to be a B-type qubit. An L-type qubit is assumed to be the next physical qubit
        # N is the total number of physical qubits
        assert i % 3 == 1
        return tools.tensor_product([tools.identity(i - 1), tools.X(), tools.X(), tools.identity(self.N - i - 1)])

    def Z(self, i):
        assert i % 3 == 1
        # i is ith physical qubit, assumed to be a B-type qubit. An R-type qubit is assumed to be the next physical qubit
        # N is the total number of physical qubits
        return tools.tensor_product([tools.identity(i), tools.Z(), tools.Z(), tools.identity(self.N - i - 2)])

    def Xn(self):
        h = tools.identity(self.N)
        for i in range(self.N // 3):
            h = h @ self.X(3 * i + 1)
        return h

    def Zn(self):
        h = tools.identity(self.N)
        for i in range(self.N // 3):
            h = h @ self.Z(3 * i + 1)
        return h

    def XX(self, i, j):
        assert (i < j)
        assert i % 3 == 1
        assert j % 3 == 1
        # i and j are physical qubit indices, assumed to be B-type qubits
        # N is the total number of physical qubits
        return tools.tensor_product(
            [tools.identity(i), tools.X(), tools.identity(j - i - 1), tools.X(), tools.identity(self.N - j - 1)])

    def ZZ(self, i, j):
        assert i % 3 == 1
        assert j % 3 == 1
        # i and j are physical qubit indices, assumed to be B-type qubits
        # N is the total number of physical qubits
        return tools.tensor_product(
            [tools.identity(i), tools.Z(), tools.identity(j - i - 1), tools.Z(), tools.identity(self.N - j - 1)])

    def Hp(self):
        # Two by two geometry (can be generalized in the future)
        hp = np.zeros([2 ** self.N, 2 ** self.N])
        for i in range(int(self.Nx * self.Ny)):
            # Add gauge interactions within a single logical qubit
            hp = hp + tools.tensor_product(
                [tools.identity(i * 3), tools.Z(), tools.Z(),
                 tools.identity(self.N - i * 3 - 2)]) + tools.tensor_product(
                [tools.identity(i * 3 + 1), tools.X(), tools.X(), tools.identity(self.N - i * 3 - 3)])
        # Between rows
        for j in range(self.Ny):
            # j is the number of rows
            for k in range(self.Nx):
                # k is the number of columns
                # Need to deal with edge effects
                # Add gauge interactions within a single logical qubit
                if k != self.Nx - 1:
                    # Along the same row
                    hp = hp + tools.tensor_product(
                        [tools.identity(j * self.Nx * 3 + k * 3), tools.X(), tools.identity(2), tools.X(),
                         tools.identity(self.N - (j * self.Nx * 3 + k * 3) - 4)]) + \
                         tools.tensor_product(
                             [tools.identity(j * self.Nx * 3 + k * 3 + 2), tools.Z(), tools.identity(2), tools.Z(),
                              tools.identity(self.N - (j * self.Nx * 3 + k * 3 + 2) - 4)])
                    # Along the same column
                if j != self.Ny - 1:
                    hp = hp + tools.tensor_product(
                        [tools.identity(j * self.Nx * 3 + k * 3), tools.X(), tools.identity(3 * self.Nx-1), tools.X(),
                         tools.identity(self.N - (j * self.Nx * 3 + k * 3) - 3 * self.Nx - 1)]) + \
                         tools.tensor_product(
                             [tools.identity(j * self.Nx * 3 + k * 3 + 2), tools.Z(), tools.identity(3 * self.Nx-1),
                              tools.Z(), tools.identity(self.N - (j * self.Nx * 3 + k * 3 + 2) - 3 * self.Nx - 1)])
        return -1 * hp

    def Hc(self):
        # MIS Hamiltonian
        hc = np.zeros([2 ** self.N, 2 ** self.N])
        for i in range(self.N // 3):
            hc = hc + self.Z(3 * i + 1)
        # Two body interaction for each edge
        # Gotta figure out the coefficient rescaling!
        for (i, j) in self.G.edges():
            # Assumes nodes are zero indexed
            gij = (self.proj @ tools.tensor_product([tools.identity(3 * i + 2), tools.Z(), tools.identity(3 * j - 3 * i - 1), tools.Z(), tools.identity(self.N - 3 * j - 3)]) @ self.proj)
            gij = np.max(np.linalg.eig(gij)[0]) / np.max(np.linalg.eig(self.proj)[0])
            hc = hc + 1 / gij * self.ZZ(3 * i + 1, 3 * j + 1)
        return hc

    def Hb(self):
        hb = np.zeros([2 ** self.N, 2 ** self.N])
        for i in range(self.N // 3):
            if i == 0:
                hb = hb + 3 * self.X(3 * i + 1)
            hb = hb + self.X(3 * i + 1)
        return hb

    def ZeroL(self, is_ket=False):
        s = np.zeros([self.onb.shape[1], 1])
        for i in range(self.onb.shape[0]):
            temp = np.array([self.onb[i]]).T
            for j in range(self.N // 3):
                temp = (self.X(3 * j + 1) @ (1 / 2 * tools.identity(self.N) - 1 / 2 * self.Z(3 * j + 1)) @ self.X(
                    3 * j + 1) @ (
                                1 / 2 * tools.identity(self.N) + 1 / 2 * self.Z(3 * j + 1))) @ temp
            s = s + temp
        s = s / np.linalg.norm(s)
        if not is_ket:
            s = tools.outer_product(s, s)
        return s

    def OneL(self, is_ket=False):
        s = np.zeros([self.onb.shape[1], 1])
        for i in range(self.onb.shape[0]):
            temp = np.array([self.onb[i]]).T
            for j in range(self.N // 3):
                temp = (self.X(3 * j + 1) @ (1 / 2 * tools.identity(self.N) + 1 / 2 * self.Z(3 * j + 1)) @ self.X(
                    3 * j + 1) @ (
                                1 / 2 * tools.identity(self.N) - 1 / 2 * self.Z(3 * j + 1))) @ temp
            s = s + temp
        s = s / np.linalg.norm(s)
        if not is_ket:
            s = tools.outer_product(s, s)
        return s

    def StateL(self, state: list, is_ket=False):
        assert len(state) == self.N // 3
        s = np.zeros([self.onb.shape[1], 1])
        for i in range(self.onb.shape[0]):
            temp = np.array([self.onb[i]]).T
            for j in range(self.N // 3):
                if state[j] == 0:
                    temp = (self.X(3 * j + 1) @ (1 / 2 * tools.identity(self.N) - 1 / 2 * self.Z(3 * j + 1)) @ self.X(
                        3 * j + 1) @ (
                                    1 / 2 * tools.identity(self.N) + 1 / 2 * self.Z(3 * j + 1))) @ temp
                if state[j] == 1:
                    temp = (self.X(3 * j + 1) @ (1 / 2 * tools.identity(self.N) + 1 / 2 * self.Z(3 * j + 1)) @ self.X(
                        3 * j + 1) @ (
                                    1 / 2 * tools.identity(self.N) - 1 / 2 * self.Z(3 * j + 1))) @ temp
            s = s + temp
        s = s / np.linalg.norm(s)
        if not is_ket:
            s = tools.outer_product(s, s)
        return s

    def random_field(self, mag):
        vec = np.random.randn(self.Nx * self.Ny * 3, 3)
        return vec / np.linalg.norm(vec) * mag

    def Hfield(self, vec = None, mag = None):
        if vec is None:
            vec = self.random_field(mag)
        # N is the number of physical qubits
        # vec is a list of B vectors (N x 3)
        hfield = np.zeros([2 ** self.N, 2 ** self.N])
        for i in range(self.N):
            hfield = hfield + tools.tensor_product(
                [tools.identity(i), tools.X() * vec[i][0] + tools.Y() * vec[i][1] + tools.Z() * vec[i][2],
                 tools.identity(self.N - i - 1)])
        return hfield

    def trace_fidelity(self, a_list, b_list):
        if len(a_list.shape) == 3 and len(b_list.shape) == 3:
            if a_list.shape[0] != b_list.shape[0]:
                print('Caution: A and B have different lengths ('+str(a_list.shape[0])+') and ('+str(b_list.shape[0]))
            length = min(a_list.shape[0], b_list.shape[0])
            return np.array([tools.trace(a_list[i] @ b_list[i]) for i in range(length)])
        elif len(a_list.shape)-len(b_list.shape)==1:
            return np.array([tools.trace(a_list[i] @ b_list) for i in range(a_list.shape[0])])
        elif len(b_list.shape)-len(a_list.shape)==1:
            return np.array([tools.trace(a_list @ b_list[i]) for i in range(b_list.shape[0])])

    def full_fidelity(self, a_list, b_list):
        if len(a_list.shape) == 3 and len(b_list.shape) == 3:
            if a_list.shape[0] != b_list.shape[0]:
                print('Caution: A and B have different lengths ('+str(a_list.shape[0])+') and ('+str(b_list.shape[0])+')')
            length = min(a_list.shape[0], b_list.shape[0])
            return np.array([tools.fidelity(a_list[i], b_list[i]) for i in range(length)])
        elif len(a_list.shape) - len(b_list.shape) == 1:
            return np.array([tools.fidelity(a_list[i], b_list) for i in range(a_list.shape[0])])
        elif len(b_list.shape) - len(a_list.shape) == 1:
            return np.array([tools.fidelity(a_list @ b_list[i]) for i in range(b_list.shape[0])])

    def run_fidelity(self, s: State, Ep, mag, freq, error='tracenorm', fid_proj=True, fid=True, hc=None, t0=0, tf=50, dt=0.005):
        if hc is None:
            hc = np.zeros([2**self.N, 2**self.N])
        hfield = self.Hfield(mag=mag)
        # Integrate the clean state
        eq = schrodinger_equation.SchrodingerEquation(lambda t: hc)
        res = eq.run_ode_solver(s.state, t0, tf, dt=dt)
        # Numerically integrates the Schrodinger equation
        eq_error = schrodinger_equation.SchrodingerEquation(lambda t: hc + Ep * self.hp + hfield * np.cos(freq * t))
        print('Running error state with Ep = ' + str(Ep))
        print('Frequency = ', str(freq))
        print('Field Magnitude =', str(mag))
        res_error = eq_error.run_ode_solver(s.state, t0, tf, dt=dt)
        if error == 'tracenorm':
            if fid:
                fidelity = self.trace_fidelity(res, res_error)
            if fid_proj:
                proj_fidelity = self.trace_fidelity(self.proj, res_error)
        if error == 'fidelity':
            if fid:
                fidelity = self.full_fidelity(res, res_error)
            if fid_proj:
                proj_fidelity = self.full_fidelity(self.proj, res_error)
        if fid and fid_proj:
            return fidelity, proj_fidelity, res_error
        elif fid:
            return fidelity, res_error
        elif fid_proj:
            return proj_fidelity, res_error

    def run_expectation(self, s: State, var, Ep, mag, freq, hc=None, t0=0, tf=50, dt=0.005):
        if hc is None:
            hc = np.zeros([2**self.N, 2**self.N])
        hfield = self.Hfield(mag=mag)
        # Integrate the clean state
        eq = schrodinger_equation.SchrodingerEquation(lambda t: hc)
        res = eq.run_ode_solver(s.state, t0, tf, dt=dt)
        # Numerically integrates the Schrodinger equation
        eq_error = schrodinger_equation.SchrodingerEquation(lambda t: hc + Ep * self.hp + hfield * np.cos(freq * t))
        print('Running error state with Ep = ' + str(Ep))
        print('Frequency = ', str(freq))
        print('Field Magnitude =', str(mag))
        res_error = eq_error.run_ode_solver(s.state, t0, tf, dt=dt)
        fidelity = self.trace_fidelity(res, var)
        return fidelity, res_error


    def plot_fidelity_vs_time(self, s:State, Ep, mag, freq, error='tracenorm', hc=None, t0=0, tf=50, dt=0.005):
        # Plots fidelity versus time
        gridspec = {'width_ratios': [1, 1]}
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=gridspec)
        plt.xlabel(r'$t$')
        plt.suptitle('Random inhomogeneous magnetic field with ' + r'$\omega = $' + str(freq))
        ax[0].set_ylabel(r'tr$\left(\rho_{ideal}\rho_{field}\right)$')
        ax[1].set_ylabel(r'tr$\left(\Pi_0\rho_{field}\right)$')
        ax[0].set_xlabel(r'$t$')
        ax[1].set_xlabel(r'$t$')

        if isinstance(Ep, list):
            for i in range(len(Ep)):
                fidelity, proj_fidelity, res = self.run_fidelity(s, Ep[i], mag, freq, error=error, fid_proj=True, fid=True, hc=hc, t0=t0, tf=tf, dt=dt)
                ax[0].plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity)
                ax[1].plot(np.linspace(t0, tf, int((tf - t0) / dt)), proj_fidelity, label=r'$Ep = $' + str(Ep[i]))

        else:
            fidelity, proj_fidelity, res = self.run_fidelity(s, Ep, mag, freq, error=error, fid_proj=True, fid=True, hc=hc, t0=t0, tf=tf, dt=dt)
            ax[0].plot(np.linspace(t0, tf, int((tf - t0) / dt)), fidelity)
            ax[1].plot(np.linspace(t0, tf, int((tf - t0) / dt)), proj_fidelity, label=r'$Ep = $' + str(Ep))
        plt.legend(loc='lower right')
        plt.show()

    def plot_frequency_vs_magnitude(self, s: State, Ep, error='tracenorm', n=10, hc=None, t0=0, tf=50):
        gridspec = {'width_ratios': [1, 1, 0.2]}
        fig, ax = plt.subplots(1, 3, figsize=(12, 5), gridspec_kw=gridspec)
        for ep in Ep:
            mag = np.linspace(0, 10, 15)
            freq = np.linspace(0, np.pi, 15)
            fidelities = np.zeros([mag.shape[0], freq.shape[0]])
            tracenorm_fidelities = np.zeros([mag.shape[0], freq.shape[0]])
            for j in range(mag.shape[0]):
                for k in range(freq.shape[0]):
                    # Average over 10 points
                    fid = np.zeros(n)
                    tracenorm = np.zeros(n)
                    for i in range(n):
                        res = self.run_fidelity(s, ep, mag[j], freq[k], error=error, hc=hc, t0=t0, tf=tf, dt=(tf-t0)/2)
                        fid[i] = res[0][-1]
                        tracenorm[i] = res[1][-1]
                    fidelities[k][j] = np.mean(fid)
                    tracenorm_fidelities[k][j] = np.mean(tracenorm)
                print('Fidelity = ', fidelities[k][j])

            surf = ax[0].imshow(fidelities, interpolation='none', aspect="auto",
                                extent=[mag[0], mag[-1], freq[-1], freq[0]])
            ax[1].imshow(tracenorm_fidelities, interpolation='none', aspect="auto",
                         extent=[mag[0], mag[-1], freq[-1], freq[0]])
            plt.sca(ax[1])
            plt.suptitle(r'$E_p = $' + str(ep))
            ax[1].set_title(r'tr$\left(\Pi_0\rho_{field}\right)$')
            ax[0].set_title(r'tr$\left(\rho_{ideal}\rho_{field}\right)$')
            ax[0].set_xlabel(r'$|\vec{B}|$')
            ax[0].set_ylabel(r'$\omega$')
            ax[1].set_xlabel(r'$|\vec{B}|$')
            ax[1].set_ylabel(r'$\omega$')
            cax = ax[2]
            plt.colorbar(surf, cax=cax)  # , shrink=0.5, aspect=5)
            plt.tight_layout()
            plt.savefig('plot_ep' + str(ep))
        plt.show()

    def plot_fidelity_vs_ep(self, s, Ep, mag, freq, error='tracenorm', n=10, hc=None, t0=0, tf=50, dt=0.005):
        gridspec = {'width_ratios': [1]}
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), gridspec_kw=gridspec)
        for j in range(mag.shape[0]):
            fidelities = np.zeros(Ep.shape)
            for i in range(Ep.shape[0]):
                fid = np.zeros(n)
                for k in range(n):
                    fid[k] = np.median(self.run_fidelity(s, Ep[i], mag[j], freq, fid=False, error=error, hc=hc, t0=t0, tf=tf, dt=dt)[0])
                fidelities[i] = np.mean(fid)
            ax.scatter(Ep, fidelities, label = r'$|B| =$'+str(mag[j]))
            ax.plot(Ep, fidelities, c='k')

        ax.set_ylabel(r'Median Code Space Fidelity (tr$\left(\Pi_0\rho_{field}\right)$)')
        ax.set_xlabel(r'$E_p$')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('fidelity_vs_ep')
        plt.show()

    def plot_fidelity_vs_N(self, N, Ep, mag, freq, error='tracenorm', n=10, hc=None, t0=0, tf=10, dt=0.1):
        gridspec = {'width_ratios': [1]}
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), gridspec_kw=gridspec)
        fidelities = np.zeros(len(N))
        sizes = np.zeros(len(N))
        for j in range(len(N)):
            print('Size = ' + str(N[j]))
            temp = MarvianCode(N[j][0], N[j][1])
            s = State(temp.ZeroL(), temp.N)
            sizes[j] = temp.N
            fid = np.zeros(n)
            for k in range(n):
                fid[k] = np.median(temp.run_fidelity(s, Ep, mag, freq, fid=False, error=error, hc=hc, t0=t0, tf=tf, dt=dt)[0])
            fidelities[j] = np.mean(fid)
        ax.scatter(sizes, fidelities)
        ax.plot(sizes, fidelities, c='k')
        plt.title(r'Code Space Fidelity with $|\vec{B}| = $'+str(mag)+r'and $\omega =$'+str(freq))
        ax.set_ylabel(r'Median Code Space Fidelity (tr$\left(\Pi_0\rho_{field}\right)$)')
        ax.set_xlabel(r'Number of Logical Qubits $N$')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('fidelity_vs_size')
        plt.show()

    def plot_expectation(self, N, Ep, mag, freq, n=10, hc=None, t0=0, tf=10, dt=0.1):
        gridspec = {'width_ratios': [1]}
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), gridspec_kw=gridspec)
        fidelities = np.zeros(len(N))
        sizes = np.zeros(len(N))
        for j in range(len(N)):
            temp = MarvianCode(N[j][0], N[j][1])
            var = temp.Z(1)
            s = State(temp.ZeroL(), temp.N)
            sizes[j] = temp.N
            fid = np.zeros(n)
            for k in range(n):
                fid[k] = np.median(temp.run_expectation(s, var, Ep, mag, freq, hc=hc, t0=t0, tf=tf, dt=dt)[0])
            fidelities[j] = np.mean(fid)
        ax.scatter(sizes, fidelities)
        ax.plot(sizes, fidelities, c='k')

        ax.set_ylabel(r'$\langle Z_1 \rangle$)')
        ax.set_xlabel(r'Number of Logical Qubits $N$')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('expectation_vs_size')
        plt.show()


    def plot_hp_gap(self, N):
        for i in range(len(N)):
            temp = MarvianCode(N[i][0], N[i][1])
            eigvals = np.linalg.eig(temp.hp)[0]
            print(sorted(eigvals)[0:2**(temp.Nx * temp.Ny)+1])


# Assume logical qubit layout is a Ny x Nx grid
code = MarvianCode(1, 2)
#code.plot_hp_gap([[2,2]])
s = State(code.ZeroL(), code.N)
#code.plot_fidelity_vs_time(s, [5, 10, 15], 1, 5.359)
#code.plot_fidelity_vs_ep(s, np.linspace(10, 70, 7), np.linspace(2, 10, 5), 0, n=10)
code.plot_fidelity_vs_N([[1, 2], [1, 4]], 10, 1, 0, n=1, tf=2)
code.plot_expectation()
#code.plot_frequency_vs_magnitude(s, [0], n=10)
#code.plot_fidelity_vs_time(s, [10, 20], 1, 0)



# Test B field orientation
#vec = np.array([[0.96008644, -0.51543102, -0.13664101],
#                [0.40128861, -0.44023774, 1.0866038],
#                [-1.51436961, -0.56834696, 0.29103908],
#                [2.31072175, 0.12211733, 0.81743674],
#                [-0.95217354, -1.14535345, -0.52514697],
#                [0.03665265, -0.45178826, -0.29247424]])



"""



def spin_echo(Ep, fieldmag, freq, error='tracenorm', gs_proj=True, color='b'):
    global res
    global proj
    global s
    vec = np.random.randn(Nx * Ny * 3, 3)
    vec = vec / np.linalg.norm(vec) * fieldmag
    hfield = Hfield(vec, Nx * Ny * 3)
    # Integrate the clean state
    eq = schrodinger_equation.SchrodingerEquation(lambda t: hb)  # + tools.identity(Nx * Ny * 3))
    res = eq.run_ode_solver(s.state, t0, 2 * tf + 1, dt=dt)
    # Numerically integrates the Schrodinger equation
    eq_error = schrodinger_equation.SchrodingerEquation(lambda t: hb + Ep * hp + hfield * np.cos(freq * t))
    print('Running error state with Ep = ' + str(Ep))
    print('Frequency = ', str(freq))
    print('Field Magnitude =', str(fieldmag))
    res_error = eq_error.run_ode_solver(s.state, t0, tf, dt=dt)
    noiY_res = Xtot(Nx * Ny * 3) @ res_error[-1] @ Xtot(Nx * Ny * 3)
    # Numerically integrates the Schrodinger equation
    eq_error = schrodinger_equation.SchrodingerEquation(lambda t: hb + Ep * hp + hfield * np.cos(freq * t))
    res_error = np.concatenate((res_error, eq_error.run_ode_solver(noiY_res, t0, tf, dt=dt)))
    res_error = np.append(res_error, np.array([Xtot(Nx * Ny * 3) @ res_error[-1] @ Xtot(Nx * Ny * 3)]), axis=0)
    if error == 'tracenorm':
        if not gs_proj:
            fidelity = np.array([tools.trace(res[i] @ res_error[i]) for i in range(res_error.shape[0])])
        else:
            fidelity = np.array([tools.trace(proj @ res_error[i]) for i in range(res_error.shape[0])])
    if error == 'fidelity':
        if not gs_proj:
            fidelity = np.array([tools.fidelity(res[i], res_error[i]) for i in range(res_error.shape[0])])
        else:
            fidelity = np.array([tools.fidelity(proj, res_error[i]) for i in range(res_error.shape[0])])
    plt.plot(np.linspace(t0, 2 * tf + 1, 2 * int((tf - t0) / dt) + 1), fidelity, c=color)
    # plt.plot(np.linspace(t0, 2 * tf + 1, 2 * int((tf - t0) / dt) + 1), prob3, c='k')
    # plt.plot(np.linspace(t0, 2 * tf + 1, 2 * int((tf - t0) / dt) + 1), prob4, c='g')
    eq_error = schrodinger_equation.SchrodingerEquation(lambda t: Ep * hp + hfield * np.cos(freq * t))
    res_error = eq_error.run_ode_solver(s.state, t0, 2 * tf + 1, dt=dt)
    res = np.append(res, res, axis=0)
    if error == 'tracenorm':
        if not gs_proj:
            fidelity = np.array([tools.trace(res[i] @ res_error[i]) for i in range(res_error.shape[0])])
            prob1 = np.array(
                [tools.trace(tools.outer_product(OneL(Nx * Ny * 3, onb), OneL(Nx * Ny * 3, onb)) @ res_error[i]) for i
                 in range(res_error.shape[0])])
            prob2 = np.array(
                [tools.trace(tools.outer_product(ZeroL(Nx * Ny * 3, onb), ZeroL(Nx * Ny * 3, onb)) @ res_error[i]) for i
                 in range(res_error.shape[0])])
        if gs_proj:
            fidelity = np.array([tools.trace(proj @ res_error[i]) for i in range(res_error.shape[0])])
    if error == 'fidelity':
        if not gs_proj:
            fidelity = np.array([tools.fidelity(res[i], res_error[i]) for i in range(res_error.shape[0])])
        if gs_proj:
            fidelity = np.array([tools.fidelity(proj, res_error[i]) for i in range(res_error.shape[0])])
    plt.plot(np.linspace(t0, 2 * tf + 1, int((2 * tf + 1 - t0) / (dt))), fidelity, c='r')
    plt.plot(np.linspace(t0, 2 * tf + 1, int((2 * tf + 1 - t0) / (dt))), prob1, c='y')
    plt.plot(np.linspace(t0, 2 * tf + 1, int((2 * tf + 1 - t0) / (dt))), prob2, c='c')
    plt.show()
    return fidelity, res_error







def localvar_plot():
    pass


fidelityvsepplot()
# spin_echo(50, 1, 0, gs_proj=False)
# timeplot([0, 10, 50, 100], 1, 0, color = ['b', 'g', 'r', 'c'])
# freqvsmagplot([0, 10, 50, 100])

"""