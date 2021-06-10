import qsim.evolution.hamiltonian
from qsim.graph_algorithms.graph import line_graph
from qsim.tools import tools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, eigsh, expm
from scipy.special import comb
from scipy.special import iv
from scipy.fft import fft
import matplotlib.pyplot as plt


def disorder_hamiltonian(states,h=1.):
    disorder = (np.random.random(size=states.shape[1])-1/2)*2*h
    return sparse.csc_matrix(
        (np.sum((1/2-states)*2*disorder, axis=1), (np.arange(states.shape[0]),
                                    np.arange(states.shape[0]))), shape=(states.shape[0], states.shape[0]))


def arnoldi(A, v, m):
    """
    Constructs an orthonormal basis of vectors V given a vector v and matrix A in the Krylov subspace of rank m.
    :param A:
    :param v:
    :param m:
    :return:
    """
    V = np.zeros((len(v), m+1))
    H = np.zeros((m+1, m))
    beta = np.linalg.norm(v)
    V[:,0] = v/beta
    for j in range(m):
        p = A @ V[:, j]
        for i in range(j):
            H[i,j] = V[:, i].conj().T @ p
            p -= H[i, j]*V[:, i]
        H[j+1, j] = np.linalg.norm(p)
        V[:, j+1] = p/H[j+1, j]
    return V, H


def lanczos(A, v, m):
    """
    Constructs an orthonormal basis of vectors V given a vector v and matrix A in the Krylov subspace of rank m.
    :param A:
    :param v:
    :param m:
    :return:
    """
    V = np.zeros((len(v), m))
    T = np.zeros((m, m))
    b = np.linalg.norm(v)
    V[:,0] = v/b
    w = A @ V[:, 0, np.newaxis]
    a = w.conj().T @ V[:,0]
    w -= a * V[:,0, np.newaxis]
    T[0,0] = a
    for j in range(1, m):
        b = np.linalg.norm(w)
        T[j, j-1] = T[j-1, j] = b
        if b != 0:
            V[:,j] = w.flatten()/b
        else:
            # Need to
            raise NotImplementedError
        w = A @ V[:, j, np.newaxis]
        a = w.conj().T @ V[:, j]
        w -= (a * V[:, j, np.newaxis] + b*V[:, j-1, np.newaxis])
        T[j,j] = a
    return V, T


def matvec(A, v):
    return A @ v


def chebyshev(A, v, tau, eps, p_check, verbose=False):
    # eps is (relative?) error
    # p is frequency of checking convergence
    norm = np.linalg.norm(v)

    def truncation_m(k0, eps, s_norm, v):
        # used to set the actual truncation point of the cheb iteration
        m = k0 + 1
        while True:
            if np.sum(np.abs(chebs_upper[m:])) * np.linalg.norm(v) / s_norm < eps:
                return m
            else:
                m += 1

    def chebs(m, l1, l2, tau):
        # first m chebyshev coefficients
        # first find the needed Bessel function values by backwards recursion
        bessels = np.zeros(m + 1)
        bessels[m] = bessel_asymptotic(m, -tau * l1)
        bessels[m - 1] = bessel_asymptotic(m - 1, -tau * l1)
        for i in range(m-2, -1, -1):
            bessels[i] = 2 * (i + 1) / (-tau * l1) * bessels[i + 1] + bessels[i + 2]
        # Now convert to Chebyshev coefficient with exponential prefactor:
        bessels *= 2 * np.exp(-tau * l2)
        bessels[0] *= 0.5
        return bessels

    def bessel_asymptotic(nu, z, order=5):
        return iv(nu, z)
        # from Abramowitz and Stegun, 9.7.
        # What is a good order to choose?
        mu = 4 * nu ** 2
        def term(n):
            tmp = 1
            for i in range(n):
                tmp *= mu - (2 * i + 1) ** 2
            return (-1) ** n * tmp / (np.math.factorial(n) * (8 * z) ** n)

        subleading = 0
        for n in range(order):
            subleading += term(n)
        # TODO: make this faster
        return np.exp(z) / (2 * np.pi * z) ** 0.5 * subleading

    def max_eig_estimate(A):
        # This is the infinity-norm. Lanczos method probably more accurate but slower
        # TODO: determine if we should use the infinity norm or just compute the eigenvalue
        # TODO: determine if we should guarantee a sufficient overestimate
        eigval = eigsh(A, k=1, which='LA', return_eigenvectors=False)
        return eigval[0]

    def min_eig_estimate(A):
        # Implement minimization of Rayleigh quotient; supposed to return min eigvec too
        def rayleigh_quotient(A, v):
            return np.dot(v, matvec(A, v)) / np.linalg.norm(v) ** 2
        # Alternatively: for H=-J\sum S_i\cdot S_{i+1}+\sum_i h_i*S_z^i (where the h_i is the disorder),
        # the min eigval must be larger than -NJ/4-\sum h_i/2. This is a bound, though it doesn’t give the eigvec.
        # Return min_eig, min_vec
        eigval, eigvec = eigsh(A, k=1, which='SA')
        return eigval[0], eigvec

    def m_max(eps, v, u_1, tau, lambda_n):
        # minimum m satisfying E(m) < bound
        # u1 is the output min eigvec of the RQ minimization function
        m = 0
        c1 = np.dot(v, u_1)
        bound = np.abs(eps * c1 / (2 * norm))
        while True:
            if E(m, tau, lambda_n) < bound:
                return m
            else:
                m += 1

    def E(m, tau, lambda_n):
        b = 0.618
        d = 0.438  # Pull more accurate values later
        if m > tau * lambda_n:
            return d ** m / (1 - d)
        else:
            return np.exp(-(m + 1) ** 2 / (tau * lambda_n)) * (1 + np.sqrt(np.pi * tau * lambda_n / (4 * b))) + \
                   d ** (tau * lambda_n) / (1 - d)

    # First estimate spectral range
    lambda_1, v_1 = min_eig_estimate(1j*A)
    lambda_n = max_eig_estimate(1j*A)
    l1 = (lambda_n - lambda_1) / 2
    ln = (lambda_n + lambda_1) / 2
    # Compute conservative upper bound for the number of terms needed
    if verbose:
        print('beginning')
    m_upper = m_max(eps, v, v_1, tau, lambda_n)
    if verbose:
        print('m_upper:{}'.format(m_upper))
    # A_k in the expansion
    chebs_upper = chebs(m_upper, l1, ln, tau)
    if verbose:
        print('chebs:')
        print(chebs_upper)
    # initialize vectors etc for cheb iteration
    v_old = v
    v_new = matvec(A, v) / l1 - ln / l1 * v
    s_current = chebs_upper[1] * v_new + chebs_upper[0] * v_old
    s_norms = [np.linalg.norm(chebs_upper[0] * v), np.linalg.norm(s_current)]
    exit_test = False
    norm_test = False
    k = 1
    while not exit_test:
        v_old = 2*(matvec(A, v_new) / l1 - ln / l1 * v_new) - v_old

        s_current = s_current+chebs_upper[k + 1] * v_old
        s_norms.append(np.linalg.norm(s_current))
        if k % p_check == 0 and not norm_test:
            r = np.linalg.norm(s_current) / s_norms[k+1-p_check]
            if np.abs(r - 1) < 0.1:
                k0 = k + 1
                norm_test = True
                #m_upper = truncation_m(k0, eps, s_norms[k+1-p_check], v)
                if verbose:
                    print('r={}, m_upper_new={}'.format(r, m_upper))
                    print('k={}'.format(k))
        k += 1
        exit_test = (k == m_upper)
        v_new, v_old = v_old, v_new
    return s_current



def krylov(v, A, t, m, dt, tol):
    """
    Compute exp(At)v
    :param v: Input vector to be multiplied
    :param A: Matrix to exponentiate
    :param t: Time prefactor in matrix exponential
    :param m: Order of the polynomial to use
    :return: exp(At)v
    """
    tk = 0
    while tk < t:
        V, H, beta = lanczos(A, v, m)
        err = np.inf
        raise NotImplementedError
        while err > tol:
            w = beta*V*np.exp(H*dt)
        tk += dt

"""d = 10
v = np.random.random(d)
v = v/np.linalg.norm(v)
A = np.random.random((d, d))
A = A+A.T
tau = 1
correct = expm_multiply(-A*tau, v)
result = chebyshev(A, v, tau, 1e-6, 7, verbose=True)
print(correct)
print(result)
print((correct-result)/correct)
raise Exception"""

def generate_time_evolved_states(graph, times, verbose=False, h=1):
    import dill
    if verbose:
        print('beginning')

    #heisenberg = qsim.evolution.hamiltonian.HamiltonianHeisenberg(graph, subspace=0, energies=(1/4, 1/2))
    #print(heisenberg.hamiltonian.todense(), heisenberg.hamiltonian.shape)
    # dill.dump(heisenberg, open('heisenberg_18.pickle', 'wb'))
    heisenberg = dill.load(open('heisenberg_18.pickle', 'rb'))
    print(heisenberg.hamiltonian)
    dim = int(comb(graph.n, graph.n // 2))
    hamiltonian = heisenberg.hamiltonian + disorder_hamiltonian(heisenberg.states, h=h/2)
    #hamiltonian_exp = np.exp(hamiltonian.todense)
    if verbose:
        print('initialized Hamiltonian')
    if isinstance(times, list):
        states_list = []
        states_exp_list = []

        z_mag_list = []
        z_mag_exp_list = []
        for time in times:
            n_times = len(time)
            hamiltonian_exp = expm(-1j *hamiltonian * (time[1] - time[0]))
            state = np.zeros((dim, 1))
            state[-1, 0] = 1
            states = np.zeros((dim, n_times), dtype=np.complex128)
            states_exp = np.zeros((dim, n_times), dtype=np.complex128)
            states[..., 0] = state.flatten()
            states_exp[..., 0] = state.flatten()
            for i in range(n_times - 1):
                if verbose:
                    print(i)
                states[..., i + 1] = expm_multiply(-1j * hamiltonian * (time[i + 1] - time[i]), states[..., i])
                states_exp[..., i + 1] = hamiltonian_exp @ states_exp[...,i]
            z_mag = ((np.abs(states) ** 2).T @ (heisenberg.states - 1 / 2) * 2).real/2
            z_mag_exp = ((np.abs(states_exp) ** 2).T @ (heisenberg.states - 1 / 2) * 2).real / 2
            print(z_mag)
            states_list.append(states)
            states_exp_list.append(states_exp)
            z_mag_list.append(z_mag)
            z_mag_exp_list.append(z_mag)
        return states_list, states_exp_list, z_mag_list, z_mag_exp_list
    else:
        n_times = len(times)
        #hamiltonian_exp = expm(-1j *hamiltonian * (times[1] - times[0]))
        state = np.zeros((dim,1))
        state[-1,0] = 1
        states = np.zeros((dim, n_times), dtype=np.complex128)
        states[...,0] = state.flatten()
        #states_exp = np.zeros((dim, n_times), dtype=np.complex128)
        states_cheb = np.zeros((dim, n_times), dtype=np.complex128)
        states = np.zeros((dim, n_times), dtype=np.complex128)
        states[..., 0] = state.flatten()
        #states_exp[..., 0] = state.flatten()
        states_cheb[..., 0] = state.flatten()
        for i in range(n_times-1):
            if verbose:
                print(i)
            states[...,i+1] = expm_multiply(-1j*hamiltonian*(times[i+1]-times[i]), states[...,i])
            #states_exp[..., i + 1] = hamiltonian_exp @ states_exp[..., i]
            states_cheb[..., i + 1] = chebyshev(-1j*hamiltonian, states_cheb[...,i], times[i+1]-times[i], 1e-6, 7)
        z_mag = ((np.abs(states)**2).T @ (heisenberg.states-1/2)*2).real/2
        #z_mag_exp = ((np.abs(states_exp) ** 2).T @ (heisenberg.states - 1 / 2) * 2).real/2
        z_mag_cheb = ((np.abs(states_cheb) ** 2).T @ (heisenberg.states - 1 / 2) * 2).real / 2
        return states, states_cheb, states_cheb, z_mag, z_mag_cheb, z_mag_cheb

t_final = 3
n_times = 1000
n = 18
h = 2
graph = line_graph(n)
times = np.linspace(1, 10**t_final, n_times)
states, states_exp, states_cheb, z_mag, z_mag_exp, z_mag_cheb = generate_time_evolved_states(graph, times, verbose=True, h=h)
print(z_mag, z_mag_exp)
# Compute z magnetization
fig, ax = plt.subplots(2,graph.n//2, sharey=True)
for i in range(graph.n//2):
    ax[0][i].semilogx(times, (z_mag[:,i].flatten()), label='Pade')
    ax[1][i].semilogx(times, (z_mag[:, i+graph.n//2].flatten()))
    #ax[0][i].semilogx(times, (z_mag_exp[:, i].flatten()), label='Exact diag.')
    #ax[1][i].semilogx(times, (z_mag_exp[:, i + graph.n // 2].flatten()))
    ax[0][i].semilogx(times, (z_mag_cheb[:, i].flatten()), label='Chebyshev')
    ax[1][i].semilogx(times, (z_mag_cheb[:, i + graph.n // 2].flatten()))
    #ax[0][i].set_xscale("log", nonpositive='clip')
    #ax[1][i].set_xscale("log", nonpositive='clip')
    ax[0][i].text(-0.1, 1.05, '$x=$'+str(i), transform=ax[0][i].transAxes,
            size=10, weight='bold')
    ax[1][i].text(-0.1, 1.05, '$x=$' + str(i+graph.n//2), transform=ax[1][i].transAxes,
                  size=10, weight='bold')
ax[0][0].legend()
ax[0][0].set_ylabel(r'$\langle S_z(x)\rangle $')
ax[1][0].set_ylabel(r'$\langle S_z(x)\rangle $')
fig.text(0.5, 0.04, 'Time $(1/J)$', ha='center')
fig.suptitle(r'$J=J_z=1, h_{\mathrm{max}}=$'+str(h))
plt.show()
z_mag = np.abs(fft(z_mag, axis=1))
fig, ax = plt.subplots(2,graph.n//2, sharey=True)
for i in range(graph.n//2):
    ax[0][i].semilogx(times, (z_mag[:, i].flatten()))
    ax[1][i].semilogx(times, (z_mag[:, i + graph.n // 2].flatten()))
    ax[0][i].text(-0.1, 1.05, '$k=$' + str(i), transform=ax[0][i].transAxes,
                  size=10, weight='bold')
    ax[1][i].text(-0.1, 1.05, '$k=$' + str(i + graph.n // 2), transform=ax[1][i].transAxes,
                  size=10, weight='bold')
ax[0][0].set_ylabel(r'$\langle S_z(k)\rangle $')
ax[1][0].set_ylabel(r'$\langle S_z(k)\rangle $')
ax[1][graph.n//4].set_xlabel('Time $(1/J)$')

plt.show()