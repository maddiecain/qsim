import qsim.evolution.hamiltonian
from qsim.graph_algorithms.graph import line_graph
from qsim.tools import tools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, eigsh, expm
from scipy.special import comb
from scipy.special import jv, iv
from qsim.codes.quantum_state import State
from scipy.fft import fft
import matplotlib.pyplot as plt
import dill
from os import path


def disorder_hamiltonian(states, h=1., subspace=None):
    if subspace is None:
        disorder = (np.random.random(size=states.shape[1])-1/2)*h
        return sparse.csc_matrix(
            (np.sum((1/2-states)*2*disorder, axis=1), (np.arange(states.shape[0]),
                                        np.arange(states.shape[0]))), shape=(states.shape[0], states.shape[0]))
    if subspace is 'all':
        return (np.random.random(size=states) - 1 / 2) * h



def matvec_heisenberg(heisenberg: qsim.evolution.hamiltonian.HamiltonianHeisenberg, disorder, state: State):
    temp = np.zeros_like(state)
    # For each logical qubit
    state_shape = state.shape
    for i in range(state.number_logical_qudits):
        ind = 2 ** i
        out = np.zeros_like(state, dtype=np.complex128)
        state = state.reshape((-1, 2, ind), order='F')
        # Note index start from the right (sN,...,s3,s2,s1)
        out = out.reshape((-1, 2, ind), order='F')
        out[:, [0, 1], :] = state[:, [0, 1], :]
        out[:, 1, :] = -disorder[i] * out[:, 1, :]
        out[:, 0, :] = disorder[i] * out[:, 0, :]
        state = state.reshape(state_shape, order='F')
        out = out.reshape(state_shape, order='F')
        temp = temp + out
    return heisenberg.left_multiply(state) + temp


def matvec(A, v):
    """

    :param A:
    :param v:
    :return:
    """
    return A @ v


def chebyshev(A, v, tau, eps, p_check, verbose=False):
    def max_eig_estimate(A):
        # This is the infinity-norm. Lanczos method probably more accurate but slower
        eigval = eigsh(A, k=1, which='LA', return_eigenvectors=False)
        return eigval[0]

    def min_eig_estimate(A):
        eigval, eigvec = eigsh(A, k=1, which='SA')
        return eigval[0], eigvec
    # eps is (relative?) error
    # p is frequency of checking convergence
    norm = np.linalg.norm(v)
    # First estimate spectral range
    lambda_1, v_1 = min_eig_estimate(1j * A)
    c1 = np.dot(v, v_1)
    lambda_n = max_eig_estimate(1j * A)
    l1 = (lambda_n - lambda_1) / 2
    ln = (lambda_n + lambda_1) / 2

    def truncation_m(k0, eps, s_norm, v):
        # used to set the actual truncation point of the cheb iteration
        m = k0 + 1
        while True:
            if np.sum(np.abs(chebs_upper[m:])) * norm / s_norm < eps:
                return m
            else:
                m += 1

    def chebs(m, l1, ln, tau):
        # first m chebyshev coefficients
        # first find the needed Bessel function values by backwards recursion
        bessels = np.zeros(m + 1, dtype=np.complex128)
        bessels[m] = bessel_asymptotic(m, tau * l1)
        bessels[m - 1] = bessel_asymptotic(m - 1, tau * l1)
        for i in range(m-2, -1, -1):
            print(i)
            bessels[i] = bessel_asymptotic(i, tau * l1)
            #bessels[i] = 2*(i+1) / (-1j*tau * l1) * bessels[i + 1] + bessels[i + 2]
        # Now convert to Chebyshev coefficient with exponential prefactor:
        bessels *= 2 * np.exp(tau * ln)
        bessels[0] *= 0.5
        return bessels

    def bessel_asymptotic(nu, z):
        return iv(nu, z)

    def bessel_approximate(nu, z, order=5):
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
        return np.exp(z) / (2 * np.pi * z) ** 0.5 * subleading

    def m_max(eps, tau, lambda_1, lambda_n):
        bound = eps * np.abs(c1) / (4 * norm)
        if ((0.5)**(np.exp(1)*tau*(lambda_n-lambda_1)/2) < bound):
            return int(np.exp(1)*tau*(lambda_n-lambda_1)/2)
        else:
            return int(np.log(1/bound)/np.log(2))




        # minimum m satisfying E(m) < bound
        # u1 is the output min eigvec of the RQ minimization function
        m = 0
        bound = eps * np.abs(c1) / (4 * norm)
        while True:
            if E(m, tau, lambda_1, lambda_n) < bound:
                return m
            else:
                m += 1

    def E(m, tau, lambda_1, lambda_n):
        b = 0.618
        d = 0.438  # Pull more accurate values later
        rho = lambda_n-lambda_1
        if m > tau * rho:
            return d ** m / (1 - d)
        else:
            return np.exp(-b * (m + 1) ** 2 / (tau * rho)) * (1 + np.sqrt(np.pi * tau * rho / (4 * b))) + \
                   d ** (tau * rho) / (1 - d)


    # Compute conservative upper bound for the number of terms needed
    if verbose:
        print('beginning')
    m_upper = m_max(eps, tau, lambda_1, lambda_n)
    if verbose:
        print('m_upper:{}'.format(m_upper))
    # A_k in the expansion
    print(chebs(5, 0.8, 2.0, -1j*1.2))
    raise Exception
    chebs_upper = chebs(m_upper, l1, ln, -1j*tau)
    #print(chebs_upper, [bessel_asymptotic(m, -tau*l1)*2 * np.exp(-tau * ln) for m in range(0, m_upper+1)])
    #print(np.sum(chebs(1000, l1, ln, tau))-np.sum(chebs_upper), 2*norm*np.exp(-tau*lambda_1)*E(m_upper, tau, lambda_1, lambda_n))
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
                m_upper = truncation_m(k0, eps, s_norms[k+1-p_check], v)
                if verbose:
                    print('r={}, m_upper_new={}'.format(r, m_upper))
                    print('k={}'.format(k))
        k += 1
        exit_test = (k == m_upper)
        v_new, v_old = v_old, v_new
    return s_current

def chebs_real(m, l1, ln, tau):
    # first m chebyshev coefficients
    # first find the needed Bessel function values by backwards recursion
    bessels = np.zeros(m + 1, dtype=np.complex128)
    bessels[m] = iv(m, tau * l1)
    bessels[m - 1] = iv(m - 1, tau * l1)
    for i in range(m-2, -1, -1):
        bessels[i] = iv(i, tau * l1)
        #bessels[i] = 2*(i+1) / (-1j*tau * l1) * bessels[i + 1] + bessels[i + 2]
    # Now convert to Chebyshev coefficient with exponential prefactor:
    bessels *= 2 * np.exp(-tau * ln)
    bessels[0] *= 0.5
    return bessels

def chebs_imag(m, l1, ln, tau):
    # first m chebyshev coefficients
    # first find the needed Bessel function values by backwards recursion
    bessels = np.zeros(m + 1, dtype=np.complex128)
    bessels[m] = jv(m, tau * l1)*(1j)**(-m)
    bessels[m - 1] = jv(m - 1, tau * l1)*(1j)**(-(m-1))
    for k in range(m-2, -1, -1):
        #bessels[k] = (1j)**(-k)*jv(k, tau * l1)
        bessels[k] = 2*(k+1) / (-1j*tau * l1) * bessels[k + 1] + bessels[k + 2]
    # Now convert to Chebyshev coefficient with exponential prefactor:
    bessels *= 2 * np.exp(1j*tau * ln)
    bessels[0] *= 0.5
    return bessels
"""print(chebs_real(5, .8, 2., -1j*1.2))
print(chebs_imag(5, .8, 2., 1.2))

raise Exception
d = 10
v = np.random.random(d)
v = np.array(v/np.linalg.norm(v), dtype=np.complex128)
A = np.random.random((d, d))
A = -1j*(A+A.T)
tau = 1
correct = expm_multiply(A*tau, v)
result = chebyshev(A, v, tau, 1e-6, 7, verbose=True)
#print(correct)
#print(result)
#print(np.linalg.norm(result))
#print((np.abs(correct)-np.abs(result))/np.abs(correct))"""


def generate_time_evolved_states(graph, times, verbose=False, h=1):
    import dill
    if verbose:
        print('beginning')

    heisenberg = qsim.evolution.hamiltonian.HamiltonianHeisenberg(graph, subspace=0, energies=(1/4, 1/2))
    #print(heisenberg.hamiltonian.todense(), heisenberg.hamiltonian.shape)
    dill.dump(heisenberg, open('heisenberg_'+str(graph.n)+'.pickle', 'wb'))
    #heisenberg = dill.load(open('heisenberg_18.pickle', 'rb'))
    dim = int(comb(graph.n, graph.n // 2))
    hamiltonian = heisenberg.hamiltonian + disorder_hamiltonian(heisenberg.states, h=h)
    print('hi', eigsh(hamiltonian, k=1, which='SA'))
    raise Exception
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


def return_probability(graph, times, verbose=False, h=1, exact=True):
    """
    For random product states in the computational basis, time evolve then compute
    :param graph:
    :param times:
    :param verbose:
    :param h:
    :return:
    """

    if verbose:
        print('beginning')

    for k in range(2**graph.n):
        # Compute the total magnetization
        z_mags_init = 2*(1/2-tools.int_to_nary(k, size=graph.n))
        if verbose:
            print(z_mags_init)
        subspace = int(np.sum(z_mags_init))
        if path.exists('heisenberg_' + str(graph.n)+'_'+str(subspace) + '.pickle'):
            heisenberg = dill.load(open('heisenberg_' + str(graph.n)+'_'+str(subspace) + '.pickle', 'rb'))
        else:
            heisenberg = qsim.evolution.hamiltonian.HamiltonianHeisenberg(graph, subspace=subspace, energies=(1 / 4, 1 / 2))
            dill.dump(heisenberg, open('heisenberg_' + str(graph.n)+'_'+str(subspace) + '.pickle', 'wb'))
            if verbose:
                print('initialized Hamiltonian')

        # For every computational basis state,
        dim = int(comb(graph.n, graph.n // 2+subspace))
        hamiltonian = heisenberg.hamiltonian + disorder_hamiltonian(heisenberg.states, h=h / 2)

        raise Exception
        n_times = len(times)
        # hamiltonian_exp = expm(-1j *hamiltonian * (times[1] - times[0]))
        state = np.zeros((dim, 1))
        state[-1, 0] = 1
        states = np.zeros((dim, n_times), dtype=np.complex128)
        states[..., 0] = state.flatten()
        for i in range(n_times - 1):
            if verbose:
                print(i)
            states[..., i + 1] = expm_multiply(-1j * hamiltonian * (times[i + 1] - times[i]), states[..., i])
        z_mags = ((np.abs(states) ** 2).T @ (heisenberg.states - 1 / 2) * 2).real / 2
        czz = z_mags*z_mags_init


def magnetization(graph, h=1):
    t_final = 3
    n_times = 10**t_final
    n=graph.n
    times = np.linspace(1, 10**t_final, n_times)
    states, states_exp, states_cheb, z_mag, z_mag_exp, z_mag_cheb = generate_time_evolved_states(graph, times, verbose=True, h=h)
    # Compute z magnetization
    fig, ax = plt.subplots(2,graph.n//2, sharey=True)
    for i in range(graph.n//2):
        ax[0][i].semilogx(times, (z_mag[:,i].flatten()), label='Pade')
        ax[1][i].semilogx(times, (z_mag[:, i+graph.n//2].flatten()))
        ax[0][i].semilogx(times, (z_mag_cheb[:, i].flatten()), label='Chebyshev')
        ax[1][i].semilogx(times, (z_mag_cheb[:, i + graph.n // 2].flatten()))
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


#generate_time_evolved_states(line_graph(8), np.linspace(0, 1000, 1), verbose=True)
n = 28
print(2**n)
import time
t0 = time.time()
print('beginning')
heisenberg = qsim.evolution.hamiltonian.HamiltonianHeisenberg(line_graph(n), subspace='all', energies=(1/4, 1/2))
disorder = disorder_hamiltonian(n, subspace='all', h=1.)
print(disorder)
state = State(np.random.random((2**n, 1)))
state = state/np.linalg.norm(state)
#state = State(np.ones((2**n, 1)))
#print(state)
state = matvec_heisenberg(heisenberg, disorder, state)
print(state)
print(time.time()-t0)