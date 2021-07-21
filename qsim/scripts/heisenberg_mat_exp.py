import numpy as np
import scipy as sp
from qsim.scripts.mat_exp_practice import matvec, matvec_heisenberg
import qsim.evolution.hamiltonian as ham
from qsim.codes.quantum_state import State
from qsim.graph_algorithms.graph import line_graph
import time
import dill


# cheb
def chebyshev(A, v, tau, eps, p_check, return_cost=False):
    if return_cost:
        t0 = time.time()
    num_matvecs = 0

    def max_eig_estimate(A):
        eigval = sp.sparse.linalg.eigsh(A, k=1, which='LA', return_eigenvectors=False, tol=eps)
        return eigval[0]

    def min_eig_estimate(A):
        eigval = sp.sparse.linalg.eigsh(A, k=1, which='SA', return_eigenvectors=False, tol=eps)
        return eigval[0]
    print('finding eigenvalues')
    lambda_1 = min_eig_estimate(A)
    lambda_n = max_eig_estimate(A)
    l1 = (lambda_n - lambda_1) / 2
    ln = (lambda_n + lambda_1) / 2
    print(lambda_1, lambda_n)
    print('done finding eigenvalues')

    def truncation_m(k0, eps, s_norm, v):
        # used to set the actual truncation point of the cheb iteration
        m = k0 + 1
        while True:
            if np.sum(np.abs(chebs_upper[m:])) * vnorm / s_norm < eps:
                return m
            else:
                m += 1

    def chebs(m, l1, ln, tau):
        # first m chebyshev coefficients
        # first find the needed Bessel function values by backwards recursion
        bessels = np.zeros(m + 1, dtype=np.complex128)
        for i in range(m + 1):
            bessels[i] = sp.special.iv(i, -1j * tau * l1)
        # bessels[i] = 2*(i+1) / (-1j*tau * l1) * bessels[i + 1] + bessels[i + 2]
        # Now convert to Chebyshev coefficient with exponential prefactor:
        bessels *= 2 * np.exp(-1j * tau * ln)
        bessels[0] *= 0.5
        return bessels

    vnorm = np.linalg.norm(v)
    print(vnorm)

    def m_max(eps, tau, lambda_1, lambda_n):
        bound = eps / (4 * vnorm)
        print(vnorm, eps)
        if ((0.5) ** (np.exp(1) * tau * (lambda_n - lambda_1) / 2) < bound):
            return int(np.exp(1) * tau * (lambda_n - lambda_1) / 2)
        else:
            print(bound)
            print(np.log2(1 / bound))
            return int(np.log(1 / bound) / np.log(2))

    # Compute conservative upper bound for the number of terms needed
    m_upper = m_max(eps, tau, lambda_1, lambda_n)
    print(m_upper)
    chebs_upper = chebs(m_upper, l1, ln, tau)
    print(chebs_upper)
    # initialize vectors etc for cheb iteration
    v_old = v
    v_new = matvec(A, v) / l1 - ln / l1 * v
    num_matvecs += 1
    s_current = chebs_upper[1] * v_new + chebs_upper[0] * v_old
    s_norms = [np.linalg.norm(chebs_upper[0] * v), np.linalg.norm(s_current)]
    exit_test = False
    norm_test = False
    k = 1
    # loop
    while not exit_test:
        v_old = 2 * (matvec(A, v_new) / l1 - ln / l1 * v_new) - v_old
        num_matvecs += 1

        s_current = s_current + chebs_upper[k + 1] * v_old
        s_norms.append(np.linalg.norm(s_current))
        if k % p_check == 0 and not norm_test:
            r = np.linalg.norm(s_current) / s_norms[k + 1 - p_check]
            if np.abs(r - 1) < 0.1:
                k0 = k + 1
                norm_test = True
                m_upper = truncation_m(k0, eps, s_norms[k + 1 - p_check], v)
        k += 1
        exit_test = (k == m_upper)
        v_new, v_old = v_old, v_new
    if return_cost:
        runtime = time.time() - t0
        return num_matvecs, runtime, s_current.squeeze()
    else:
        return s_current.squeeze()


# kryl
def krylov(A, v, tau, eps, p_check, nu_max, return_cost=False):
    if return_cost:
        t0 = time.time()
    num_matvecs = 0
    # pessimistic number of lanczos vectors from a priori estimate, just to bound the for loop
    # initialize lanczos process quantities
    second_run_necessary = False
    V = np.zeros((v.shape[0], nu_max), dtype=A.dtype)  # lanczos vectors
    vnorm = np.linalg.norm(v)
    V[:, 0] = v[:, 0] / vnorm
    alphas = []  # np.zeros(m, dtype=A.dtype)
    betas = []  # np.zeros(m-1, dtype=A.dtype)
    last_beta = 0
    norm_estimates = []  # np.zeros(int(1.0*m/p_check), dtype=np.float64)
    # first values:
    V[:, 1] = matvec(A, V[:, 0])
    alphas.append(np.dot(V[:, 1].conj(), V[:, 0]))
    V[:, 1] -= alphas[0] * V[:, 0]
    k = 1
    converged = False
    # do process
    while not converged:
        if k < nu_max - 1:
            last_beta = np.linalg.norm(V[:, k])
            if last_beta == 0:
                converged = True
                m_conv = k - 1
                T = sp.sparse.diags([betas[:k - 1], alphas[:k], betas[:k - 1]], offsets=[-1, 0, 1], dtype=A.dtype)
                cvec = np.zeros(k + 1, dtype=A.dtype)
                cvec[0] = 1
                cvec = vnorm * chebyshev(T, cvec, tau, eps, p_check, return_cost=False)
                break
            else:
                betas.append(last_beta)
                V[:, k] /= betas[k - 1]
                V[:, k + 1] = matvec(A, V[:, k])
                num_matvecs += 1
                alphas.append(np.dot(V[:, k + 1].conj(), V[:, k]))
                V[:, k + 1] -= alphas[k] * V[:, k] + betas[k - 1] * V[:, k - 1]
        else:
            second_run_necessary = True
            last_beta = np.linalg.norm(V[:, nu_max - 1])
            if last_beta == 0:
                converged = True
                m_conv = k - 1
                T = sp.sparse.diags([betas[:k - 1], alphas[:k], betas[:k - 1]], offsets=[-1, 0, 1], dtype=A.dtype)
                cvec = np.zeros(k + 1, dtype=A.dtype)
                cvec[0] = 1
                cvec = vnorm * chebyshev(T, cvec, tau, eps, p_check, return_cost=False)
                break
            else:
                betas.append(last_beta)
                V[:, nu_max - 1] /= betas[k - 1]
                V[:, nu_max - 3] = matvec(A, V[:, nu_max - 1])  # store matvec in nu-3
                num_matvecs += 1
                alphas.append(np.dot(V[:, nu_max - 3].conj(), V[:, nu_max - 1]))
                V[:, nu_max - 3] -= alphas[k] * V[:, nu_max - 1] + betas[k - 1] * V[:, nu_max - 2]
                # rearrange for next iteration
                V[:, [nu_max - 1, nu_max - 2, nu_max - 3]] = V[:, [nu_max - 3, nu_max - 1, nu_max - 2]]
        # convergence check
        if k % p_check == 0:
            # small matrix exponential
            T = sp.sparse.diags([betas[:k - 2], alphas[:k - 1], betas[:k - 2]], offsets=[-1, 0, 1], dtype=A.dtype)
            cvec = np.zeros(k - 1, dtype=A.dtype)
            cvec[0] = 1
            cvec = vnorm * chebyshev(T, cvec, tau, eps, p_check, return_cost=False)
            norm_estimates.append(np.linalg.norm(cvec))  # update norm list
            if k > p_check:  # to avoid indexing errors
                if np.abs(norm_estimates[int(k / p_check) - 1] / norm_estimates[
                    int(k / p_check) - 2] - 1) < 0.1:  # norm convergence test
                    cvec[0] -= vnorm
                    y = sp.sparse.linalg.spsolve(T.tocsr(), cvec)
                    error = (betas[k - 1] / tau) * np.abs(y[-1])
                    if error / norm_estimates[int(k / p_check) - 1] < eps:
                        converged = True
                        m_conv = k
                        T = sp.sparse.diags([betas[:k], alphas[:k + 1], betas[:k]], offsets=[-1, 0, 1], dtype=A.dtype)
                        cvec = np.zeros(k + 1, dtype=A.dtype)
                        cvec[0] = 1
                        cvec = vnorm * chebyshev(T, cvec, tau, eps, p_check, return_cost=False)
        k += 1
    if second_run_necessary:
        # go back through and recompute lanczos vectors
        v_out = np.matmul(V[:, :nu_max - 3], cvec[:nu_max - 3])  # first use vectors that are OK, up to nu_max-3
        for k in range(nu_max - 3, m_conv):
            if k < nu_max:
                V[:, k] = (matvec(A, V[:, k - 1]) - alphas[k - 1] * V[:, k - 1] - betas[k - 2] * V[:, k - 2]) / betas[
                    k - 1]
                num_matvecs += 1
                v_out += cvec[k] * V[:, k]  # update vout
            else:
                V[:, nu_max - 3] = (matvec(A, V[:, nu_max - 1]) - alphas[k - 1] * V[:, nu_max - 1] - betas[k - 2] * V[:,
                                                                                                                    nu_max - 2]) / \
                                   betas[k - 1]
                num_matvecs += 1
                v_out += cvec[k] * V[:, nu_max - 3]
                # rearrange for next iteration
                V[:, [nu_max - 1, nu_max - 2, nu_max - 3]] = V[:, [nu_max - 3, nu_max - 1, nu_max - 2]]
    else:
        v_out = np.matmul(V[:, :m_conv + 1], cvec)
    # return
    if return_cost:
        runtime = time.time() - t0
        return num_matvecs, runtime, v_out
    else:
        return v_out


def time_evolve(tau, eps, L=12, h=1, verbose=False):
    if verbose:
        print('starting ham')
    disorder = (np.random.random(size=L) - 1 / 2) * h
    disorder = np.array([0.848539, -0.038314, -0.170679, 0.384688, 0.00254869, 0.877209, -0.523275, -0.628435])/2
    np.set_printoptions(threshold=np.inf)
    print(repr(disorder))
    graph = line_graph(L)
    Heis = ham.HamiltonianHeisenberg(graph, energies=(1/4, 1/4), subspace='all')

    def matvec_disorder(x):
        if len(x.shape) == 2:
            return matvec_heisenberg(Heis, disorder, State(x))
        else:
            return matvec_heisenberg(Heis, disorder, State(np.expand_dims(x, axis=-1)))

    disorder_heis = sp.sparse.linalg.LinearOperator((2 ** L, 2 ** L), matvec=matvec_disorder)
    dill.dump(disorder_heis, open('heisenberg_linear_op_' + str(graph.n) + '.pickle', 'wb'))
    if verbose:
        print('done with ham')
    # initial state:
    v0 = State(np.zeros((2 ** L, 1)) )
    v0[0]=1
    p_check = 5
    nu_max = 7
    if verbose:
        print('start kryl')
    #nmatvec, runtime, expk = krylov(disorder_heis, v0, tau, eps, p_check, nu_max, return_cost=True)
    #print('done kryl, {} matvec, {} s'.format(nmatvec, runtime))
    if verbose:
        print('start cheb')
    nmatvec, runtime, expc = chebyshev(disorder_heis, v0, tau, eps, p_check, return_cost=True)
    print('done cheb, {} matvec, {} s'.format(nmatvec, runtime))


# print('diff', np.linalg.norm(expc-expk)/np.linalg.norm(expc))
# expe = sp.sparse.linalg.expm_multiply(-1j*tau*disorder_heis, v0.squeeze())
# print('done scipy', expe.shape)
# print(np.max(np.abs((expc-expe)/expe)))
# print(np.max(np.abs((expk-expe)/expe)))
n = 8
print(2 ** n)
#taus = 10 ** np.linspace(-2, 0, 3)
#tols = 10 ** np.linspace(-8, -2, 10)
time_evolve(1e0, 1e-5, L=n, verbose=True)
