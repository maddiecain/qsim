"""
This is a collection of useful functions that help simulate QAOA efficiently
Has been used on a standard laptop up to system size N=24

Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA

Quick start:
    1) generate graph as a networkx.Graph object
    2) f = minimizable_f(graph)
    3) calculate objective function and gradient as (F, Fgrad) = f(parameters)

"""
from scipy.optimize import minimize, OptimizeResult, brute
import networkx as nx
import numpy as np
from timeit import default_timer as timer

from qsim import tools
from qsim import state
from qsim.evolution.hamiltonian import HamiltonianC


class SimulateQAOA(object):
    def __init__(self, graph: nx.Graph, p, m, hamiltonian=None, noise=None, is_ket=True, code=None, C=None):
        self.graph = graph
        self.hamiltonian = hamiltonian
        self.noise = noise
        self.N = self.graph.number_of_nodes()
        # Depth of circuit
        self.p = p
        self.m = m
        if code is None:
            self.code = state.qubit
        else:
            self.code = code
        if C is None:
            C = HamiltonianC(graph, mis=True, code=self.code)
        self.C = C
        self.is_ket = is_ket

    def variational_grad(self, param, initial_state=None):
        """Calculate the objective function F and its gradient exactly
            Input:
                param = parameters of QAOA, should have dimensions (p, m) where m is the number of variational operators

            Output: (F, Fgrad)
               F = <HamC> for minimization
               Fgrad = gradient of F with respect to param
        """
        p = int(param.shape[0] / self.m)
        m = self.m
        param = param.reshape(m, p).T
        # Preallocate space for storing mp+2 copies of wavefunction - necessary for efficient computation of analytic
        # gradient
        if initial_state is None:
            psi = tools.equal_superposition(self.N, basis=self.code.logical_basis)
        else:
            psi = initial_state
        if self.is_ket:
            memo = np.zeros([2 ** self.N, 2 * m * p + 2], dtype=np.complex128)
            memo[:, 0] = np.squeeze(psi.T)
            tester = psi[:]
        else:
            memo = np.zeros([2 ** self.N, 2 ** self.N, m * p + 1], dtype=np.complex128)
            memo[..., 0] = np.squeeze(tools.outer_product(psi, psi))
            tester = tools.outer_product(psi, psi)
        # Evolving forward
        for j in range(p):
            for i in range(m):
                if self.is_ket:
                    tester = self.hamiltonian[i].evolve(tester, param[j][i], is_ket=self.is_ket)
                    memo[:, j * m + i + 1] = np.squeeze(tester.T)
                else:
                    self.hamiltonian[i].evolve(tester, param[j][i], is_ket=self.is_ket)
                    # Goes through memo, evolves every density matrix in it, and adds one more in the j*m+i+1 position
                    # corresponding to H_i*p
                    s0_prenoise = memo[..., 0]
                    for k in range(m * j + i + 1):
                        s = memo[..., k]
                        s = self.hamiltonian[i].evolve(s, param[j][i], is_ket=self.is_ket)
                        if k == 0:
                            s0_prenoise = s
                        s = self.noise[i].all_qubit_channel(s)
                        memo[..., k] = s[:]
                    s0_prenoise = self.hamiltonian[i].left_multiply(s0_prenoise, is_ket=self.is_ket)
                    s0_prenoise = self.noise[i].all_qubit_channel(s0_prenoise)
                    memo[..., m * j + i + 1] = s0_prenoise[:]

        # Multiply by C
        if self.is_ket:
            memo[:, m * p + 1] = np.squeeze(self.C.hamiltonian.T) * memo[:, m * p]
            s = np.array([memo[:, m * p + 1]]).T
        else:
            for k in range(m * p + 1):
                s = memo[..., k]
                s = self.C.hamiltonian * s
                memo[..., k] = s

        # Evolving backwards, if ket:
        if self.is_ket:
            for k in range(p):
                for l in range(m):
                    s = self.hamiltonian[m - l - 1].evolve(s, -1 * param[p - k - 1][m - l - 1], is_ket=self.is_ket)
                    memo[:, (p + k) * m + 2 + l] = np.squeeze(s.T)

        # Evaluating objective function
        if self.is_ket:
            F = np.real(np.vdot(memo[:, m * p], memo[:, m * p + 1]))
        else:
            F = np.real(np.trace(memo[..., 0]))

        # evaluating gradient analytically
        Fgrad = np.zeros(m * p)
        for q in range(p):
            for r in range(m):
                if self.is_ket:
                    s = np.array([memo[:, m * (2 * p - q) + 1 - r]]).T
                    s = self.hamiltonian[r].left_multiply(s, is_ket=self.is_ket)
                    Fgrad[q * m + r] = -2 * np.imag(np.vdot(memo[:, q * m + r], np.squeeze(s.T)))
                else:
                    Fgrad[q * m + r] = 2 * np.imag(np.trace(memo[..., q * m + r + 1]))
        return F, Fgrad

    def run(self, param, initial_state=None):
        if initial_state is None:
            initial_state = tools.equal_superposition(self.N, basis=self.code.logical_basis)
        if self.is_ket:
            s = initial_state
        else:
            s = tools.outer_product(initial_state, initial_state)

        for i in range(self.p):
            for j in range(self.m):
                s = self.hamiltonian[j].evolve(s, param[j * self.p + i], is_ket=self.is_ket)
                if not (self.noise[j] is None):
                    s = self.noise[j].all_qubit_channel(s)
        # Return the expected value of the cost function
        # Note that the state's defined expectation function won't work here due to the shape of C
        return self.C.cost_function(s, is_ket=self.is_ket)

    def fix_param_gauge(self, param, gamma_period=np.pi, beta_period=np.pi / 2, degree_parity=None):
        EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1
        """ Use symmetries to reduce redundancies in the parameter space
        This is useful for the interp heuristic that relies on smoothness of parameters

        Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA/
        """
        p = len(param) // 2

        gammas = np.array(param[:p]) / gamma_period
        betas = -np.array(param[p:2 * p]) / beta_period
        # We expect gamma to be positive and beta to be negative, so flip sign of beta for now and flip it back later

        # Reduce the parameters to be between [0, 1] * period
        gammas = gammas % 1
        betas = betas % 1

        # Use time-reversal symmetry to make first gamma small
        if (gammas[0] > 0.25 and gammas[0] < 0.5) or gammas[0] > 0.75:
            gammas = -gammas % 1
            betas = -betas % 1

        # Further simplification if all nodes have same degree parity
        if degree_parity == EVEN_DEGREE_ONLY:  # Every node has even degree
            gamma_period = np.pi / 2
            gammas = (gammas * 2) % 1
        elif degree_parity == ODD_DEGREE_ONLY:  # Every node has odd degree
            for i in range(p):
                if gammas[i] > 0.5:
                    gammas[i] = gammas[i] - 0.5
                    betas[i:] = 1 - betas[i:]

        for i in range(1, p):
            # try to impose smoothness of gammas
            delta = gammas[i] - gammas[i - 1]
            if delta >= 0.5:
                gammas[i] -= 1
            elif delta <= -0.5:
                gammas[i] += 1

            #  Try to impose smoothness of betas
            delta = betas[i] - betas[i - 1]
            if delta >= 0.5:
                betas[i] -= 1
            elif delta <= -0.5:
                betas[i] += 1

        return np.concatenate((gammas * gamma_period, -betas * beta_period, param[2 * p:])).tolist()

    def find_initial_parameters(self, init_param_guess=None, verbose=False):
        r"""
        Given a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j

        Uses the interpolation-based heuristic from arXiv:1812.01041. THIS IS ONLY VALID FOR VANILLA QAOA

        Input:
            p_max: maximum p you want to optimize to (optional, default p_max=10)

        Output: is given in dictionary format {p: (F_p, param_p)}
            p = depth/level of QAOA, goes from 1 to p_max
            F_p = <C> achieved by the optimum found at depth p
            param_p = 2*p parameters for the QAOA at depth p
        """



        # Construct function to be passed to scipy.optimize.minimize
        if self.C.optimization == 'min':
            opt_c = min(self.C.hamiltonian)
            f = lambda param: self.variational_grad(param)[0]
        if self.C.optimization == 'max':
            opt_c = -1*max(self.C.hamiltonian)
            f = lambda param: -1*self.variational_grad(param)[0]


        # check if the node degrees are always odd or even
        degree_list = np.array([deg for (node, deg) in self.graph.degree]) % 2
        parity = None
        if np.all(degree_list % 2 == 0):
            parity = 0
        elif np.all(degree_list % 2 == 1):
            parity = 1

        # Start the optimization process incrementally from p = 1 to p_max
        Fvals = self.p * [0]
        params = self.p * [None]

        for p in range(self.p):  # Note here, p goes from 0 to p_max - 1
            # Use heuristic to produce good initial guess of parameters
            if p == 0:
                param0 = init_param_guess
            elif p == 1:
                param0 = np.repeat(params[0], 2)
            else:
                # Interpolate to find the next parameter
                xp = np.linspace(0, 1, p)
                xp1 = np.linspace(0, 1, p + 1)
                param0 = np.concatenate([np.interp(xp1, xp, params[p - 1][n * p:(n + 1) * p]) for n in range(self.m)])

            start = timer()
            if param0 is not None:
                results = minimize(f, param0, jac=None, method='BFGS',
                                   tol=.01)
            else:  # Run with 10 random guesses of parameters and keep best one
                # Will only apply to the lowest depth (p=0 here)
                # First run with a guess known to work most of the time
                param0 = np.ones((p + 1) * self.m) * np.pi / 8
                param0[p:2 * p] = param0[p:2 * p] * -1
                results = minimize(f, param0, jac=None, method='BFGS',
                                   tol=.01)

                for _ in range(1, 10):
                    # Some reasonable random guess
                    param0 = np.ones((p + 1) * self.m) * np.random.rand((p + 1) * self.m) * np.pi / 2
                    param0[p:2 * p] = param0[0:p] * -1 / 4
                    test_results = minimize(f, param0, jac=None,
                                            method='BFGS', tol=.01)
                if test_results.fun < results.fun:  # found a better minimum
                    results = test_results

            if verbose:
                end = timer()
                print(
                    f'-- p={p + 1}, F = {results.fun:0.3f} / {opt_c}, nfev={results.nfev}, time={end - start:0.2f} s')

            Fvals[p] = np.real(results.fun)
            params[p] = self.fix_param_gauge(results.x, degree_parity=parity)

        if verbose:
            for p, f_val, param in zip(np.arange(1, self.p + 1), Fvals, params):
                print('p:', p)
                print('f_val:', f_val)
                print('params:', np.array(param).reshape(self.m, -1))
                print('approximation_ratio:', f_val / opt_c[0])

        return [OptimizeResult(p=p,
                               f_val=f_val,
                               params=np.array(param).reshape(self.m, -1),
                               approximation_ratio=f_val / opt_c)
                for p, f_val, param in zip(np.arange(1, self.p + 1), Fvals, params)]

    def find_parameters_brute(self, n=20, verbose=True):
        r"""
        Given a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j by brute-force methods
        by evaluating on a grid
        """
        # Ranges of values to search over
        ranges = [(0, np.pi)] * self.m * self.p
        # Set a reasonable grid size


        if self.C.optimization == 'min':
            opt_c = min(self.C.hamiltonian)
            f = self.run
        if self.C.optimization == 'max':
            opt_c = -1*max(self.C.hamiltonian)
            f = lambda param: -1*self.run(param)
        results = brute(f, ranges, Ns=n, full_output=True)

        if verbose:
            print('p:', self.p)
            if self.C.optimization == 'max':
                print('f_val:', -1*np.real(results[1]))
            else:
                print('f_val:', *np.real(results[1]))
            print('params:', np.array(results[0]).reshape(self.m, -1))
            print('approximation_ratio:', np.real(results[1]) / opt_c[0])
        return results

    def find_parameters_minimize(self, init_param_guess=None, verbose=True, initial_state=None):
        """a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j

        Uses the interpolation-based heuristic from arXiv:1812.01041

        Input:
            p_max: maximum p you want to optimize to (optional, default p_max=10)

        Output: is given in dictionary format {p: (F_p, param_p)}
            p = depth/level of QAOA, goes from 1 to p_max
            F_p = <C> achieved by the optimum found at depth p
            param_p = 2*p parameters for the QAOA at depth p
        """
        # Start the optimization process incrementally from p = 1 to p_max
        if init_param_guess is None:
            init_param_guess = np.concatenate((np.linspace(0.1, 0.5, self.p),
                                               np.linspace(-0.5, -0.1, self.p),
                                               np.zeros(self.p * (self.m - 2))))

        if self.C.optimization == 'min':
            opt_c = min(self.C.hamiltonian)
            results = minimize(lambda param: self.variational_grad(param)[0], init_param_guess)
        if self.C.optimization == 'max':
            opt_c = -1*max(self.C.hamiltonian)
            results = minimize(lambda param: -1*self.variational_grad(param)[0], init_param_guess)

        if verbose:
            print('p:', self.p)
            print('f_val:', np.real(results.fun))
            print('params:', np.array(results.x).reshape(self.m, -1))
            print('approximation_ratio:', np.real(results.fun) / opt_c[0])

        return results
