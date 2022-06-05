from scipy.optimize import minimize, OptimizeResult, brute, basinhopping
import numpy as np
from timeit import default_timer as timer
import networkx as nx
from qsim.tools.tools import tensor_product, outer_product
from qsim.codes import qubit
from qsim.evolution.hamiltonian import HamiltonianMIS


class SimulateQAOA(object):
    def __init__(self, graph: nx.Graph, hamiltonian=None, noise_model=None, noise=None, code=None, cost_hamiltonian=None):
        """Noise_model is one of channel, continuous, monte_carlo, or None."""
        self.graph = graph
        self.hamiltonian = hamiltonian
        self.noise_model = noise_model
        self.noise = noise
        self.N = self.graph.number_of_nodes()
        # Depth of circuit
        self.depth = len(self.hamiltonian)
        if code is None:
            self.code = qubit
        else:
            self.code = code
        # TODO: revisit if this is the right cost function default
        if cost_hamiltonian is None:
            cost_hamiltonian = HamiltonianMIS(graph, code=self.code)
        self.cost_hamiltonian = cost_hamiltonian

    # Code for auto-updating depth when Hamiltonian is updated
    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, val):
        self._hamiltonian = val
        self._depth = len(val)

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, val):
        self._depth = val

    def variational_grad(self, param, initial_state=None):
        """Calculate the objective function F and its gradient exactly
            Input:
                param = parameters of QAOA

            Output: (F, Fgrad)
               F = <HamC> for minimization
               Fgrad = gradient of F with respect to param
        """
        # TODO: make this work for continuous noise models
        if self.noise_model == 'continuous':
            raise NotImplementedError('Variational gradient does not currently support continuous noise model')
        param = np.asarray(param)
        # Preallocate space for storing copies of wavefunction - necessary for efficient computation of analytic
        # gradient
        if self.code.logical_code and initial_state is None:
            if isinstance(self.cost_hamiltonian, HamiltonianMIS):
                initial_state = tensor_product([self.code.logical_basis[1]] * self.N)
        elif initial_state is None:
            if isinstance(self.cost_hamiltonian, HamiltonianMIS):
                initial_state = np.zeros((self.cost_hamiltonian.hamiltonian.shape[0], 1))
                initial_state[-1, -1] = 1
            else:
                initial_state = np.ones((self.cost_hamiltonian.hamiltonian.shape[0], 1)) / \
                                np.sqrt(self.cost_hamiltonian.hamiltonian.shape[0])
        if not (self.noise_model is None or self.noise_model == 'monte_carlo'):
            # Initial s should be a density matrix
            initial_state = outer_product(initial_state, initial_state)
        psi = initial_state
        if initial_state.shape[1] == 1:
            memo = np.zeros([psi.shape[0], 2 * self.depth + 2], dtype=np.complex128)
            memo[:, 0] = np.squeeze(psi.T)
            tester = psi.copy()
        else:
            memo = np.zeros([psi.shape[0], psi.shape[0], self.depth + 1], dtype=np.complex128)
            memo[..., 0] = np.squeeze(outer_product(psi, psi))
            tester = outer_product(psi, psi)
        # Evolving forward
        for j in range(self.depth):
            if initial_state.shape[1] == 1:
                tester = self.hamiltonian[j].evolve(tester, param[j])
                memo[:, j + 1] = np.squeeze(tester.T)
            else:
                self.hamiltonian[j].evolve(tester, param[j])
                # Goes through memo, evolves every density matrix in it, and adds one more in the j*m+i+1 position
                # corresponding to H_i*p
                s0_prenoise = memo[..., 0]
                for k in range(j + 1):
                    s = memo[..., k]
                    s = self.hamiltonian[j].evolve(s, param[j])
                    if k == 0:
                        s0_prenoise = s.copy()
                    if self.noise_model is not None:
                        if not (self.noise[j] is None):
                            s = self.noise[j].evolve(s, param[j])
                    memo[..., k] = s.copy()
                s0_prenoise = self.hamiltonian[j].left_multiply(s0_prenoise)
                if self.noise_model is not None:
                    if not (self.noise[j] is None):
                        s0_prenoise = self.noise[j].evolve(s0_prenoise, param[j])
                memo[..., j + 1] = s0_prenoise.copy()

        # Multiply by cost_hamiltonian
        if initial_state.shape[1] == 1:
            memo[:, self.depth + 1] = self.cost_hamiltonian.hamiltonian @ memo[:, self.depth]

            s = np.array([memo[:, self.depth + 1]]).T
        else:
            for k in range(self.depth + 1):
                s = memo[..., k]
                s = self.cost_hamiltonian.hamiltonian * s
                memo[..., k] = s

        # Evolving backwards, if ket:
        if initial_state.shape[1] == 1:
            for k in range(self.depth):
                s = self.hamiltonian[self.depth - k - 1].evolve(s, -1 * param[self.depth - k - 1])
                memo[:, self.depth + k + 2] = np.squeeze(s.T)

        # Evaluating objective function
        if initial_state.shape[1] == 1:
            F = np.real(np.vdot(memo[:, self.depth], memo[:, self.depth + 1]))
        else:
            F = np.real(np.trace(memo[..., 0]))

        # Evaluating gradient analytically
        Fgrad = np.zeros(self.depth)
        for r in range(self.depth):
            if initial_state.shape[1] == 1:
                s = np.array([memo[:, 2 * self.depth + 1 - r]]).T
                s = self.hamiltonian[r].left_multiply(s)
                Fgrad[r] = -2 * np.imag(np.vdot(memo[:, r], np.squeeze(s.T)))
            else:
                Fgrad[r] = 2 * np.imag(np.trace(memo[..., r + 1]))
        return F, Fgrad

    def run(self, param, initial_state=None):
        if self.code.logical_code and initial_state is None:
            initial_state = tensor_product([self.code.logical_basis[1]] * self.N)
        elif initial_state is None:
            if isinstance(self.cost_hamiltonian, HamiltonianMIS):
                initial_state = np.zeros((self.cost_hamiltonian.hamiltonian.shape[0], 1))
                initial_state[-1, -1] = 1
            else:
                initial_state = np.ones((self.cost_hamiltonian.hamiltonian.shape[0], 1)) / np.sqrt(self.cost_hamiltonian.hamiltonian.shape[0])
        if not (self.noise_model is None or self.noise_model == 'monte_carlo'):
            # Initial s should be a density matrix
            initial_state = outer_product(initial_state, initial_state)
        s = initial_state
        for j in range(self.depth):
            s = self.hamiltonian[j].evolve(s, param[j])
            if self.noise_model is not None:
                if self.noise[j] is not None:
                    s = self.noise[j].evolve(s, param[j])
        # Return the expected value of the cost function
        # Note that the codes's defined expectation function won't work here due to the shape of C
        return self.cost_hamiltonian.cost_function(s)

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

    def find_initial_parameters(self, init_param_guess=None, verbose=False, initial_state=None):
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
        raise NotImplementedError
        if self.cost_hamiltonian.optimization == 'min':
            opt_c = min(self.cost_hamiltonian.hamiltonian)
            f = lambda param: self.run(param, initial_state=initial_state)
        if self.cost_hamiltonian.optimization == 'max':
            opt_c = -1 * max(self.cost_hamiltonian.hamiltonian)
            f = lambda param: -1 * self.run(param, initial_state=initial_state)

        # check if the node degrees are always odd or even
        degree_list = np.array([(self.graph.graph.degree[i], i) for i in range(self.N)]) % 2
        parity = None
        if np.all(degree_list % 2 == 0):
            parity = 0
        elif np.all(degree_list % 2 == 1):
            parity = 1
        else:
            raise NotImplementedError

        # Start the optimization process incrementally from p = 1 to p_max
        beta_first = self.depth % 2
        if not beta_first:
            Fvals = self.depth // 2 * [0]
        else:
            Fvals = (self.depth // 2 + 1) * [0]
        params = self.depth * [None]

        for p in range(self.depth // 2):  # Note here, p goes from 0 to p_max - 1
            # Use heuristic to produce good initial guess of parameters
            if p == 0:
                param0 = init_param_guess
            elif p == 1:
                param0 = np.repeat(params[0:2], 2)
            else:
                # Interpolate to find the next parameter
                xp = np.linspace(0, 1, p)
                xp1 = np.linspace(0, 1, p + 1)
                # param0 has 2p+1 entries
                param0 = np.concatenate([np.interp(xp1, xp, params[p - 1][n * p:(n + 1) * p]) for n in range(self.m)])

            start = timer()
            if param0 is not None:
                results = minimize(f, param0, jac=None, method='BFGS',
                                   tol=.01)
            else:  # Run with 10 random guesses of parameters and keep best one
                # Will only apply to the lowest depth (p=0 here)
                # First run with a guess known to work most of the time
                if not beta_first:
                    param0 = np.ones((p + 1) * self.m) * np.pi / 8
                    param0[p:2 * p] = param0[p:2 * p] * -1
                    results = self.find_parameters_minimize(param0, initial_state=initial_state)
                else:
                    param0 = np.ones((p + 1) * self.m + 1) * np.pi / 8
                    param0[p + 1:2 * p + 2] = param0[p + 1:2 * p + 2] * -1
                    results = self.find_parameters_minimize(param0, initial_state=initial_state)

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

    def find_parameters_brute(self, n=20, verbose=True, initial_state=None, ranges=None):
        r"""
        Given a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j by brute-force methods
        by evaluating on a grid
        """
        # Ranges of values to search over
        if ranges is None:
            ranges = [(0, 2 * np.pi)] * self.depth
        # Set a reasonable grid size

        # We can't use analytic gradient here
        if self.cost_hamiltonian.optimization == 'max':
            opt_c = -1 * self.cost_hamiltonian.optimum
            f = lambda param: -1 * self.run(param, initial_state=initial_state)
        else:
            opt_c = self.cost_hamiltonian.optimum
            f = lambda param: self.run(param)
        results = brute(f, ranges, Ns=n, full_output=True)

        if self.cost_hamiltonian.optimization == 'max':
            f_val = -1 * np.real(results[1])
        else:
            f_val = np.real(results[1])

        params = np.array(results[0])
        approximation_ratio = np.real(results[1]) / opt_c
        if self.cost_hamiltonian.optimization == 'max':
            opt = -1 * opt_c
        else:
            opt = opt_c
        if verbose:
            print('depth:', self.depth)
            print('f_val:', f_val)
            print('params:', params)
            print('approximation_ratio:', approximation_ratio)
            print('opt:', opt)
        return {'depth': self.depth, 'f_val': f_val, 'params': params, 'approximation_ratio': approximation_ratio,
                'opt': opt}

    def find_parameters_minimize(self, init_param_guess=None, verbose=True, initial_state=None, analytic_gradient=False,
                                 ranges=None):
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
            init_param_guess = np.random.uniform(0, 1, self.depth) * 2 * np.pi
        # Identify bounds
        if ranges is None:
            ranges = [(0, 2 * np.pi)] * self.depth

        if self.cost_hamiltonian.optimization == 'max':
            opt_c = -1 * self.cost_hamiltonian.optimum
            if not analytic_gradient:
                results = minimize(
                    lambda param: -1 * self.run(param, initial_state=initial_state),
                    init_param_guess, bounds=ranges)
            else:
                def f(param):
                    res = self.variational_grad(param, initial_state=initial_state)
                    return -1 * res[0], -1 * res[1]

                results = minimize(f, init_param_guess, bounds=ranges, jac=True)
        else:
            opt_c = self.cost_hamiltonian.optimum
            if not analytic_gradient:
                results = minimize(
                    lambda param: self.run(param, initial_state=initial_state),
                    init_param_guess, bounds=ranges)
            else:
                results = minimize(lambda param: self.variational_grad(param, initial_state=initial_state),
                                   init_param_guess, jac=True, bounds=ranges)

        # Designate function outputs
        if self.cost_hamiltonian.optimization == 'max':
            f_val = -1 * np.real(results.fun)
        else:
            f_val = np.real(results.fun)
        params = np.asarray(results.x)
        approximation_ratio = np.real(results.fun) / opt_c
        if self.cost_hamiltonian.optimization == 'max':
            opt = -1 * opt_c
        else:
            opt = opt_c

        if verbose:
            print('depth:', self.depth)
            print('f_val:', f_val)
            print('params:', params)
            print('approximation_ratio:', approximation_ratio)
            print('opt:', opt)
        return {'depth': self.depth, 'f_val': f_val, 'params': params, 'approximation_ratio': approximation_ratio,
                'opt': opt}

    def find_parameters_basinhopping(self, n=20, verbose=True, initial_state=None, init_param_guess=None,
                                     analytic_gradient=False, ranges=None):
        r"""
            Given a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j by brute-force methods
            by evaluating on a grid
            """
        # Ranges of values to search over
        if ranges is None:
            ranges = [(0, 2 * np.pi)] * self.depth
        # Set a reasonable grid size
        if init_param_guess is None:
            init_param_guess = np.random.uniform(0, 1, self.depth) * 2 * np.pi

        if self.cost_hamiltonian.optimization == 'max':
            opt_c = -1 * self.cost_hamiltonian.optimum
            if not analytic_gradient:
                f = lambda param: -1 * self.run(param, initial_state=initial_state)
            else:
                def f(param):
                    res = self.variational_grad(param, initial_state=initial_state)
                    return -1 * res[0], -1 * res[1]

                f = f
        else:
            opt_c = self.cost_hamiltonian.optimum
            if not analytic_gradient:
                f = lambda param: self.run(param, initial_state=initial_state)
            else:
                f = lambda param: self.variational_grad(param, initial_state=initial_state)

        if analytic_gradient:
            results = basinhopping(f, init_param_guess, niter=n, stepsize=.7, T=1,
                                   minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': ranges, 'jac': True})
        else:
            results = basinhopping(f, init_param_guess, niter=n, stepsize=.7, T=1,
                                   minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': ranges})
        # Designate function outputs
        if self.cost_hamiltonian.optimization == 'max':
            f_val = -1 * np.real(results.fun)
        else:
            f_val = np.real(results.fun)
        params = np.asarray(results.x)
        approximation_ratio = np.real(results.fun) / opt_c
        if self.cost_hamiltonian.optimization == 'max':
            opt = -1 * opt_c
        else:
            opt = opt_c

        if verbose:
            print('depth:', self.depth)
            print('f_val:', f_val)
            print('params:', params)
            print('approximation_ratio:', approximation_ratio)
            print('opt:', opt)
        return {'depth': self.depth, 'f_val': f_val, 'params': params, 'approximation_ratio': approximation_ratio,
                'opt': opt}
