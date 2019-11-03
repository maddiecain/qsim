"""
This is a collection of useful functions that help simulate QAOA efficiently
Has been used on a standard laptop up to system size N=24

Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA

Quick start:
    1) generate graph as a networkx.Graph object
    2) f = minimizable_f(graph)
    3) calculate objective function and gradient as (F, Fgrad) = f(parameters)

"""
from scipy.optimize import minimize, OptimizeResult
import networkx as nx
import numpy as np
from timeit import default_timer as timer

from qsim.state import State
from qsim import tools, operations
from qsim.qaoa import optimize

EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1

"""
What do we actually want out of a simulation class? What are the heavy uses of this code?
Important functions:
    - Given parameters, start state, Hamiltonian, and noise model, simulate the performance on the full density
    matrix
    - Given the Hamiltonian and depth, find the optimal parameters
I'd like this to be as general as possible - Take in Hamiltonians, associate a variational parameter with
each. 
"""

"""Unitaries is a dictionary formatted in the following way:
    For each unitary to apply, associate a zero-indexed number that represents the order to apply in a given
    cycle. That should be the dictionary key. Then there should be sub-keys:
        - '
        - 'evolve': function that takes in variational parameter and state and acts on that state like
        e^{-i*v*H}, where v is the variational parameter and H is the Hamiltonian
"""


class VariationalParameter(object):
    def __init__(self, evolve, multiply, param=None):
        self.evolve = evolve
        self.multiply = multiply
        self.param = param


class HamiltonianB(VariationalParameter):
    def __init__(self, param=None):
        super().__init__(self.evolve_B, self.multiply_B, param)

    def multiply_B(self, s: State):
        out = np.zeros(s.state.shape, dtype=np.complex128)
        for i in range(s.N):
            if s.is_ket:
                out = out + operations.single_qubit_operation(s.state, i, tools.SIGMA_X_IND, is_pauli=True, is_ket=s.is_ket)
            else:
                state = s.state
                state = state.reshape((2 ** i, 2, -1), order='F').copy()
                state = np.flip(state, 1)
                state = state.reshape(s.state.shape, order='F')
                out = out + state
        s.state = out

    def evolve_B(self, s: State, beta):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        s.all_qubit_rotation(beta, tools.SX)


class HamiltonianC(VariationalParameter):
    def __init__(self, C, param=None):
        super().__init__(self.evolve_C, self.multiply_C, param)
        self.C = C

    def evolve_C(self, s, gamma):
        if s.is_ket:
            s.state = np.exp(-1j * gamma * self.C) * s.state
        else:
            s.state = np.exp(-1j * gamma * self.C) * s.state * np.exp(1j * gamma * self.C).T

    def multiply_C(self, s: State):
        s.state = self.C * s.state


class SimulateQAOA(object):
    def __init__(self, graph: nx.Graph, p, m, variational_operators=None, noise_model=None,
                error_probability=0.001, error_function=None, is_ket = True):
        self.graph = graph
        self.variational_operators = variational_operators
        self.N = self.graph.number_of_nodes()
        # Single qubit noise model (to be applied to all qubits)
        self.noise_model = noise_model
        # Depth of circuit
        self.p = p
        self.m = m
        # Hidden parameter
        self.C = self.create_C()
        self.error_probability = error_probability
        self.error_function = error_function
        if self.error_function is None:
            self.error_function = lambda p: self.error_probability
        self.is_ket = is_ket

    def create_C(self, node_to_index_map = None):
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.
        """
        C = np.zeros([2 ** self.N, 1])
        SZ = np.asarray([[1], [-1]])

        if node_to_index_map is None:
            node_to_index_map = {q: i for i, q in enumerate(self.graph.nodes)}

        for a, b in self.graph.edges:
            C += self.graph[a][b]['weight'] * operations.two_local_term(SZ, SZ, node_to_index_map[a],
                                                                        node_to_index_map[b], self.N)
        return C

    def noise_channel(self, s: State, param):
        if not self.noise_model is None:
            for i in range(self.N):
                s.state = self.noise_model(s.state, i, self.error_function(param))

    def variational_grad(self, param):
        """Calculate the objective function F and its gradient exactly
            Input:
                param = parameters of QAOA, should have dimensions (p, m) where m is the number of variational operators

            Output: (F, Fgrad)
               F = <HamC> for minimization
               Fgrad = gradient of F with respect to param
        """
        # DO NOT USE SELF.P
        p = param.shape[0]//self.m
        m = self.m
        param = param.reshape((param.shape[0]//self.m, self.m))

        # Preallocate space for storing mp+2 copies of wavefunction - necessary for efficient computation of analytic
        # gradient
        psi = np.ones(2 ** self.N) / 2 ** (self.N / 2)
        if self.is_ket:
            memo = np.zeros([2 ** self.N, 2 * m * p + 2], dtype=np.complex128)
            memo[:, 0] = psi
            s = State(np.array([psi]).T, self.N, is_ket=self.is_ket)
        else:
            memo = np.zeros([2 ** self.N, 2 ** self.N, m * p + 1], dtype=np.complex128)
            memo[..., 0] = tools.outer_product(psi, psi)
            s = State(memo[..., 0], self.N, is_ket=self.is_ket)

        # Evolving forward
        for j in range(p):
            for i in range(m):
                if self.is_ket:
                    self.variational_operators[i].evolve(s, param[j][i])
                    #self.noise_channel(s, param[j][i])
                    memo[:, j * m + i + 1] = np.squeeze(s.state.T)
                else:
                    # Goes through memo, evolves every density matrix in it, and adds one more in the j*m+i+1 position
                    # corresponding to H_i*p
                    s0_prenoise=State(memo[..., 0], self.N, is_ket=self.is_ket)
                    for k in range(m * j + i + 1):
                        s = State(memo[..., k], self.N, is_ket=self.is_ket)
                        self.variational_operators[i].evolve(s, param[j][i])
                        if k==0:
                            s0_prenoise.state=memo[..., 0]
                        self.noise_channel(s, param[j][i])
                        memo[..., k] = s.state
                    self.variational_operators[i].multiply(s0_prenoise)
                    self.noise_channel(s0_prenoise, param[j][i])
                    memo[..., m * j + i + 1] = s0_prenoise.state

        # Multiply by C
        # TODO: Make this work for a non-diagonal C
        if self.is_ket:
            memo[:, m * p + 1] = np.squeeze(self.C.T) * memo[:, m * p]
            s.state = np.array([memo[:, m * p + 1]]).T
        else:
            for k in range(m * p + 1):
                s = State(memo[..., k], self.N, is_ket=self.is_ket)
                s.state = self.C * s.state
                memo[..., k] = s.state

        # Evolving backwards, if ket:
        if self.is_ket:
            for k in range(p):
                for l in range(m):
                    self.variational_operators[m-l-1].evolve(s, -1 * param[p-k-1][m-l-1])
                    #self.noise_channel(s, param[p-k-1][m-l-1])
                    memo[:, (p + k) * m + 2 + l] = np.squeeze(s.state.T)

        # Evaluating objective function
        if self.is_ket:
            F = np.real(np.vdot(memo[:, m * p], memo[:, m * p + 1]))
        else:
            F = np.real(np.trace(memo[..., 0]))

        # evaluating gradient analytically
        Fgrad = np.zeros(m * p)
        for q in range(p):
            for r in range(m):
                # TODO: Make this work for a non-diagonal C
                if self.is_ket:
                    s = State(np.array([memo[:, m * (2 * p - q) + 1 - r]]).T, self.N, is_ket=self.is_ket)
                    self.variational_operators[r].multiply(s)
                    Fgrad[q * m + r] = -2 * np.imag(np.vdot(memo[:, q * m + r], np.squeeze(s.state.T)))
                else:
                    Fgrad[q * m + r] = 2 * np.imag(np.trace(memo[..., q * m + r + 1]))

        return F, Fgrad

    def run(self):
        psi = np.ones((2 ** self.N, 1)) / 2 ** (self.N / 2)
        if self.is_ket:
            s = State(psi, self.N, is_ket=self.is_ket)
        else:
            s = State(tools.outer_product(psi, psi), self.N, is_ket=self.is_ket)
        for i in range(self.p):
            for j in range(self.m):
                self.variational_operators[j].evolve(s, self.variational_operators[j].param[i])
                self.noise_channel(s, self.variational_operators[j].param[i])
        # Return the expected value of the cost function
        # Note that the state's defined expectation function won't work here due to the shape of C
        if self.is_ket:
            return np.real(np.vdot(s.state, self.C * s.state))
        else:
            return np.real(np.squeeze(tools.trace(self.C * s.state)))


    def find_optimal_params(self, init_param_guess=None, verbose=False):
        r"""
        Given a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j

        Uses the interpolation-based heuristic from arXiv:1812.01041

        Input:
            p_max: maximum p you want to optimize to (optional, default p_max=10)

        Output: is given in dictionary format {p: (F_p, param_p)}
            p = depth/level of QAOA, goes from 1 to p_max
            F_p = <C> achieved by the optimum found at depth p
            param_p = 2*p parameters for the QAOA at depth p
        """

        # Construct function to be passed to scipy.optimize.minimize
        min_c = min(self.C)
        max_c = max(self.C)

        # check if the node degrees are always odd or even
        degree_list = np.array([deg for (node, deg) in self.graph.degree]) % 2
        parity = None
        if np.all(degree_list % 2 == 0):
            parity = EVEN_DEGREE_ONLY
        elif np.all(degree_list % 2 == 1):
            parity = ODD_DEGREE_ONLY

        # Start the optimization process incrementally from p = 1 to p_max
        Fvals = self.p * [0]
        params = self.p * [None]

        for p in range(self.p):  # Note here, p goes from 0 to p_max - 1
            # Use heuristic to produce good initial guess of parameters
            if p == 0:
                param0 = init_param_guess
            elif p == 1:
                print(params[0])
                param0 = [params[0][0], params[0][0], params[0][1], params[0][1]]
            else:
                xp = np.linspace(0, 1, p)
                xp1 = np.linspace(0, 1, p + 1)
                param0 = np.concatenate([np.interp(xp1, xp, params[p - 1][:p]), np.interp(xp1, xp, params[p - 1][p:])])

            start = timer()
            if param0 is not None:
                results = minimize(self.variational_grad, param0, jac=True, method='BFGS')
            else:  # Run with 10 random guesses of parameters and keep best one
                # Will only apply to the lowest depth (p=0 here)
                # First run with a guess known to work most of the time
                param0 = np.ones((p + 1) * self.m)*np.pi/8
                param0[0:self.m] = param0[0:self.m]*-1
                results = minimize(self.variational_grad, param0, jac=True, method='BFGS')

                for _ in range(1, 10):
                    # Some reasonable random guess
                    param0 = np.array([np.random.rand(p + 1) * np.pi / 2, -np.ones(p + 1) * np.pi / 8])
                    test_results = minimize(self.variational_grad, param0, jac=True, method='BFGS')
                if test_results.fun < results.fun:  # found a better minimum
                    results = test_results

            if verbose:
                end = timer()
                print(
                    f'-- p={p + 1}, F = {results.fun:0.3f} / {min_c}, nfev={results.nfev}, time={end - start:0.2f} s')

            Fvals[p] = np.real(results.fun)
            params[p] = optimize.fix_param_gauge(results.x, degree_parity=parity)

        return [OptimizeResult(p=p,
                               f_val=f_val,
                               gammas=param[:p],
                               betas=param[p:],
                               min_c=min_c,
                               max_c=max_c)
                for p, f_val, param in zip(range(1, self.p + 1), Fvals, params)]
