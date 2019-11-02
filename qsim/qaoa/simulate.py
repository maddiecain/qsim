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
from qsim.simulate import Simulate
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
    def __init__(self, param=None, flag_z2_sym=False):
        super().__init__(self.evolve_B, self.multiply_B, param)
        self.flag_z2_sym = flag_z2_sym

    def multiply_B(self, s):
        # TODO: Check that this works with flag_z2_sym
        s_state = s.state
        s_temp = np.zeros(s.state.shape, dtype=np.complex128)
        if not self.flag_z2_sym:
            for i in range(s.N):
                if s.is_ket:
                    s.single_qubit_operation(i, tools.SIGMA_X_IND, is_pauli=True)
                else:
                    s.state=s.state.reshape((2**i, 2, -1), order='F').copy()
                    s.state=np.flip(s.state, 1)
                    s.state=s.state.reshape(s_state.shape, order='F')
                s_temp += s.state
                s.state = s_state
        else:
            for i in range(s.N - 1):
                s.single_qubit_operation(i, tools.SIGMA_X_IND, is_pauli=True)
                s_temp += s.state
                s.state = s_state
        s.state = s_temp
        if self.flag_z2_sym:
            s.state += np.flipud(s_state)

    def evolve_B(self, s: State, beta):
        r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
        if not self.flag_z2_sym:
            s.all_qubit_rotation(beta, tools.SX)
        else:
            s.all_qubit_rotation(beta, tools.SX, end=self.N - 1)
            # TODO: check that this line works for density matrices and not just kets
            s.state = np.cos(beta) * s.state - 1j * np.sin(beta) * np.flipud(s.state)


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


class SimulateQAOA(Simulate):
    def __init__(self, graph: nx.Graph, p, m, variational_params=None, noise_model=None, flag_z2_sym=False,
                 node_to_index_map=None, noisy=False):
        super().__init__(noise_model)
        self.graph = graph
        self.variational_params = variational_params
        self.node_to_index_map = node_to_index_map
        if self.node_to_index_map is None:
            self.node_to_index_map = {q: i for i, q in enumerate(self.graph.nodes)}
        self.N = self.graph.number_of_nodes()
        self.noisy = noisy
        self.noise_model = noise_model
        # Depth of circuit
        self.p = p
        self.m = m
        self.flag_z2_sym = flag_z2_sym
        # Hidden parameter
        self.C = self.create_C()

    def create_C(self):
        r"""
        Generate a vector corresponding to the diagonal of the C Hamiltonian.

        Setting flag_z2_sym to True means we're only working in the X^\otimes N = +1
        symmetric sector which reduces the Hilbert space dimension by a factor of 2.
        (default: True to save memory)
        """
        C = np.zeros([2 ** self.N, 1])
        SZ = np.asarray([[1], [-1]])

        for a, b in self.graph.edges:
            C += self.graph[a][b]['weight'] * operations.two_local_term(SZ, SZ, self.node_to_index_map[a],
                                                                        self.node_to_index_map[b], self.N)
        if self.flag_z2_sym:
            return C[:2 ** (self.N - 1), :]  # Restrict to first half of Hilbert space
        else:
            return C

    def variational_grad(self, param, is_ket=True):
        """Calculate the objective function F and its gradient exactly
            Input:
                param = parameters of QAOA, should have dimensions (p, m) where m is the number of variational operators

            Output: (F, Fgrad)
               F = <HamC> for minimization
               Fgrad = gradient of F with respect to param
        """
        p = param.shape[0]
        m = param.shape[1]

        # Preallocate space for storing mp+2 copies of wavefunction - necessary for efficient computation of analytic
        # gradient
        if self.flag_z2_sym:
            # Memo should have size m*p+2
            if is_ket:
                memo = np.zeros([2 ** (self.N - 1), 2 * m * p + 2], dtype=np.complex128)
                memo[:, 0] = np.ones(2 ** self.N - 1) / 2 ** ((self.N - 1) / 2)  # Initial state
            else:
                memo = np.zeros([2 ** (self.N - 1), 2 ** (self.N - 1), m * p + 1], dtype=np.complex128)
                memo[..., 0] = tools.outer_product(np.ones(2 ** self.N - 1) / 2 ** ((self.N - 1) / 2), np.ones(2 ** self.N - 1) / 2 ** ((self.N - 1) / 2))
        else:
            if is_ket:
                memo = np.zeros([2 ** self.N, 2 * m * p + 2], dtype=np.complex128)
                memo[:, 0] = np.ones(2 ** self.N) / 2 ** (self.N / 2)
            else:
                memo = np.zeros([2 ** (self.N), 2 ** (self.N), m * p + 1], dtype=np.complex128)
                memo[..., 0] = tools.outer_product(np.ones(2 ** self.N) / 2 ** (self.N / 2), np.ones(2 ** self.N) / 2 ** (self.N / 2))

        s = State(np.array([memo[..., 0]]).T, self.N, is_ket=is_ket)
        # Evolving forward
        for j in range(p):
            for i in range(m):
                if is_ket:
                    self.variational_params[i].evolve(s, param[j][i])
                    memo[:, j * m + i + 1] = np.squeeze(s.state.T)
                else:
                    # Goes through memo, evolves every density matrix in it, and adds one more in the j*m+i+1 position
                    # corresponding to H_i*p
                    for k in range(m * j + i + 1):
                        s = State(memo[..., k], self.N, is_ket=is_ket)
                        self.variational_params[i].evolve(s, param[j][i])
                        memo[..., k] = s.state
                    s = State(memo[..., 0], self.N, is_ket=is_ket)
                    self.variational_params[i].multiply(s)
                    memo[..., m * j + i + 1] = s.state

        # Multiply by C
        # TODO: Make this work for a non-diagonal C
        if is_ket:
            memo[:, m * p + 1] = np.squeeze(self.C.T) * memo[:, m * p]
            s.state = np.array([memo[:, m * p + 1]]).T
        else:
            for k in range(m * p + 1):
                s = State(memo[..., k], self.N, is_ket=is_ket)
                s.state = self.C * s.state
                memo[..., k] = s.state

        # Evolving backwards, if ket:
        if is_ket:
            for k in range(p):
                for l in range(m):
                    self.variational_params[m-l-1].evolve(s, -1 * param[p-k-1][m-l-1])
                    memo[:, (p + k) * m + 2 + l] = np.squeeze(s.state.T)

        # Evaluating objective function
        if is_ket:
            F = np.real(np.vdot(memo[:, m * p], memo[:, m * p + 1]))
        else:
            F = np.trace(memo[..., 0])

        # evaluating gradient analytically
        Fgrad = np.zeros(m * p)
        for q in range(p):
            for r in range(m):
                # TODO: Make this work for a non-diagonal C
                if is_ket:
                    s = State(np.array([memo[:, m * (2 * p - q) + 1 - r]]).T, self.N, is_ket=is_ket)
                    self.variational_params[r].multiply(s)
                    Fgrad[q * m + r] = -2 * np.imag(np.vdot(memo[:, q * m + r], np.squeeze(s.state.T)))
                else:
                    Fgrad[q * m + r] = 2 * np.imag(np.trace(memo[..., q * m + r + 1]))

        return (F, Fgrad)

    def run(self):
        if not noisy:
            s = State(np.ones((2 ** self.N, 1)), self.N, is_ket=True)
        else:
            s = State(np.ones((2 ** self.N, 2 ** self.N)), self.N, is_ket=False)
        for i in range(self.unitaries['p']):
            j = 0
            while True:
                try:
                    self.unitaries[j]['evolve'](s, self.unitaries[j]['param'])
                    if self.noisy:
                        self.noise_model(s, self.unitaries[j]['param'])
                except KeyError:
                    break
                j += 1

    def find_optimal_params(self, f, p_max=10, init_param_guess=None, verbose=True, is_ket=True):
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
        if f is None:
            # Default to standard QAOA
            f = lambda param: self.variational_grad(param, is_ket=is_ket)
        min_c = min(self.C)
        max_c = max(self.C)

        # check if the node degrees are always odd or even
        print(self.graph.degree)
        degree_list = np.array([deg for (node, deg) in self.graph.degree]) % 2
        parity = None
        if np.all(degree_list % 2 == 0):
            parity = EVEN_DEGREE_ONLY
        elif np.all(degree_list % 2 == 1):
            parity = ODD_DEGREE_ONLY

        # Start the optimization process incrementally from p = 1 to p_max
        Fvals = p_max * [0]
        params = p_max * [None]

        for p in range(p_max):  # Note here, p goes from 0 to p_max - 1
            # Use heuristic to produce good initial guess of parameters
            if p == 0:
                param0 = init_param_guess
            elif p == 1:
                param0 = [params[0][0], params[0][0], params[0][1], params[0][1]]
            else:
                xp = np.linspace(0, 1, p)
                xp1 = np.linspace(0, 1, p + 1)
                param0 = np.concatenate([np.interp(xp1, xp, params[p - 1][:p]), np.interp(xp1, xp, params[p - 1][p:])])

            start = timer()
            if param0 is not None:
                results = minimize(f, param0, jac=True, method='BFGS')
            else:  # Run with 10 random guesses of parameters and keep best one
                # Will only apply to the lowest depth (p=0 here)
                # First run with a guess known to work most of the time
                results = minimize(f, np.concatenate([np.ones(p + 1) * np.pi / 8, -np.ones(p + 1) * np.pi / 8]),
                                   jac=True, method='BFGS')

                for _ in range(1, 10):
                    # Some reasonable random guess
                    param0 = np.concatenate([np.random.rand(p + 1) * np.pi / 2, -np.ones(p + 1) * np.pi / 8])
                    test_results = minimize(f, param0, jac=True, method='BFGS')
                if test_results.fun < results.fun:  # found a better minimum
                    results = test_results

            if verbose:
                end = timer()
                print(
                    f'-- p={p + 1}, F = {results.fun:0.3f} / {min_c}, nfev={results.nfev}, time={end - start:0.2f} s')

            Fvals[p] = results.fun
            params[p] = optimize.fix_param_gauge(results.x, degree_parity=parity)

        return [OptimizeResult(p=p,
                               f_val=f_val,
                               gammas=param[:p],
                               betas=param[p:],
                               min_c=min_c,
                               max_c=max_c)
                for p, f_val, param in zip(range(1, p_max + 1), Fvals, params)]
