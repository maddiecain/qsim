import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from odeintw import odeintw
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence

from qsim.codes.quantum_state import State
from qsim.evolution.lindblad_operators import LindbladJumpOperator
from qsim.evolution.quantum_channels import QuantumChannel
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.tools import tools

__all__ = ['LindbladMasterEquation']


class LindbladMasterEquation(object):
    def __init__(self, hamiltonians=None, jump_operators=None):
        # Jump operators is a list of LindbladNoise objects
        # Hamiltonian is a function of time
        if hamiltonians is None:
            hamiltonians = []
        if jump_operators is None:
            jump_operators = []
        self.hamiltonians = hamiltonians
        self.jump_operators = jump_operators

    def evolution_generator(self, s: State):
        res = State(np.zeros(s.shape), is_ket=s.is_ket, code=s.code, IS_subspace=s.IS_subspace, graph=s.graph)
        for i in range(len(self.hamiltonians)):
            res = res - 1j * (self.hamiltonians[i].left_multiply(s) - self.hamiltonians[i].right_multiply(s))
        for i in range(len(self.jump_operators)):
            res = res + self.jump_operators[i].liouvillian(s)
        return res

    def run_ode_solver(self, state: State, t0, tf, num=50, schedule=lambda t: None, times=None, method='RK45',
                       full_output=True, verbose=False):
        """

        :param verbose:
        :param method:
        :param times:
        :param full_output:
        :param num:
        :param schedule:
        :param state:
        :param t0:
        :param tf:
        :return:
        """
        assert not state.is_ket
        # Save s properties
        is_ket = state.is_ket
        code = state.code
        IS_subspace = state.IS_subspace
        graph = state.graph

        def f(t, s):
            if method == 'odeint':
                t, s = s, t
            if method != 'odeint':
                s = np.reshape(np.expand_dims(s, axis=0), state_shape)
            schedule(t)
            s = State(s, is_ket=is_ket, code=code, IS_subspace=IS_subspace, graph=graph)
            return np.asarray(self.evolution_generator(s)).flatten()

        # s is a ket or density matrix
        # tf is the total simulation time
        state_asarray = np.asarray(state)
        if method == 'odeint':
            # Use the odeint wrapper
            if full_output:
                if times is None:
                    times = np.linspace(0, 1, num=int(num)) * (tf - t0) + t0
                z, infodict = odeintw(f, state_asarray, times, full_output=True)
                infodict['t'] = times
                norms = np.trace(z, axis1=-2, axis2=-1)
                if verbose:
                    print('Fraction of integrator results normalized:',
                          len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
                    print('Final state norm - 1:', norms[-1] - 1)
                for i in range(z.shape[0]):
                    z[i, ...] = tools.make_valid_state(z[i, ...], is_ket=False)
                return z, infodict
            else:
                if times is None:
                    times = np.linspace(0, 1, num=int(num)) * (tf - t0) + t0
                norms = np.zeros(len(times))
                s = state_asarray.copy()
                for (i, t) in zip(range(len(times)), times):
                    if i == 0:
                        norms[i] = 1
                    else:
                        s = odeintw(f, s, [times[i - 1], times[i]], full_output=False)[-1]
                        # Normalize output?
                        norms[i] = np.trace(s)
                infodict = {'t': times}
                if verbose:
                    print('Fraction of integrator results normalized:',
                          len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
                    print('Final state norm - 1:', norms[-1] - 1)
                s = np.array([tools.make_valid_state(s, is_ket=False)])
                return s, infodict
        else:
            state_shape = state_asarray.shape
            state_asarray = state_asarray.flatten()
            if full_output:
                res = scipy.integrate.solve_ivp(f, (t0, tf), state_asarray, t_eval=times, method=method,
                                                vectorized=True)
            else:
                res = scipy.integrate.solve_ivp(f, (t0, tf), state_asarray, t_eval=[tf], method=method, vectorized=True)
            res.y = np.swapaxes(res.y, 0, 1)
            res.y = np.reshape(res.y, (-1, state_shape[0], state_shape[1]))
            norms = np.trace(res.y, axis1=-2, axis2=-1)
            if verbose:
                print('Fraction of integrator results normalized:',
                      len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
                print('Final state norm - 1:', norms[-1] - 1)

            for i in range(res.y.shape[0]):
                res.y[i, ...] = tools.make_valid_state(res.y[i, ...], is_ket=False)
            return res.y, res

    def run_trotterized_solver(self, state: State, t0, tf, num=50, schedule=lambda t: None, times=None,
                               full_output=True, verbose=False):
        """Trotterized approximation of the Schrodinger equation"""
        assert not state.is_ket

        # s is a ket specifying the initial codes
        # tf is the total simulation time
        if times is None:
            times = np.linspace(0, 1, num=int(num)) * (tf - t0) + t0
        n = len(times)
        if full_output:
            z = np.zeros((n, state.shape[0], state.shape[1]), dtype=np.complex128)
        infodict = {'t': times}
        s = state.copy()

        for (i, t) in zip(range(n), times):
            schedule(t)
            if t == times[0] and full_output:
                z[i, ...] = state
            else:
                if i != 0:
                    dt = times[i] - times[i - 1]
                else:
                    dt = times[i+1] - times[i]
                for hamiltonian in self.hamiltonians:
                    s = hamiltonian.evolve(s, dt)
                for jump_operator in self.jump_operators:
                    if isinstance(jump_operator, LindbladJumpOperator):
                        if i == 1:
                            print('Warning: Evolving by a LindbladJumpOperator involves exponentiating a large matrix.',
                                  'Consider a QuantumChannel.')
                        s = jump_operator.evolve(s, dt)
                    elif isinstance(jump_operator, QuantumChannel):
                        s = jump_operator.evolve(s, dt)
            if full_output:
                z[i, ...] = s
        if not full_output:
            z = np.array([s])
        norms = np.trace(z, axis1=-2, axis2=-1)
        if verbose:
            print('Fraction of integrator results normalized:',
                  len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
            print('Final state norm - 1:', norms[-1] - 1)
        norms = norms[:, np.newaxis, np.newaxis]
        z = z / norms
        return z, infodict

    def run_stochastic_wavefunction_solver(self, s, t0, tf, num=50, schedule=lambda t: None, times=None,
                                           full_output=True, method='trotterize', verbose=False, iterations=None):
        if iterations is None:
            iterations = 1
        # Compute probability that we have a jump
        # For the stochastic solver, we have to return a times dictionary
        assert s.is_ket
        # Save state properties
        is_ket = s.is_ket
        code = s.code
        IS_subspace = s.IS_subspace
        state_shape = s.shape
        graph = s.graph

        num_jumps = []
        jump_times = []
        jump_indices = []

        if times is None and (method == 'odeint' or method == 'trotterize'):
            times = np.linspace(0, 1, num=int(num)) * (tf - t0) + t0
        if not (method == 'odeint' or method == 'trotterize'):
            raise NotImplementedError

        schrodinger_equation = SchrodingerEquation(hamiltonians=self.hamiltonians + self.jump_operators)

        def f(t, state):
            if method == 'odeint':
                t, state = state, t
            if method != 'odeint':
                state = np.reshape(np.expand_dims(state, axis=0), state_shape)
            state = State(state, is_ket=is_ket, code=code, IS_subspace=IS_subspace)
            return np.asarray(schrodinger_equation.evolution_generator(state)).flatten()

        assert len(times) > 1
        if full_output:
            outputs = np.zeros((iterations, len(times), s.shape[0], s.shape[1]), dtype=np.complex128)
        else:
            outputs = np.zeros((iterations, s.shape[0], s.shape[1]), dtype=np.complex128)
        dt = times[1] - times[0]
        for k in range(iterations):
            jump_time = []
            jump_indices_iter = []
            num_jump = 0
            out = s.copy()
            if verbose:
                print('Iteration', k)
            for (j, time) in zip(range(times.shape[0]), times):
                # Update energies
                schedule(time)
                for i in range(len(self.jump_operators)):
                    if i == 0:
                        jumped_states, jump_probabilities = self.jump_operators[i].jump_rate(out, list(
                            range(out.number_physical_qudits)))
                        jump_probabilities = jump_probabilities * dt
                    elif i > 0:
                        js, jp = self.jump_operators[i].jump_rate(out, list(range(out.number_physical_qudits)))
                        jump_probabilities = np.concatenate([jump_probabilities, jp * dt])
                        jumped_states = np.concatenate([jumped_states, js])
                if len(self.jump_operators) == 0:
                    jump_probability = 0
                else:
                    jump_probability = np.sum(jump_probabilities)
                if np.random.uniform() < jump_probability and len(self.jump_operators) != 0:
                    # Then we should do a jump
                    num_jump += 1
                    jump_time.append(time)
                    if verbose:
                        print('Jumped with probability', jump_probability, 'at time', time)
                    jump_index = np.random.choice(list(range(len(jump_probabilities))),
                                                  p=jump_probabilities / np.sum(jump_probabilities))
                    jump_indices_iter.append(jump_index)
                    out = State(jumped_states[jump_index, ...] * np.sqrt(dt / jump_probabilities[jump_index]),
                                is_ket=is_ket, code=code, IS_subspace=IS_subspace, graph=graph)
                    # Normalization factor
                else:
                    state_asarray = np.asarray(out)
                    if method == 'odeint':
                        z = odeintw(f, state_asarray, [0, dt], full_output=False)
                        out = State(z[-1], code=code, IS_subspace=IS_subspace, is_ket=is_ket, graph=graph)

                    else:
                        for hamiltonian in self.hamiltonians:
                            out = hamiltonian.evolve(out, dt)

                        for jump_operator in self.jump_operators:
                            if isinstance(jump_operator, LindbladJumpOperator):
                                # Non-hermitian evolve
                                out = jump_operator.nh_evolve(out, dt)
                            elif isinstance(jump_operator, QuantumChannel):
                                out = jump_operator.evolve(out, dt)
                        out = State(out, code=code, IS_subspace=IS_subspace, is_ket=is_ket, graph=graph)

                    # Normalize the output
                    out = out / np.linalg.norm(out)
                    # We don't do np.sqrt(1 - jump_probability) because it is only a first order expansion,
                    # and is often inaccurate. Things will quickly diverge if the state is not normalized
                if full_output:
                    outputs[k, j, ...] = out
            jump_times.append(jump_time)
            jump_indices.append(jump_indices_iter)
            num_jumps.append(num_jump)
            if not full_output:
                outputs[k, ...] = out

        return outputs, {'t': times, 'jump_times':jump_times, 'num_jumps':num_jumps, 'jump_indices':jump_indices}

    def eig(self, state: State, k=6, which='SM', use_initial_guess=False, plot=False):
        """Returns a list of the eigenvalues and the corresponding valid density matrix.
        Functionality only for if input is a density matrix."""
        assert not state.is_ket
        is_ket = state.is_ket
        code = state.code
        IS_subspace = state.IS_subspace
        graph = state.graph
        state_shape = state.shape


        def f(flattened):
            s = State(flattened.reshape(state_shape))
            res = self.evolution_generator(s)
            return res.reshape(flattened.shape)

        state_flattened = state.flatten()

        lindbladian = LinearOperator(shape=(len(state_flattened), len(state_flattened)), dtype=np.complex128, matvec=f)

        if not use_initial_guess:
            v0 = None
        else:
            v0 = state_flattened
        try:
            eigvals, eigvecs = eigs(lindbladian, k=k, which=which, v0=v0)
        except ArpackNoConvergence as exception_info:
            eigvals = exception_info.eigenvalues
            eigvecs = exception_info.eigenvectors
        if eigvals.size != 0:
            # Do some basic reshaping to post process the results and select for only the steady states
            eigvecs = np.reshape(eigvecs, [state.shape[0], state.shape[1], eigvecs.shape[-1]])
            eigvecs = np.moveaxis(eigvecs, -1, 0)
            # If there is one steady s, then normalize it because it is a valid density matrix

            if plot:
                eigvals_cc = eigvals.conj()
                eigvals_real = np.concatenate((eigvals.real, eigvals.real))
                eigvals_complex = np.concatenate(
                    (eigvals_cc.imag, eigvals_cc.imag))
                plt.hlines(0, xmin=-1, xmax=.1)
                plt.vlines(0, ymin=-100, ymax=100)
                plt.scatter(eigvals_real, eigvals_complex, c='m', s=4)
                plt.show()
            return eigvals, eigvecs
        else:
            return None, None

    def steady_state(self, state: State, k=6, which='LR', use_initial_guess=False, plot=False, tol=1e-8):
        """Returns a list of the eigenvalues and the corresponding valid density matrix."""
        assert not state.is_ket
        state_shape = state.shape

        def f(flattened):
            s = State(flattened.reshape(state_shape))
            res = self.evolution_generator(s)
            return res.reshape(flattened.shape)

        state_flattened = state.flatten()
        lindbladian = LinearOperator(shape=(len(state_flattened), len(state_flattened)), dtype=np.complex128, matvec=f)
        if not use_initial_guess:
            v0 = None
        else:
            v0 = state_flattened
        try:
            eigvals, eigvecs = eigs(lindbladian, k=k, which=which, v0=v0)
        except ArpackNoConvergence as exception_info:
            eigvals = exception_info.eigenvalues
            eigvecs = exception_info.eigenvectors
        if eigvals.size != 0:
            # Do some basic reshaping to post process the results and select for only the steady states
            eigvecs = np.moveaxis(eigvecs, -1, 0)
            eigvecs = np.reshape(eigvecs, [eigvecs.shape[0], state_shape[0], state_shape[1]])
            steady_state_indices = np.argwhere(eigvals.real > -1 * tol).T[0]
            steady_state_eigvecs = eigvecs[steady_state_indices, :, :]
            # If there is one steady s, then normalize it because it is a valid density matrix
            if steady_state_eigvecs.shape[0] == 1:
                steady_state_eigvecs[0, :, :] = steady_state_eigvecs[0, :, :] / np.trace(steady_state_eigvecs[0, :, :])
                print('Steady state is a valid density matrix:',
                      tools.is_valid_state(steady_state_eigvecs[0, :, :], verbose=False))
            steady_state_eigvals = eigvals[steady_state_indices]
            if plot:
                steady_state_eigvals_cc = steady_state_eigvals.conj()
                steady_state_eigvals_real = np.concatenate((steady_state_eigvals.real, steady_state_eigvals.real))
                steady_state_eigvals_complex = np.concatenate(
                    (steady_state_eigvals_cc.imag, steady_state_eigvals_cc.imag))
                plt.hlines(0, xmin=-1, xmax=.1)
                plt.vlines(0, ymin=-100, ymax=100)
                plt.scatter(steady_state_eigvals_real, steady_state_eigvals_complex, c='m', s=4)
                plt.show()
            return steady_state_eigvals, steady_state_eigvecs
        else:
            return None, None

    def dg(self, state: State, k=6, which='LR', use_initial_guess=False, tol=1e-8):
        eigval, eigvec = self.eig(state, k=k, which=which, use_initial_guess=use_initial_guess, plot=False)
        nonzero = eigval[eigval.real < -1 * tol].real
        where_max = np.argmax(nonzero)
        min_eigval = np.abs(nonzero[where_max])
        return min_eigval

    def edg(self, state: State, k=6, which='LR', use_initial_guess=False, tol=1e-8):
        dim = state.dimension
        eigval, eigvec = self.steady_state(state, k=k, which=which, use_initial_guess=use_initial_guess, plot=False)
        steady_state = eigvec[np.argwhere(eigval[np.abs(eigval) <= tol])[0, 0], :, :]
        steady_state = steady_state / np.trace(steady_state)
        # Diagonalize the steady state
        ss_eigvals, ss_eigvecs = np.linalg.eigh(steady_state)
        where_nonzero = np.abs(ss_eigvals) > tol
        if np.sum(where_nonzero) < dim:
            print('Steady state is not full rank.')
        projected_ss_eigvals = where_nonzero * ss_eigvecs
        P = projected_ss_eigvals @ projected_ss_eigvals.conj().T
        Q = np.identity(dim)-P

        state_shape = state.shape

        def f(flattened):
            s = State(flattened.reshape(state_shape))
            res = P @ self.evolution_generator(P @ s @ P) @ P + Q @ self.evolution_generator(Q @ s @ P) @ P + \
                  P @ self.evolution_generator(P @ s @ Q) @ Q
            return res.reshape(flattened.shape)

        state_flattened = state.flatten()

        lindbladian = LinearOperator(shape=(len(state_flattened), len(state_flattened)), dtype=np.complex128, matvec=f)

        if not use_initial_guess:
            v0 = None
        else:
            v0 = state_flattened
        try:
            eigvals, eigvecs = eigs(lindbladian, k=k, which=which, v0=v0)
        except ArpackNoConvergence as exception_info:
            eigvals = exception_info.eigenvalues
        if eigvals.size == 0:
            return None

        nonzero = np.abs(eigvals[np.abs(eigvals) > tol])
        where_max = np.argmin(nonzero)
        min_eigval = np.abs(nonzero[where_max])
        return min_eigval

