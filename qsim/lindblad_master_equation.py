from qsim.tools import tools
from qsim.schrodinger_equation import SchrodingerEquation
from odeintw import odeintw
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.sparse.linalg import ArpackNoConvergence
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

    def run_ode_solver(self, state, t0, tf, num=50, schedule=None, return_infodict=False):
        """

        :param schedule:
        :param state:
        :param t0:
        :param tf:
        :param dt:
        :param func: A function which includes the time dependence of each Hamiltonian and lindblad operator.
        :param return_infodict:
        :return:
        """
        is_ket = tools.is_ket(state)
        assert not is_ket
        if schedule is None:
            def schedule(t):
                return [[1] * len(self.hamiltonians), [1] * len(self.jump_operators)]

        def f(s, t):
            # hbar is set to one
            res = np.zeros(s.shape)
            coefficients = schedule(t)
            for i in range(len(self.hamiltonians)):
                res = res - 1j * coefficients[0][i] * (self.hamiltonians[i].left_multiply(s, is_ket=is_ket) -
                                                       self.hamiltonians[i].right_multiply(s, is_ket=is_ket))
            for i in range(len(self.jump_operators)):
                res = res + coefficients[1][i] * self.jump_operators[i].liouvillian(s, is_ket=is_ket)
            return res

        # s is a ket or density matrix
        # tf is the total simulation time
        z, infodict = odeintw(f, state, np.linspace(t0, tf, num=num), full_output=True)
        if return_infodict: return z, infodict
        return z

    def run_stochastic_wavefunction_solver(self, state, t0, tf, dt, func=None):
        # Compute probability that we have a jump
        times = np.arange(t0, tf, dt)
        outputs = np.zeros((times.shape[0], state.shape[0], state.shape[1]), dtype=np.complex128)
        se = SchrodingerEquation(hamiltonians=self.hamiltonians)
        for (j, time) in zip(range(times.shape[0]), times):
            output = se.run_ode_solver(state, time, time + 2 * dt, dt, func)[-1]
            # Make a list that allows you to select the possible jumps in a random order
            order = list(range(len(self.jump_operators)))
            np.random.shuffle(order)
            for i in order:
                jump = self.jump_operators[i]
                # Compute jump probability
                jump_probability = jump.jump_rate(state) * dt
                # Jump based on jump probability
                if np.random.uniform() < jump_probability:
                    state = jump.random_jump(state)
                    # Renormalize state
                    state = state / np.linalg.norm(state)
                    break  # Don't consider any other jumps in this round
                else:
                    jump_probability_integrated = 1 - np.linalg.norm(np.squeeze(output.T)) ** 2
                    # Evolve instead
                    state = output / (1 - jump_probability_integrated) ** .5
            outputs[j, ...] = state
        return outputs

    def eig(self, state=None, shape=None, k=6, func=None, coefficients=None, which='LR'):
        """Returns a list of the eigenvalues and the corresponding valid density matrix.
        Functionality only for if input is a density matrix."""
        if state is None:
            assert not (shape is None)
            state_flattened = None
        else:
            shape = state.shape
            assert not tools.is_ket(state)
            state_flattened = state.flatten()
        lindbladian_shape = (shape[0] * shape[1], shape[0] * shape[1])

        if coefficients is None:
            coefficients = [[1] * len(self.hamiltonians), [1] * len(self.jump_operators)]
        if func is None:
            def f(flattened):
                s = flattened.reshape(shape)
                res = np.zeros(shape)
                for i in range(len(self.hamiltonians)):
                    res = res - 1j * coefficients[0][i] * (self.hamiltonians[i].left_multiply(s, is_ket=False) -
                                                           self.hamiltonians[i].right_multiply(s, is_ket=False))
                for i in range(len(self.jump_operators)):
                    res = res + coefficients[1][i] * self.jump_operators[i].liouvillian(s, is_ket=False)
                return res.reshape(flattened.shape)
            func = f

        lindbladian = LinearOperator(shape=lindbladian_shape, dtype=np.complex128, matvec=func)
        try:
            return eigs(lindbladian, k=k, which=which, v0=state_flattened)
        except ArpackNoConvergence as exception_info:
            return exception_info.eigenvalues, exception_info.eigenvectors


    def steady_state(self, state, k=1, func=None, coefficients=None):
        """Returns a list of the eigenvalues and the corresponding valid density matrix."""
        is_ket = tools.is_ket(state)
        assert not is_ket
        if coefficients is None:
            coefficients = [[1] * len(self.hamiltonians), [1] * len(self.jump_operators)]
        if func is None:
            def f(flattened):
                s = flattened.reshape(state.shape)
                res = np.zeros(state.shape)
                for i in range(len(self.hamiltonians)):
                    res = res - 1j * coefficients[0][i] * (self.hamiltonians[i].left_multiply(s, is_ket=is_ket) -
                                                           self.hamiltonians[i].right_multiply(s, is_ket=is_ket))
                for i in range(len(self.jump_operators)):
                    res = res + coefficients[1][i] * self.jump_operators[i].liouvillian(s, is_ket=is_ket)
                return res.reshape(flattened.shape)

            func = f
        state_flattened = state.flatten()
        lindbladian = LinearOperator(shape=(len(state_flattened), len(state_flattened)), dtype=np.complex128,
                                     matvec=func)
        try:
            return eigs(lindbladian, k=k, which='LR', v0=state_flattened)
        except ArpackNoConvergence as exception_info:
            return exception_info.eigenvalues, exception_info.eigenvectors
