from qsim.codes.quantum_state import State
from odeintw import odeintw
import numpy as np
import scipy.integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

__all__ = ['SchrodingerEquation']


class SchrodingerEquation(object):
    def __init__(self, hamiltonians=None):
        # Hamiltonian is a function of time
        if hamiltonians is None:
            hamiltonians = []
        self.hamiltonians = hamiltonians

    def evolution_generator(self, state: State):
        res = State(np.zeros(state.shape), is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)
        for i in range(len(self.hamiltonians)):
            res = res - 1j * self.hamiltonians[i].left_multiply(state)
        return res

    def evolve(self, state: State, time):
        sparse_hamiltonian = csr_matrix((state.dimension, state.dimension))
        for i in range(len(self.hamiltonians)):
            sparse_hamiltonian = sparse_hamiltonian + self.hamiltonians[i].hamiltonian
        if state.is_ket:
            return expm_multiply(-1j * time * sparse_hamiltonian, state)
        else:
            return

    def run_ode_solver(self, state: State, t0, tf, num=50, schedule=lambda t: None, times=None, method='RK45',
                       full_output=True, verbose=False):
        """Numerically integrates the Schrodinger equation"""
        assert state.is_ket
        # Save s properties
        is_ket = state.is_ket
        code = state.code
        IS_subspace = state.IS_subspace
        def f(t, s):
            global state
            if method == 'odeint':
                t, s = s, t
            if method != 'odeint':
                s = np.reshape(np.expand_dims(s, axis=0), state_shape)
            schedule(t)
            s = State(s, is_ket=is_ket, code=code, IS_subspace=IS_subspace)
            return np.asarray(self.evolution_generator(s)).flatten()

        # s is a ket specifying the initial codes
        # tf is the total simulation time
        state_asarray = np.asarray(state)
        if method == 'odeint':
            if times is None:
                times = np.linspace(t0, tf, num=num)
            z, infodict = odeintw(f, state_asarray, times, full_output=True)
            infodict['t'] = times
            norms = np.linalg.norm(z, axis=(-2, -1))
            if verbose:
                print('Integrator results normalized?', np.isclose(norms, 1))
            norms = norms[:, np.newaxis, np.newaxis]
            z = z / norms
            return z, infodict
        else:
            # You need to flatten the array
            state_shape = state.shape
            state_asarray = state_asarray.flatten()
            if full_output:
                res = scipy.integrate.solve_ivp(f, (t0, tf), state_asarray, t_eval=times, method=method)
            else:
                res = scipy.integrate.solve_ivp(f, (t0, tf), state_asarray, t_eval=[tf], method=method)
            res.y = np.swapaxes(res.y, 0, 1)
            res.y = np.reshape(res.y, (-1, state_shape[0], state_shape[1]))
            norms = np.linalg.norm(res.y, axis=(-2, -1))
            if verbose:
                print('Integrator results normalized?', np.isclose(norms, 1))
            norms = norms[:,np.newaxis, np.newaxis]
            res.y = res.y/norms
            return res.y, res

    def eig(self):
        # Construct a LinearOperator for the Hamiltonians
        pass

    def ground_state(self):
        pass
