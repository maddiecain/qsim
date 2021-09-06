from qsim.codes.quantum_state import State
from odeintw import odeintw
import numpy as np
import scipy.integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply, eigsh

__all__ = ['SchrodingerEquation']


class SchrodingerEquation(object):
    def __init__(self, hamiltonians=None):
        # Hamiltonian is a function of time
        if hamiltonians is None:
            hamiltonians = []
        self.hamiltonians = hamiltonians

    @property
    def hamiltonian(self):
        ham = self.hamiltonians[0].hamiltonian
        for i in range(1, len(self.hamiltonians)):
            ham = ham + self.hamiltonians[i].hamiltonian
        return ham

    def evolution_generator(self, state: State):
        res = State(np.zeros(state.shape), is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace,
                    graph=state.graph)
        for i in range(len(self.hamiltonians)):
            #print(self.hamiltonians[i].left_multiply(state)[:10], state[:10], self.hamiltonians[i])
            res = res - 1j * self.hamiltonians[i].left_multiply(state)
        return res

    def evolve(self, state: State, time):
        assert state.is_ket
        sparse_hamiltonian = csr_matrix((state.dimension, state.dimension))
        for i in range(len(self.hamiltonians)):
            sparse_hamiltonian = sparse_hamiltonian + self.hamiltonians[i].hamiltonian
        return expm_multiply(-1j * time * sparse_hamiltonian, state)

    def run_ode_solver(self, state: State, t0, tf, num=50, schedule=lambda t: None, times=None, method='odeint',
                       full_output=True, verbose=False):
        """Numerically integrates the Schrodinger equation"""
        assert state.is_ket
        # Save s properties
        is_ket = state.is_ket
        code = state.code
        IS_subspace = state.IS_subspace
        graph = state.graph

        def f(t, s):
            global state
            if method == 'odeint':
                t, s = s, t
            if method != 'odeint':
                s = np.reshape(np.expand_dims(s, axis=0), state_shape)
            schedule(t)
            s = State(s, is_ket=is_ket, code=code, IS_subspace=IS_subspace, graph=graph)
            return np.asarray(self.evolution_generator(s)).flatten()

        # s is a ket specifying the initial codes
        # tf is the total simulation time
        state_asarray = np.asarray(state)
        if method == 'odeint':
            if full_output:
                if times is None:
                    times = np.linspace(t0, tf, num=num)
                z, infodict = odeintw(f, state_asarray, times, full_output=True)
                infodict['t'] = times
                norms = np.linalg.norm(z, axis=(-2, -1))
                if verbose:
                    print('Fraction of integrator results normalized:',
                          len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
                    print('Final state norm - 1:', norms[-1] - 1)
                norms = norms[:, np.newaxis, np.newaxis]
                z = z / norms
                return z, infodict
            else:
                if times is None:
                    times = np.linspace(t0, tf, num=int(num))
                norms = np.zeros(len(times))
                s = state_asarray.copy()
                for (i, t) in zip(range(len(times)), times):
                    if i == 0:
                        norms[i] = 1
                    else:
                        s = odeintw(f, s, [times[i - 1], times[i]], full_output=False)[-1]
                        # Normalize output?
                        norms[i] = np.linalg.norm(s)
                infodict = {'t': times}
                if verbose:
                    print('Fraction of integrator results normalized:',
                          len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
                    print('Final state norm - 1:', norms[-1] - 1)
                s = np.array([s / norms[-1]])
                return s, infodict
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
                print('Fraction of integrator results normalized:',
                      len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
                print('Final state norm - 1:', norms[-1] - 1)

            norms = norms[:, np.newaxis, np.newaxis]
            res.y = res.y / norms
            return res.y, res

    def run_trotterized_solver(self, state: State, t0, tf, num=50, schedule=lambda t: None, times=None,
                               full_output=True, verbose=False):
        """Trotterized approximation of the Schrodinger equation"""
        assert state.is_ket

        # s is a ket specifying the initial codes
        # tf is the total simulation time
        if times is None:
            times = np.linspace(t0, tf, num=num)
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
                dt = times[i] - times[i - 1]
                for hamiltonian in self.hamiltonians:
                    s = hamiltonian.evolve(s, dt)
            if full_output:
                z[i, ...] = s
        else:
            z = np.array([s])
        norms = np.linalg.norm(z, axis=(-2, -1))
        if verbose:
            print('Fraction of integrator results normalized:',
                  len(np.argwhere(np.isclose(norms, np.ones(norms.shape)) == 1)) / len(norms))
            print('Final state norm - 1:', norms[-1] - 1)
        norms = norms[:, np.newaxis, np.newaxis]
        z = z / norms
        return z, infodict

    def eig(self, k=2, which='S', return_eigenvectors=True):
        # Construct a LinearOperator for the Hamiltonians
        linear_operator = False
        ham = None
        for h in self.hamiltonians:
            if not hasattr(h, 'hamiltonian'):
                linear_operator = True
            else:
                if ham is None:
                    ham = h.hamiltonian
                else:
                    ham = ham + h.hamiltonian
        if not linear_operator:
            if k == 'all':
                try:
                    eigvals, eigvecs = np.linalg.eigh(ham.todense())
                    if isinstance(eigvecs, np.matrix):
                        eigvecs = np.squeeze(np.asarray(eigvecs))
                except:
                    eigvals, eigvecs = np.linalg.eigh(np.asarray(ham))

            else:
                # Hamiltonian is a sparse matrix
                try:
                    if return_eigenvectors:
                        if which == 'S':
                            eigvals, eigvecs = eigsh(ham, k=k, which='SA')
                        else:
                            eigvals, eigvecs = eigsh(ham, k=k, which='LA')
                    else:
                        if which == 'S':
                            eigvals = eigsh(ham, k=k, which='SA', return_eigenvectors=return_eigenvectors)
                        else:
                            eigvals = eigsh(ham, k=k, which='LA', return_eigenvectors=return_eigenvectors)
                except TypeError:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(ham.todense())
                        if isinstance(eigvecs, np.matrix):
                            eigvecs = np.squeeze(np.asarray(eigvecs))
                    except:
                        eigvals, eigvecs = np.linalg.eigh(np.asarray(ham))
                    # Return the correct array
                    if which == 'S':
                        eigvals = eigvals[0:k]
                        eigvecs = eigvecs[:, 0:k]
                    else:
                        eigvals = eigvals[-k:]
                        eigvecs = eigvecs[:, -k:]
            if return_eigenvectors:
                eigvecs = np.moveaxis(eigvecs, -1, 0)
        else:
            # Construct a LinearOperator from the Hamiltonians
            raise NotImplementedError
        if return_eigenvectors:
            # First, order the eigenvalues and eigenvectors
            eigvecs = eigvecs[eigvals.argsort()]
            eigvals = np.sort(eigvals)
            # Enforce orthogonality in the case of degenerate eigenvalues
            where_degenerate = np.argwhere(np.isclose(np.diff(eigvals), 0))
            if len(where_degenerate) > 0:
                eigvecs, r = np.linalg.qr(eigvecs.T)
                eigvecs = eigvecs.T
            return eigvals, eigvecs
        else:
            return np.sort(eigvals)

    def ground_state(self, which='S'):
        """Returns the ground state and ground state energy"""
        # Construct a LinearOperator for the Hamiltonians
        linear_operator = False

        ham = None
        for h in self.hamiltonians:
            if not hasattr(h, 'hamiltonian'):
                linear_operator = True
            else:
                if ham is None:
                    ham = h.hamiltonian
                else:
                    ham = ham + h.hamiltonian

        if not linear_operator:
            if which == 'S':
                w = 'SA'
            else:
                w = 'LA'
            try:
                eigvals, eigvecs = scipy.sparse.linalg.eigsh(ham, k=1, which=w)
                eigvecs = eigvecs.T
            except TypeError:
                try:
                    eigvals, eigvecs = np.linalg.eigh(ham.todense())
                    eigvecs = eigvecs.T
                    if isinstance(eigvecs, np.matrix):
                        eigvecs = np.squeeze(np.asarray(eigvecs))
                except:
                    eigvals, eigvecs = np.linalg.eigh(np.asarray(ham))

            # Reorder eigenvalues and eigenvectors
        else:
            # Construct a LinearOperator from the Hamiltonians
            raise NotImplementedError
        if which == 'S':
            return eigvals[0], State(eigvecs[0, np.newaxis].T, is_ket=True)
        elif which == 'L':
            return eigvals[-1], State(eigvecs[-1, np.newaxis].T, is_ket=True)
