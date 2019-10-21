import numpy as np
from . import tools

class State(object):
    """Contains information about the system density matrix"""
    def __init__(self, dmatrix, basis = None):
        self.dmatrix = dmatrix
        self.N = int(np.sqrt(dmatrix.shape[-1]))
        if basis is None:
            self.basis = np.identity(2**self.N)

    def is_pure_state(self):
        return np.array_equal(self.dmatrix @ self.dmatrix, self.dmatrix)

    def is_valid_dmatrix(self):
        return np.all(np.linalg.eigvals(self.dmatrix) >= 0) and np.trace(self.dmatrix) == 1

    def is_orthonormal(self, B):
        return np.array_equal(np.linalg.inv(B) @ B, np.identity(B.shape[-1]))

    def change_basis(self, B):
        """B is the new basis. B is assumed to be orthonormal."""
        T = np.linalg.solve(self.basis, B)
        self.basis = B
        self.dmatrix = T.H @ self.dmatrix @ T

    def expectation(self, operator, basis = None):
        """Operator is an Operator-class object"""
        if basis is None:
            basis = self.basis
        return

    def measurement(self, basis = None):
        if basis is None:
            basis = self.basis


class PureState(State):
    """Contains extra methods for handling pure states"""
    def __init__(self):
        pass


