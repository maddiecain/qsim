import numpy as np

class Operator(object):
    def __init__(self, operator, basis=None):
        self.operator = operator
        self.N = int(np.sqrt(dmatrix.shape[-1]))
        if basis is None:
            self.basis = np.identity(2 ** self.N)

class HermitianOperator(Operator):
    def __init__(self):
        pass