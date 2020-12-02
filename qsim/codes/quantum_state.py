import math
import numpy as np
from qsim.codes import qubit
from qsim.tools import tools


class State(np.ndarray):
    def __new__(cls, state, is_ket=None, code=qubit, IS_subspace=False, graph=None):
        # TODO: add code manipulation tools as class attributes
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        arr = state.view(cls).astype(np.complex128, copy=False)
        # add the new attribute to the created instance
        # Assume is_ket is a static attribute
        if is_ket is None:
            arr.is_ket = tools.is_ket(state)
        else:
            arr.is_ket = is_ket
        arr.code = code
        arr.dimension = state.shape[0]
        arr.IS_subspace = IS_subspace
        # Identify the number of physical qudits
        # TODO: identify how to deal with this if you're on a graph
        arr.graph = graph
        if IS_subspace:
            arr.number_physical_qudits = graph.n
            arr.number_logical_qudits = graph.n
        else:
            arr.number_physical_qudits = int(math.log(state.shape[0], code.d))
            arr.number_logical_qudits = int(math.log(state.shape[0], code.d) / code.n)
        # Finally, we must return the newly created object:
        return arr

    def __array_finalize__(self, arr):
        if arr is None: return
        self.is_ket = getattr(arr, 'is_ket', None)
        self.dimension = getattr(arr, 'dimension', None)
        self.code = getattr(arr, 'code', None)
        self.IS_subspace = getattr(arr, 'IS_subspace', None)
        self.graph = getattr(arr, 'graph', None)
        self.number_logical_qudits = getattr(arr, 'number_logical_qudits', None)
        self.number_physical_qudits = getattr(arr, 'number_physical_qudits', None)
