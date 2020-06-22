import math


def num_qubits(state, code):
    """Computes the number of logical qubits composing a state"""
    return int(math.log(state.shape[0], code.d) / code.n)
