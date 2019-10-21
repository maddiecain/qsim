import numpy as np


def int_to_binary(n):
    assert n >= 0
    return np.array([np.array(list(np.binary_repr(n)), dtype=int)])

def binary_to_int(b):
    return b.dot(2**np.arange(b.size)[::-1])

def tensor_product(a, b):
    return np.kron(a, b)

def X(n = 1):
    x = np.array([[0, 1], [1, 0]])
    for i in range(n-1):
        x = tensor_product(x, np.array([[0, 1], [1, 0]]))
    return x

def Y(n = 1):
    y = np.array([[0, -1j], [1j, 0]])
    for i in range(n - 1):
        y = tensor_product(y, np.array([[0, -1j], [1j, 0]]))
    return y

def Z(n = 1):
    z = np.array([[1, 0], [0, -1]])
    for i in range(n - 1):
        z = tensor_product(z, np.array([[1, 0], [0, -1]]))
    return z

def hadamard(n = 1):
    # Returns a Hadamard transform acting on n qubits
    h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    for i in range(n - 1):
        h = tensor_product(np.array([[1, 1], [1, -1]]/np.sqrt(2)), h)
    return h

def trace(a, b = None):
    # Operator a and basis b
    # If b is None, trace over in the standard basis
    if b is None:
        return np.trace(a)
    return np.trace(b.T.conjugate()@a@b)