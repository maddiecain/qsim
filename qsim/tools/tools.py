import numpy as np

# Define global variables for single qubit Pauli operations
SX = np.array([[0, 1], [1, 0]])
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]])

SIGMA_X_IND, SIGMA_Y_IND, SIGMA_Z_IND = (1, 2, 3)


def int_to_binary(n):
    # Converts an integer N to a 2xlog(N) binary array
    assert n >= 0
    return np.array([np.array(list(np.binary_repr(n)), dtype=int)])


def binary_to_int(b):
    # Converts a 2xlog(N) binary array to an integer
    return b.dot(2 ** np.arange(b.size)[::-1])


def tensor_product(A):
    # Full tensor product of tensors all elements in A
    a = 1
    for i in A:
        a = np.kron(a, i)
    return a


def outer_product(a, b):
    # Full tensor product of tensors a and b, returns |a><b|
    return a @ b.conj().T


def X(n=1):
    # Returns X^{\otimes N}
    x = [np.array([[0, 1], [1, 0]])] * n
    return tensor_product(x)


def Y(n=1):
    # Returns Y^{\otimes N}
    y = [np.array([[0, -1j], [1j, 0]])] * n
    return tensor_product(y)


def Z(n=1):
    # Returns Z^{\otimes N}
    z = [np.array([[1, 0], [0, -1]])] * n
    return tensor_product(z)


def hadamard(n=1):
    # Returns a Hadamard transform acting on n qubits
    h = [np.array([[1, 1], [1, -1]]) / np.sqrt(2)] * n
    return tensor_product(h)


def trace(a, b=None):
    # Operator a and basis b
    # If b is None, trace over all indices
    if b is None:
        return np.trace(a)
    return np.trace(b.T.conjugate() @ a @ b)


def is_orthonormal(B):
    return np.array_equal(np.linalg.inv(B) @ B, np.identity(B.shape[-1]))


def equal_superposition(N: int, basis=np.array([[[1], [0]], [[0], [1]]])):
    """Basis is an array of dimension (2, 2**n, 1) containing the two basis states of the code
    comprised of n qubits. N is the number of logical qubits"""
    plus = (basis[0] + basis[1])/np.sqrt(2)
    return tensor_product([plus] * N)

def multiply(state, operator, is_ket=False):
    # Multiplies a state by an operator
    if is_ket:
        return operator @ state
    else:
        return operator @ state @ operator.conj().T
