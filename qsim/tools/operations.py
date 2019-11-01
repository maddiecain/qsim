"""
Single qubit operations
"""

import numpy as np

__all__ = ['ham_two_local_term']

def ham_two_local_term(op1, op2, ind1, ind2, N):
    r"""Utility function for conveniently create a 2-Local term op1 \otimes op2 among N spins"""
    if ind1 > ind2:
        return ham_two_local_term(op2, op1, ind2, ind1, N)

    if op1.shape != op2.shape or op1.ndim != 2:
        raise ValueError('ham_two_local_term: invalid operator input')

    if ind1 < 0 or ind2 > N-1:
        raise ValueError('ham_two_local_term: invalid input indices')

    if op1.shape[0] == 1 or op1.shape[1] == 1:
        myeye = lambda n : np.ones(np.asarray(op1.shape)**n)
    else:
        myeye = lambda n : np.eye(np.asarray(op1.shape)**n)

    return np.kron(myeye(ind1), np.kron(op1, np.kron(myeye(ind2-ind1-1), np.kron(op2, myeye(N-ind2-1)))))
