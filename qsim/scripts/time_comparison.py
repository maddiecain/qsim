import numpy as np
from qsim.codes import qubit
from qsim.codes.quantum_state import State
from qsim import tools
import time
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix

n = 10
print('Timer test with', n, 'qubits')

Hb = np.zeros((2 ** n, 2 ** n), dtype=int)
for i in range(n):
    Hb = Hb + tools.tensor_product([tools.identity(i), qubit.X, tools.identity(n - i - 1)])

# Make sparse matrix for Hb
Hb_sparse = csr_matrix(Hb)
psi0 = State(np.zeros((2 ** n, 1)))
psi0[-1, -1] = 1
t0 = time.perf_counter()
res = expm_multiply(Hb, psi0)
t1 = time.perf_counter()
print('scipy expm_multiply with non-sparse matrix: ', t1 - t0)
res = expm(Hb) @ psi0
t2 = time.perf_counter()
print('numpy expm with non-spares matrix: ', t2 - t1)
res = expm_multiply(Hb_sparse, psi0)
t3 = time.perf_counter()
print('scipy expm_multiply with sparse matrix:', t3 - t2)
for j in range(n):
    # Do single qubit rotations
    psi0 = qubit.rotation(psi0, 1, j, qubit.X, is_idempotent=True)
print('qsim qubit operations:', time.perf_counter() - t3)
