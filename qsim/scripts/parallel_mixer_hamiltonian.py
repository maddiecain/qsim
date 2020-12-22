#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
from qsim.evolution.hamiltonian import HamiltonianDriver
from qsim.codes import qubit
from qsim.codes.quantum_state import State
from qsim import tools
import time
nproc = MPI.COMM_WORLD.Get_size()  # Size of communicator
iproc = MPI.COMM_WORLD.Get_rank()  # Ranks in communicator
inode = MPI.Get_processor_name()  # Node where this MPI process runs


class HamiltonianDriverMPI(object):
    def __init__(self, n, p, m, transition: tuple = (0, 1), energies: tuple = (1,), code=qubit):
        # Default is that the first element in transition is the higher energy s.
        # n-p is the number of qubits stored on a single node
        self.n = n
        self.p = p
        self.m = m
        self.transition = transition
        self.energies = energies
        self.code = code

    def evolve(self, state: State, time):
        # First handle intra-node operations
        # Our state has size 2^{n-p}, an equivalent of n-p logical qubits
        for i in range(state.number_logical_qudits):
            state = self.code.rotation(state, [i], self.energies[0] * time, self.code.X)

        # Binary representation of current process
        binary_iproc = tools.int_to_nary(iproc, size=self.p)

        # Initialize temporary storage of size 2^{n-p-m}
        temp = np.zeros((2 ** (self.n - self.p - self.m), 1), dtype=np.complex128)

        # Loop over all partitions for internode operations
        for i in range(2**self.m):
            # Initialize empty list to store reduce operations
            reqs = []

            # We need to send to 2**p processors
            for j in range(2**self.p):
                exponent = np.sum(np.abs(binary_iproc - tools.int_to_nary(j, size=self.p)))
                prefactor = np.cos(time) ** (self.p - exponent) * (-1j * np.sin(time)) ** exponent
                reqs.append(MPI.COMM_WORLD.Ireduce(prefactor *
                                                   state[i * 2 ** (self.n - self.p - self.m):
                                                         (i + 1) * 2 ** (self.n - self.p - self.m), :],
                                                   temp, root=j, op=MPI.SUM))
            # Wait for all processes to finish
            MPI.Request.Waitall(requests=reqs)
            # Now in principle we can replace the state
            state[i * 2 ** (self.n - self.p - self.m):(i + 1) * 2 ** (self.n - self.p - self.m), :] = temp

        return state


class HamiltonianMISMPI(object):
    def __init__(self):
        pass


# Number of qubits
# Size of full state is 2**n
n = 24

# Number of processors is 2**p
p = nproc

# Split each processor state into 2**m partitions, each of size 2**(n-p-m)
m = 6
times = []
# Interesting to look at total time as a number of partitions, and total time as a function of nodes
for n in [28]:
    if nproc == 1:
        # Benchmark serial
        ham = HamiltonianDriver()
        state = np.zeros((2 ** n, 1), dtype=np.complex128)

    else:
        ham = HamiltonianDriverMPI(n, p, m)

        # Size of state stored at each processor is 2**(n-p)
        state = np.zeros((2 ** (n - p), 1), dtype=np.complex128)
    if iproc == 0:
        state[0,0] = 1
    state = State(state)
    t1 = time.time_ns()
    state = ham.evolve(state, np.pi / 4)
    t2 = time.time_ns()
    print('Process', iproc,'Time', t2-t1, flush=True)
