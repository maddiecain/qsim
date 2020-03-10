from typing import Tuple
from qsim.tools import operations, tools
import numpy as np
from qsim.state import *


class LindbladNoise(object):
    def __init__(self, povm=np.array([]), weights=None):
        # POVM and weights are numpy arrays
        self.povm = povm
        if weights is None:
            # Is this a good default?
            self.weights = np.array([1] * povm.shape[0])
        # Assert that it's a valid POVM?

    def all_qubit_channel(self, s):
        for i in range(int(np.log2(s.shape[0]))):
            s = self.channel(s, i)
        return s

    def channel(self, s, i):
        # Output placeholder
        if len(self.povm) == 0:
            return s
        a = np.zeros(s.shape)
        for j in range(len(self.povm)):
            a = a + self.weights[j] * (operations.single_qubit_operation(s, i, self.povm[j]))
        return a

    def liouvillian(self, s, i):
        a = np.zeros(s.shape)
        for j in range(len(self.povm)):
            a = a + self.weights[j] * (operations.single_qubit_operation(s, i, self.povm[j]) -
                1 / 2 * operations.left_multiply(s, i, self.povm[j].conj().T @ self.povm[j]) -
                1 / 2 * operations.right_multiply(s, i, self.povm[j].conj().T @ self.povm[j]))
        return a

    def all_qubit_liouvillian(self, s):
        a = np.zeros(s.shape)
        for i in range(int(np.log2(s.shape[-1]))):
            a = a + self.liouvillian(s, i)
        return a

    def is_valid_povm(self):
        return np.allclose(np.sum(np.transpose(self.povm.conj(), [0, 2, 1])@self.povm, axis = 0), np.identity(self.povm[0].shape[0]))


class DepolarizingNoise(LindbladNoise):
    def __init__(self, p):
        super().__init__(np.array([(np.sqrt(1 - 3 * p)) * np.identity(2), np.sqrt(p) * tools.X(), np.sqrt(p) * tools.Y(),
                                   np.sqrt(p) * tools.Z()]))
        self.p = p

    def channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return s * (1 - self.p) + self.p / 3 * (operations.single_qubit_pauli(s, i, 'X') +
                                      operations.single_qubit_pauli(s, i, 'Y') +
                                      operations.single_qubit_pauli(s, i, 'Z',))


class PauliNoise(LindbladNoise):
    def __init__(self, p: Tuple[float]):
        super().__init__(self, np.array(
            [(np.sqrt(1 - np.sum(p))) * np.identity(2), np.sqrt(p[0]) * tools.X(), np.sqrt(p[1]) * tools.Y(),
             np.sqrt(p[2]) * tools.Z()]))
        assert len(p) == 3
        self.p = p

    def channel(self, s, i: int):
        """ Perform general Pauli channel on the i-th qubit of an input density matrix

            Input:
                rho = input density matrix (as numpy.ndarray)
                i = zero-based index of qubit location to apply pauli
                ps = a 3-tuple (px, py, pz) indicating the probability
                     of applying each Pauli operator

            Returns:
                (1-px-py-pz) * rho + px * Xi * rho * Xi
                                   + py * Yi * rho * Yi
                                   + pz * Zi * rho * Zi
        """
        return (1 - sum(self.p)) * s + (self.p[0] * operations.single_qubit_pauli(s, i, 'X')
                                    + self.p[1] * operations.single_qubit_pauli(s, i, 'Y')
                                    + self.p[2] * operations.single_qubit_pauli(s, i, 'Z'))



class AmplitudeDampingNoise(LindbladNoise):
    def __init__(self, p):
        super().__init__(povm = np.array([[[1, 0], [0, np.sqrt(1 - p)]], [[0, np.sqrt(p)], [0, 0]]]))
        self.p = p

    def channel(self, s, i: int):
        # Define Kraus operators
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
        """

        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.p)]])
        K1 = np.array([[0, np.sqrt(self.p)], [0, 0]])

        return operations.single_qubit_operation(s, i, K0, is_ket=False) + \
               operations.single_qubit_operation(s, i, K1, is_ket=False)


class ZProjectionNoise(LindbladNoise):
    def __init__(self, p):
        """Zeno-type noise that projects along the sigma_z eigenstates with probability p."""
        projectors = 1 / np.sqrt(2) * np.array([[[1, 0], [0, 0]], [[0, 0], [0, -1]]])
        super().__init__(projectors, weights=(p, p))
        self.p = p



class ThermalNoise(LindbladNoise):
    def __init__(self, hamiltonian, temp):
        self.temp = temp 
        self.hamiltonian = hamiltonian
        # TODO: finish this!



