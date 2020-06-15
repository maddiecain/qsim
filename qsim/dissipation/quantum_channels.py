from qsim.tools import operations, tools
import numpy as np
import networkx as nx
from typing import Tuple
import math


class QuantumChannel(object):
    def __init__(self, povm=None, d=2):
        """
        Basic operations for operating with a quantum channels on a state.
        :param povm: A list containing elements of a positive operator-valued measure :math:`M_{\\mu}`, such that :math:`\\sum_{\\mu}M_{\\mu}^{\\dagger}M_{\\mu}`.
        """
        if povm is None:
            self.povm = []
        else:
            self.povm = povm
        self.d = d

    def is_valid_povm(self):
        """
        :return: ``True`` if and only if ``povm`` is valid.
        """
        assert not (self.povm is None)
        return np.allclose(
            np.sum(np.transpose(np.array(self.povm).conj(), [0, 2, 1]) @ np.array(self.povm),
                   axis=0), np.identity(self.povm[0].shape[0]))

    def all_qubit_channel(self, s):
        """
        Applies ``povm`` homogeneously to each qubit.

        :param s: State to operate on.
        :type s: np.ndarray
        :param d: Dimension of the qudit.
        :type d: int
        :return:
        """
        # For each qudit
        for i in range(int(math.log(s.shape[0], self.d))):
            s = self.single_qubit_channel(s, i)
        return s

    def single_qubit_channel(self, s, i: int):
        """
        Applies ``povm`` to a single qubit at index :math:`i`.

        :param i:
        :param s: State to operate on.
        :type s: np.ndarray
        :return:
        """
        # Output placeholder
        if len(self.povm) == 0:
            return s
        a = np.zeros(s.shape)
        for j in range(len(self.povm)):
            a = a + operations.single_qubit_operation(s, i, self.povm[j], d=self.d)
        return a

    def multi_qubit_channel(self, s, apply_to):
        if len(self.povm) == 0:
            return s
        a = np.zeros(s.shape)
        for j in range(len(self.povm)):
            a = a + operations.multi_qubit_operation(s, self.povm[j], apply_to[j], self.povm[j], d=self.d)
        return a

    def channel(self, s, apply_to):
        if len(self.povm) == 0:
            return s
        for j in range(len(apply_to)):
            s = self.multi_qubit_channel(s, apply_to[j])
        return s


class DepolarizingChannel(QuantumChannel):
    def __init__(self, p: float):
        super().__init__(povm=[
            np.sqrt(1 - 3 * p) * np.identity(2), np.sqrt(p) * tools.X(), np.sqrt(p) * tools.Y(),
            np.sqrt(p) * tools.Z()], d=2)
        self.p = p

    def single_qubit_channel(self, s, i):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        return s * (1 - self.p) + self.p / 3 * (operations.single_qubit_pauli(s, i, 'X') +
                                                operations.single_qubit_pauli(s, i, 'Y') +
                                                operations.single_qubit_pauli(s, i, 'Z'))

    def all_qubit_channel(self, s):
        for j in range(int(np.log2(s.shape[0]))):
            s = self.single_qubit_channel(s, j)
        return s

    def channel(self, s, apply_to):
        if len(self.povm) == 0:
            return s
        for j in range(len(apply_to)):
            s = self.single_qubit_channel(s, apply_to[j])
        return s


class PauliChannel(QuantumChannel):
    def __init__(self, p: Tuple[float, float, float]):
        assert len(p) == 3
        povm = []
        # Only include nonzero terms
        for i in range(3):
            if p[i] != 0:
                if i == 0:
                    povm.append(np.sqrt(p[i]) * tools.X())
                elif i == 1:
                    povm.append(np.sqrt(p[i]) * tools.Y())
                elif i == 2:
                    povm.append(np.sqrt(p[i]) * tools.Z())
        if p[0] + p[1] + p[2] < 1:
            povm.append(np.sqrt(1 - np.sum(p)) * np.identity(2))
        super().__init__(povm=povm)
        self.p = p

    def single_qubit_channel(self, s, i: int):
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
        temp = np.zeros_like(s)
        for i in range(3):
            if self.p[i] != 0:
                if i == 0:
                    temp = temp + self.p[i] * operations.single_qubit_pauli(s, i, 'X')
                if i == 1:
                    temp = temp + self.p[i] * operations.single_qubit_pauli(s, i, 'Y')
                if i == 2:
                    temp = temp + self.p[i] * operations.single_qubit_pauli(s, i, 'Z')
        if self.p[0] + self.p[1] + self.p[2] < 1:
            temp = temp + (1 - sum(self.p)) * s
        return temp

    def all_qubit_channel(self, s):
        for j in range(int(np.log2(s.shape[0]))):
            s = self.single_qubit_channel(s, j)
        return s

    def channel(self, s, apply_to):
        if len(self.povm) == 0:
            return s
        for j in range(len(apply_to)):
            s = self.single_qubit_channel(s, apply_to[j])
        return s


class AmplitudeDampingChannel(QuantumChannel):
    def __init__(self, p):
        assert 0 < p <= 1
        super().__init__(povm=[np.array([[1, 0], [0, np.sqrt(1 - p)]]), np.array([[0, np.sqrt(p)], [0, 0]])])
        self.p = p

    def single_qubit_channel(self, s, i: int):
        # Define Kraus operators
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

        Input:
            rho = input density matrix (as numpy.ndarray)
            i = zero-based index of qubit location to apply pauli
            p = probability of depolarization, between 0 and 1
        """
        return operations.single_qubit_operation(s, i, self.povm[0], is_ket=False) + \
               operations.single_qubit_operation(s, i, self.povm[1], is_ket=False)

    def all_qubit_channel(self, s):
        for j in range(int(np.log2(s.shape[0]))):
            s = self.single_qubit_channel(s, j)
        return s

    def channel(self, s, apply_to):
        if len(self.povm) == 0:
            return s
        for j in range(len(apply_to)):
            s = self.single_qubit_channel(s, apply_to[j])
        return s


class ZenoChannel(QuantumChannel):
    def __init__(self, p: Tuple[float, float, float]):
        """Zeno-type dissipation."""
        assert len(p) == 3
        povm = []
        # Only include nonzero terms
        for i in range(3):
            if p[i] != 0:
                if i == 0:
                    povm.append(np.sqrt(p[i]) * np.array([[0, 1], [0, 0]]))
                    povm.append(np.sqrt(p[i]) * np.array([[0, 0], [1, 0]]))
                elif i == 1:
                    povm.append(np.sqrt(p[i]) * np.array([[1, 0], [0, 0]]))
                    povm.append(np.sqrt(p[i]) * np.array([[0, 0], [0, -1]]))
                elif i == 2:
                    povm.append(np.sqrt(p[i]) * np.array([[0, -1j], [0, 0]]))
                    povm.append(np.sqrt(p[i]) * np.array([[0, 0], [1j, 0]]))
        if p[0] + p[1] + p[2] < 1:
            povm.append(np.sqrt(1 - np.sum(p)) * np.identity(2))
        super().__init__(povm=povm)
        self.p = p

    def single_qubit_channel(self, s, i: int):
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
        temp = np.zeros_like(s)
        for i in range(len(self.povm)):
            temp = temp + self.p[i] * operations.single_qubit_operation(s, i, self.povm[i])
        return temp

    def all_qubit_channel(self, s):
        for j in range(int(np.log2(s.shape[0]))):
            s = self.single_qubit_channel(s, j)
        return s

    def channel(self, s, apply_to):
        if len(self.povm) == 0:
            return s
        for j in range(len(apply_to)):
            s = self.single_qubit_channel(s, apply_to[j])
        return s
