from qsim.tools import tools
import numpy as np
from qsim.codes.quantum_state import State
from qsim.codes import qubit
from typing import Union, List


class QuantumChannel(object):
    def __init__(self, povm: np.ndarray = None, code=qubit):
        """
        Basic operations for operating with a quantum channels on a codes.
        :param povm: A list containing elements of a positive operator-valued measure :math:`M_{\\mu}`, such that :math:`\\sum_{\\mu}M_{\\mu}^{\\dagger}M_{\\mu}`.
        """
        if povm is None:
            self.povm = []
        else:
            self.povm = povm
        self.code = code

    def is_valid_povm(self):
        """
        :return: ``True`` if and only if ``povm`` is valid.
        """
        assert not (self.povm is None)
        return np.allclose(
            np.sum(np.transpose(np.array(self.povm).conj(), [0, 2, 1]) @ np.array(self.povm),
                   axis=0), np.identity(self.povm[0].shape[0]))

    def channel(self, state: State, apply_to: Union[int, list] = None):
        """
        Applies ``povm`` homogeneously to the qudits identified in apply_to.

        :param apply_to:
        :param state: State to operate on.
        :type state: np.ndarray
        :return:
        """
        # If the input s is a ket, convert it to a density matrix
        if state.is_ket:
            state = State(tools.outer_product(state, state))
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
        # Empty s to store the output
        out = State(np.zeros_like(state), is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)

        # Assume that apply_to is a list of integers
        if isinstance(apply_to, int):
            apply_to = [apply_to]

        if state.code.logical_code:
            # Assume that logical codes are composed of qubits
            code = self.code
        else:
            code = state.code

        # Handle apply_to recursively
        # Only apply to one qudit
        if len(apply_to) == 1:
            for j in range(len(self.povm)):
                out = out + code.multiply(state, [apply_to], self.povm[j])
            return out
        else:
            last_element = apply_to.pop()
            recursive_solution = self.channel(state, apply_to)
            for j in range(len(self.povm)):
                out = out + code.multiply(recursive_solution, [last_element], self.povm[j])
            return out


class DepolarizingChannel(QuantumChannel):
    def __init__(self, p: float = 1 / 3, code=qubit):
        super().__init__(povm=np.asarray([
            np.sqrt(1 - p) * np.identity(code.d), np.sqrt(p / 3) * code.X, np.sqrt(p / 3) * code.Y,
            np.sqrt(p / 3) * code.Z]))
        self.code = code
        self.p = p

    def channel(self, state: State, apply_to: Union[int, list] = None):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        # If the input s is a ket, convert it to a density matrix
        if state.is_ket:
            state = State(tools.outer_product(state, state))
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
        # Assume that apply_to is a list of integers
        if isinstance(apply_to, int):
            apply_to = [apply_to]

        if state.code.logical_code:
            # Assume that logical codes are composed of qubits
            code = self.code
        else:
            code = state.code
        # Handle apply_to recursively
        # Only apply to one qudit
        if len(apply_to) == 1:
            return state * (1 - self.p) + self.p / 3 * (code.multiply(state, [apply_to[0]], ['X']) +
                                                        code.multiply(state, [apply_to[0]], ['Y']) +
                                                        code.multiply(state, [apply_to[0]], ['Z']))
        else:
            last_element = apply_to.pop()
            recursive_solution = self.channel(state, apply_to)
            return recursive_solution * (1 - self.p) + self.p / 3 * (
                    code.multiply(recursive_solution, [last_element], ['X']) +
                    code.multiply(recursive_solution, [last_element], ['Y']) +
                    code.multiply(recursive_solution, [last_element], ['Z']))


class PauliChannel(QuantumChannel):
    def __init__(self, p: List[float], code=qubit):
        if isinstance(p, tuple):
            p = list(p)
        assert len(p) == 3 and sum(p) <= 1
        self.code = code
        povm = []
        # Only include nonzero terms
        for i in range(3):
            if p[i] != 0:
                if i == 0:
                    povm.append(np.sqrt(p[i]) * code.X)
                elif i == 1:
                    povm.append(np.sqrt(p[i]) * code.Y)
                elif i == 2:
                    povm.append(np.sqrt(p[i]) * code.Z)
        if sum(p) < 1:
            povm.append(np.sqrt(1 - sum(p)) * np.identity(code.d))
            p.append(1 - sum(p))
        super().__init__(povm=np.asarray(povm))
        self.p = p

    def channel(self, state: State, apply_to: Union[int, list] = None):
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
        # If the input s is a ket, convert it to a density matrix
        if state.is_ket:
            state = State(tools.outer_product(state, state))
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))

        # Assume that apply_to is a list of integers
        if isinstance(apply_to, int):
            apply_to = [apply_to]

        if state.code.logical_code:
            # Assume that logical codes are composed of qubits
            code = self.code
        else:
            code = state.code
        # Handle apply_to recursively
        # Only apply to one qudit
        if len(apply_to) == 1:
            return state * (1 - sum(self.p)) + (self.p[0] * code.multiply(state, [apply_to[0]], ['X']) +
                                                self.p[1] * code.multiply(state, [apply_to[0]], ['Y']) +
                                                self.p[3] * code.multiply(state, [apply_to[0]], ['Z']))
        else:
            last_element = apply_to.pop()
            recursive_solution = self.channel(state, apply_to)
            return recursive_solution * (1 - sum(self.p)) + self.p[0] * \
                   code.multiply(recursive_solution, [last_element], ['X']) + self.p[1] * \
                   code.multiply(recursive_solution, [last_element], ['Y']) + self.p[2] * \
                   code.multiply(recursive_solution, [last_element], ['Z'])


class AmplitudeDampingChannel(QuantumChannel):
    def __init__(self, p: float, code=qubit, transition=(0, 1)):
        assert 0 < p <= 1
        self.code = code
        self.transition = transition
        self.p = p
        povm = []
        op1 = np.identity(code.d, dtype=np.complex128)
        op1[transition[0], transition[0]] = 1
        op1[transition[1], transition[1]] = np.sqrt(1 - p)
        povm.append(op1)
        op2 = np.zeros((code.d, code.d), dtype=np.complex128)
        op2[transition[0], transition[1]] = np.sqrt(p)
        povm.append(op2)
        super().__init__(povm=np.asarray(povm))
        # We can't really speed up the default operations from the super class, so use those


class ZenoChannel(QuantumChannel):
    def __init__(self, p: List[float], code=qubit):
        """Zeno-type evolution."""
        if isinstance(p, tuple):
            p = list(p)
        assert len(p) == 3
        self.code = code
        povm = []
        # Only include nonzero terms
        # TODO: generalize this to different codes
        for i in range(3):
            if p[i] != 0:
                if i == 0:
                    povm.append(np.sqrt(p[i] / 2) * np.array([[0, 1], [0, 0]]))
                    povm.append(np.sqrt(p[i] / 2) * np.array([[0, 0], [1, 0]]))
                elif i == 1:
                    povm.append(np.sqrt(p[i] / 2) * np.array([[1, 0], [0, 0]]))
                    povm.append(np.sqrt(p[i] / 2) * np.array([[0, 0], [0, -1]]))
                elif i == 2:
                    povm.append(np.sqrt(p[i] / 2) * np.array([[0, -1j], [0, 0]]))
                    povm.append(np.sqrt(p[i] / 2) * np.array([[0, 0], [1j, 0]]))
        if sum(p) < 1:
            povm.append(np.sqrt(1 - sum(p)) * np.identity(2))
        super().__init__(povm=np.asarray(povm))
        self.p = p
