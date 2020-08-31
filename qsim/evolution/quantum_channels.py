from qsim.tools import tools
import numpy as np
from qsim.codes.quantum_state import State
from qsim.codes import qubit
from typing import Union
from scipy import sparse
from qsim.graph_algorithms.graph import Graph


class QuantumChannel(object):
    def __init__(self, povm=lambda p: np.array([]), code=qubit, rates=(1,), IS_subspace=False, graph=None):
        """
        Basic operations for operating with a quantum channels on a codes.
        :param povm: A function taking in a probability and returning a list containing elements of a positive
        operator-valued measure :math:`M_{\\mu}`, such that :math:`\\sum_{\\mu}M_{\\mu}^{\\dagger}M_{\\mu}`.
        """
        self.povm = povm
        self.code = code
        self.rates = rates
        self.IS_subspace = IS_subspace
        self.graph = graph
        if self.IS_subspace:
            assert graph is not None
            assert isinstance(graph, Graph)

    def is_valid_povm(self):
        """
        :return: ``True`` if and only if ``povm`` is valid.
        """
        assert not (self.povm is None)
        probabilities = np.arange(0, 1, .1)
        for p in probabilities:
            povm_p = self.povm(p)
            try:
                assert np.allclose(np.sum(np.transpose(np.array(povm_p).conj(), [0, 2, 1]) @ np.array(povm_p), axis=0),
                                   np.identity(povm_p[0].shape[0]))
            except AssertionError:
                print('Is valid POVM?', False)
                return False
        print('Is valid POVM?', True)
        return True

    def channel(self, state: State, p: float, apply_to: Union[int, list] = None):
        """
        Applies ``povm`` homogeneously to the qudits identified in apply_to.

        :param p:
        :param apply_to:
        :param state: State to operate on.
        :type state: np.ndarray
        :return:
        """
        # If the input s is a ket, convert it to a density matrix
        if state.is_ket:
            print('Converting ket to density matrix.')
            state = State(tools.outer_product(state, state))
        if apply_to is None and not self.IS_subspace:
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

        if self.IS_subspace:
            povm = self.povm(p)
            for j in range(len(povm)):
                out = out + povm[j] @ state @ povm[j].conj().T
            return out
        # Handle apply_to recursively
        # Only apply to one qudit
        elif len(apply_to) == 1:
            povm = self.povm(p)
            for j in range(len(povm)):
                out = out + code.multiply(state, apply_to, povm[j])
            return out
        else:
            last_element = apply_to.pop()
            recursive_solution = self.channel(state, p, apply_to=apply_to)
            povm = self.povm(p)
            for j in range(len(povm)):
                out = out + code.multiply(recursive_solution, [last_element], povm[j])
            return out

    def evolve(self, state: State, time, threshold=.05, apply_to: Union[int, list] = None):
        if state.is_ket:
            print('Converting ket to density matrix.')
            state = State(tools.outer_product(state, state))
        if apply_to is None and not self.IS_subspace:
            apply_to = list(range(state.number_physical_qudits))
        # Assume that apply_to is a list of integers
        if isinstance(apply_to, int):
            apply_to = [apply_to]
        n = 1
        # Find a number of repetitions n small enough so that channel evolution is well approximated
        while (self.rates[0] * time) ** 2 / n > threshold:
            n += 1
        p = self.rates[0] * time / n
        s = state.copy()
        # Apply channel n times
        for i in range(n):
            s = self.channel(s, p, apply_to=apply_to)
        return s


class DepolarizingChannel(QuantumChannel):
    def __init__(self, code=qubit, rates=(1,), IS_subspace=False, graph=None):
        def povm(p):
            return np.asarray([np.sqrt(1 - p) * np.identity(code.d), np.sqrt(p / 3) * code.X, np.sqrt(p / 3) * code.Y,
                               np.sqrt(p / 3) * code.Z])

        super().__init__(povm=povm, code=code, rates=rates, IS_subspace=IS_subspace, graph=graph)

    def channel(self, state: State, p: float, apply_to: Union[int, list] = None):
        """ Perform depolarizing channel on the i-th qubit of an input density matrix

                    Input:
                        rho = input density matrix (as numpy.ndarray)
                        i = zero-based index of qubit location to apply pauli
                        p = probability of depolarization, between 0 and 1
                """
        # If the input s is a ket, convert it to a density matrix
        if state.is_ket:
            print('Converting ket to density matrix.')
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
            return state * (1 - p) + p / 3 * (code.multiply(state, apply_to, ['X']) +
                                              code.multiply(state, apply_to, ['Y']) +
                                              code.multiply(state, apply_to, ['Z']))
        else:
            last_element = apply_to.pop()
            recursive_solution = self.channel(state, p, apply_to=apply_to)
            return recursive_solution * (1 - p) + p / 3 * (
                    code.multiply(recursive_solution, [last_element], ['X']) +
                    code.multiply(recursive_solution, [last_element], ['Y']) +
                    code.multiply(recursive_solution, [last_element], ['Z']))


class PauliChannel(QuantumChannel):
    def __init__(self, code=qubit, rates=(1,), IS_subspace=False, graph=None):

        def povm(p):
            povm_p = []
            # Only include nonzero terms
            for i in range(3):
                if p[i] != 0:
                    if i == 0:
                        povm_p.append(np.sqrt(p[i]) * code.X)
                    elif i == 1:
                        povm_p.append(np.sqrt(p[i]) * code.Y)
                    elif i == 2:
                        povm_p.append(np.sqrt(p[i]) * code.Z)
            if sum(p) < 1:
                povm_p.append(np.sqrt(1 - sum(p)) * np.identity(code.d))
            return np.asarray(povm_p)

        super().__init__(povm=povm, code=code, rates=rates, IS_subspace=IS_subspace, graph=graph)

    def channel(self, state: State, p: tuple, apply_to: Union[int, list] = None):
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
        try:
            assert len(p) == 3
        except AssertionError:
            print('Length of tuple must be 3.')
        # If the input s is a ket, convert it to a density matrix
        if state.is_ket:
            print('Converting ket to density matrix.')
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
            return state * (1 - sum(p)) + (p[0] * code.multiply(state, apply_to, ['X']) +
                                           p[1] * code.multiply(state, apply_to, ['Y']) +
                                           p[2] * code.multiply(state, apply_to, ['Z']))
        else:
            last_element = apply_to.pop()
            recursive_solution = self.channel(state, p, apply_to=apply_to)
            return recursive_solution * (1 - sum(p)) + p[0] * code.multiply(recursive_solution, [last_element], ['X']) \
                   + p[1] * code.multiply(recursive_solution, [last_element], ['Y']) + p[2] * \
                   code.multiply(recursive_solution, [last_element], ['Z'])


class AmplitudeDampingChannel(QuantumChannel):
    def __init__(self, code=qubit, transition=(0, 1), rates=(1,), IS_subspace=False, graph=None):
        self.transition = transition
        # We can't really speed up the default operations from the super class, so use those
        super().__init__(code=code, rates=rates, IS_subspace=IS_subspace, graph=graph)

        # Initially generate all indices which should correspond to terms that look like \sqrt{p} and \sqrt{1-p}
        if self.IS_subspace:
            if self.code is not qubit:
                IS, nary_to_index, num_IS = graph.independent_sets_code(self.code)
            else:
                # We have already solved for this information
                IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets

            self._povm_coherence = []
            self._povm_population_decay = []
            self._povm_population_stable = []
            self._last_p = None
            self._last_povm = None

            for j in range(self.graph.n):
                num_terms_coherence = 0
                num_terms_population_decay = 0
                num_terms_population_stable = 0
                rows_coherence = np.zeros(graph.n * num_IS, dtype=int)
                columns_coherence = np.zeros(graph.n * num_IS, dtype=int)
                rows_population_decay = np.zeros(num_IS, dtype=int)
                columns_population_decay = np.zeros(num_IS, dtype=int)
                rows_population_stable = np.zeros(num_IS, dtype=int)
                columns_population_stable = np.zeros(num_IS, dtype=int)
                for i in IS:
                    # For each spin
                    if IS[i][2][j] == self.transition[0]:
                        # Higher energy state
                        # Assign population terms
                        columns_population_decay[num_terms_population_decay] = i
                        rows_population_decay[num_terms_population_decay] = i
                        num_terms_population_decay += 1

                        # Flip spin at this location
                        # Get binary representation
                        temp = IS[i][2].copy()
                        temp[j] = self.transition[1]
                        flipped_temp = tools.nary_to_int(temp, base=code.d)
                        if flipped_temp in nary_to_index:
                            # This is a valid spin flip
                            rows_coherence[num_terms_coherence] = nary_to_index[flipped_temp]
                            columns_coherence[num_terms_coherence] = i
                            num_terms_coherence += 1
                    else:
                        rows_population_stable[num_terms_population_stable] = i
                        columns_population_stable[num_terms_population_stable] = i
                        num_terms_population_stable += 1
                # Cut off the excess in the arrays
                columns_coherence = columns_coherence[:num_terms_coherence]
                rows_coherence = rows_coherence[:num_terms_coherence]
                columns_population_decay = columns_population_decay[: num_terms_population_decay]
                rows_population_decay = rows_population_decay[: num_terms_population_decay]
                columns_population_stable = columns_population_stable[: num_terms_population_stable]
                rows_population_stable = rows_population_stable[: num_terms_population_stable]
                self._povm_coherence.append(sparse.csr_matrix((np.ones(num_terms_coherence, dtype=np.complex128),
                                                               (rows_coherence, columns_coherence)),
                                                              shape=(self.graph.num_independent_sets,
                                                                     self.graph.num_independent_sets)))
                self._povm_population_decay.append(
                    sparse.csr_matrix((np.ones(num_terms_population_decay, dtype=np.complex128),
                                       (rows_population_decay, columns_population_decay)),
                                      shape=(self.graph.num_independent_sets, self.graph.num_independent_sets)))
                self._povm_population_stable.append(
                    sparse.csr_matrix((np.ones(num_terms_population_stable, dtype=np.complex128),
                                       (rows_population_stable, columns_population_stable)),
                                      shape=(self.graph.num_independent_sets, self.graph.num_independent_sets)))

        def povm(p):
            if not self.IS_subspace:
                op1 = np.identity(code.d, dtype=np.complex128)
                op1[transition[0], transition[0]] = np.sqrt(1 - p)
                op1[transition[1], transition[1]] = 1
                op2 = np.zeros((code.d, code.d), dtype=np.complex128)
                op2[transition[1], transition[0]] = np.sqrt(p)
                return np.array([op1, op2])
            else:
                # Return an array with shape[0] = 2 * n
                if self.code.logical_code:
                    raise NotImplementedError
                elif p == self._last_p:
                    # Don't regenerate the new povm if you can reuse the previously generated povm
                    return self._last_povm
                else:
                    new_povm = [self._povm_coherence[k] * np.sqrt(p) for k in range(self.graph.n)] + \
                               [self._povm_population_decay[l] * np.sqrt(1 - p) + self._povm_population_stable[l] for l
                                in range(self.graph.n)]
                    self._last_p = p
                    self._last_povm = new_povm
                    return new_povm

        # Update povm attribute
        self.povm = povm


class ZenoChannel(QuantumChannel):
    def __init__(self, code=qubit, rates=(1, 1, 1), IS_subspace=False, graph=None):
        """Zeno-type evolution."""

        def povm(p):
            povm_p = []
            # Only include nonzero terms
            # TODO: generalize this to different codes
            for i in range(3):
                if p[i] != 0:
                    if i == 0:
                        povm_p.append(np.sqrt(p[i] / 2) * np.array([[0, 1], [0, 0]]))
                        povm_p.append(np.sqrt(p[i] / 2) * np.array([[0, 0], [1, 0]]))
                    elif i == 1:
                        povm_p.append(np.sqrt(p[i] / 2) * np.array([[1, 0], [0, 0]]))
                        povm_p.append(np.sqrt(p[i] / 2) * np.array([[0, 0], [0, -1]]))
                    elif i == 2:
                        povm_p.append(np.sqrt(p[i] / 2) * np.array([[0, -1j], [0, 0]]))
                        povm_p.append(np.sqrt(p[i] / 2) * np.array([[0, 0], [1j, 0]]))
            if sum(p) < 1:
                povm_p.append(np.sqrt(1 - sum(p)) * np.identity(2))
            return np.asarray(povm_p)

        super().__init__(povm=povm, code=code, rates=rates, IS_subspace=IS_subspace, graph=graph)
