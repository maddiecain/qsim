from qsim.tools import tools
import numpy as np
from qsim.codes import qubit
from qsim.codes.quantum_state import State
import scipy.sparse as sparse
from qsim.graph_algorithms.graph import Graph


class LindbladJumpOperator(object):
    def __init__(self, jump_operators: np.ndarray, rates, code=qubit, graph=None, IS_subspace=False):
        # Assume jump operators and rates are the same length
        self._jump_operators = jump_operators
        self.rates = np.asarray(rates)
        self.code = code
        self.IS_subspace = IS_subspace
        if self.IS_subspace:
            assert isinstance(graph, Graph)
        self.graph = graph

    @property
    def jump_operators(self):
        # Add new axes so that shapes are broadcastable
        return np.sqrt(self.rates[:, np.newaxis, np.newaxis]) * self._jump_operators

    def liouvillian(self, state: State, apply_to):
        out = np.zeros(state.shape)
        if self.IS_subspace:
            for i in range(len(self.jump_operators)):
                out = out + self.jump_operators[i] @ state @ self.jump_operators[i].T - 1 / 2 * self.jump_operators[
                    i].T @ self.jump_operators[i] @ state - 1 / 2 * state @ self.jump_operators[i].T @ \
                      self.jump_operators[i]

        else:
            for i in range(len(apply_to)):
                for j in range(len(self.jump_operators)):
                    out = out + self.code.multiply(state, [apply_to[i]],
                                                   self.jump_operators[j]) - 1 / 2 * \
                          self.code.left_multiply(state, [apply_to[i]], self.jump_operators[j].T @ self.jump_operators[
                              j]) - 1 / 2 * self.code.right_multiply(
                        state, [apply_to[i]],
                        self.jump_operators[j].T @
                        self.jump_operators[j])
        return State(out, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)

    def jump_rate(self, state: State, apply_to):
        assert state.is_ket
        if isinstance(apply_to, int):
            apply_to = [apply_to]
        jump_rates = []
        jumped_states = []
        if not self.IS_subspace:
            for j in range(len(self.jump_operators)):
                for i in apply_to:
                    out = self.code.left_multiply(state, i, self.jump_operators[j])
                    jump_rates.append(np.vdot(out, out).real)
                    # IMPORTANT: add in a factor of sqrt(rates) for normalization purposes later
                    jumped_states.append(out)
        else:
            for j in range(len(self.jump_operators)):
                out = self.jump_operators[j] @ state
                jump_rates.append(np.vdot(out, out).real)
                # IMPORTANT: add in a factor of sqrt(rates) for normalization purposes later
                jumped_states.append(State(out, is_ket=state.is_ket, code=state.code,
                                           IS_subspace=state.IS_subspace))

        return np.asarray(jumped_states), np.asarray(jump_rates)

    def left_multiply(self, state: State, apply_to=None):
        # Evolve under the non-Hermitian Hamiltonian
        assert state.is_ket
        if isinstance(apply_to, int):
            apply_to = [apply_to]
        out = np.zeros(state.shape)
        if not self.IS_subspace:
            for j in range(len(self.jump_operators)):
                for i in apply_to:
                    out = out - 1j * self.code.left_multiply(state, i,
                                                             self.jump_operators[
                                                                 j].conj().T @
                                                             self.jump_operators[j])
        else:
            for j in range(len(self.jump_operators)):
                out = out - 1j * self.jump_operators[j].conj().T @ \
                      self.jump_operators[j] @ state
        return State(out / 2, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)

    def global_liouvillian(self, state: State):
        return self.liouvillian(state, list(range(state.number_physical_qudits)))


class SpontaneousEmission(LindbladJumpOperator):
    def __init__(self, transition: tuple = (0, 1), rates=None, code=qubit, IS_subspace=False,
                 graph=None):
        # jump_operators and weights are numpy arrays
        self.code = code
        self.rates = rates
        self.transition = transition
        self.IS_subspace = IS_subspace
        if not self.IS_subspace:
            jump_operator = np.zeros((self.code.d, self.code.d))
            jump_operator[self.transition[1], self.transition[0]] = 1
            self._jump_operators = [jump_operator]
        if code is None:
            code = qubit
        if rates is None:
            rates = [1]
        self._evolution_operator = None
        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            if code is not qubit:
                IS, nary_to_index, num_IS = graph.independent_sets_code(self.code)
            else:
                # We have already solved for this information
                IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets
            self._jump_operators = []
            # For each atom, consider the states spontaneous emission can generate transitions between
            # Over-allocate space
            for j in range(graph.n):
                rows = np.zeros(num_IS, dtype=int)
                columns = np.zeros(num_IS, dtype=int)
                entries = np.zeros(num_IS, dtype=int)
                num_terms = 0
                for i in IS:
                    if IS[i][2][j] == self.transition[0]:
                        # Flip spin at this location
                        # Get binary representation
                        temp = IS[i][2].copy()
                        temp[j] = self.transition[1]
                        flipped_temp = tools.nary_to_int(temp, base=code.d)
                        if flipped_temp in nary_to_index:
                            # This is a valid spin flip
                            rows[num_terms] = nary_to_index[flipped_temp]
                            columns[num_terms] = i
                            entries[num_terms] = 1
                            num_terms += 1
                # Cut off the excess in the arrays
                columns = columns[:num_terms]
                rows = rows[:num_terms]
                entries = entries[:num_terms]
                # Now, append the jump operator
                jump_operator = sparse.csr_matrix((entries, (rows, columns)), shape=(num_IS, num_IS))
                self._jump_operators.append(jump_operator)

        super().__init__(np.asarray(self._jump_operators), rates, code=code, graph=graph, IS_subspace=IS_subspace)

    @property
    def jump_operators(self):
        return np.sqrt(self.rates[0]) * self._jump_operators

    def evolution_operator(self):
        if self._evolution_operator is None:
            num_IS = self._jump_operators.shape[1]
            self._evolution_operator = sparse.csr_matrix((num_IS ** 2, num_IS ** 2))
            for jump_operator in self._jump_operators:
                # Jump operator is real, so we don't need to conjugate
                self._evolution_operator = self._evolution_operator + sparse.kron(jump_operator,
                                                                                  jump_operator) - 1 / 2 * \
                                           sparse.kron(jump_operator.T @ jump_operator, sparse.identity(num_IS)) - \
                                           1 / 2 * sparse.kron(sparse.identity(num_IS), jump_operator.T @ jump_operator)

        return self.rates[0] * self._evolution_operator

    def liouvillian(self, state, apply_to):
        out = np.zeros(state.shape)
        if isinstance(apply_to, int):
            apply_to = [apply_to]
        if self.IS_subspace:
            for i in range(self.graph.n):
                out = out + self.jump_operators[i] @ state @ self.jump_operators[i].T - 1 / 2 * self.jump_operators[
                    i].T @ self.jump_operators[i] @ state - 1 / 2 * state @ self.jump_operators[i].T @ \
                      self.jump_operators[i]

        else:
            for i in range(len(apply_to)):
                out = out + self.code.multiply(state, [apply_to[i]],
                                               self.jump_operators[0]) - 1 / 2 * self.code.left_multiply(state,
                                                                                                         [apply_to[i]],
                                                                                                         self.jump_operators[
                                                                                                             0].T @
                                                                                                         self.jump_operators[
                                                                                                             0]) - 1 / 2 \
                      * self.code.right_multiply(state, [apply_to[i]],
                                                 self.jump_operators[0].T @ self.jump_operators[0])
        return State(out, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)

    def global_liouvillian(self, state: State):
        return self.liouvillian(state, list(range(state.number_physical_qudits)))

    def jump_rate(self, state: State, apply_to):
        assert state.is_ket
        if isinstance(apply_to, int):
            apply_to = [apply_to]
        jump_rates = []
        jumped_states = []
        if not self.IS_subspace:
            for j in range(len(self.jump_operators)):
                for i in apply_to:
                    out = self.code.left_multiply(state, i, self.jump_operators[j])
                    jump_rates.append(np.vdot(out, out).real)
                    # IMPORTANT: add in a factor of sqrt(rates) for normalization purposes later
                    jumped_states.append(out)
        else:
            for j in range(len(self.jump_operators)):
                out = self.jump_operators[j] @ state
                jump_rates.append(np.vdot(out, out).real)
                # IMPORTANT: add in a factor of sqrt(rates) for normalization purposes later
                jumped_states.append(out)

        return np.asarray(jumped_states), np.asarray(jump_rates)

    def left_multiply(self, state: State, apply_to=None):
        # Left multiply by the non-Hermitian Hamiltonian
        assert state.is_ket
        if isinstance(apply_to, int):
            apply_to = [apply_to]
        out = np.zeros(state.shape)
        if not self.IS_subspace:
            for j in range(len(self.jump_operators)):
                for i in apply_to:
                    out = out - 1j * self.code.left_multiply(state, i,
                                                             self.jump_operators[j].conj().T @
                                                             self.jump_operators[j])
        else:
            for j in range(len(self.jump_operators)):
                out = out - 1j * self.jump_operators[j].conj().T @ (self.jump_operators[j] @ state)
        return State(out / 2, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace)
