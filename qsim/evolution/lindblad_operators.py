from qsim.tools import tools
import numpy as np
from qsim.codes import qubit
from qsim.codes.quantum_state import State
import scipy.sparse as sparse
from qsim.graph_algorithms.graph import Graph
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


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
        self._nh_hamiltonian = None
        self._evolution_operator = None

    @property
    def jump_operators(self):
        # Add new axes so that shapes are broadcastable
        return np.sqrt(self.rates[:, np.newaxis, np.newaxis]) * self._jump_operators

    @property
    def nh_hamiltonian(self):
        if self._nh_hamiltonian is None and self.IS_subspace:
            if self.IS_subspace:
                num_IS = self.jump_operators[0].shape[0]
                out = np.zeros((num_IS, num_IS))
                for j in range(len(self.jump_operators)):
                    out = out - self.jump_operators[j].conj().T @ self.jump_operators[j]
                self._nh_hamiltonian = out
            else:
                raise NotImplementedError
        return 1j * self.rates[0] / 2 * self._nh_hamiltonian

    @property
    def liouville_evolution_operator(self):
        if self._evolution_operator is None and self.IS_subspace:
            num_IS = self._jump_operators.shape[1]
            self._evolution_operator = sparse.csr_matrix((num_IS ** 2, num_IS ** 2))
            for jump_operator in self._jump_operators:
                # Jump operator is real, so we don't need to conjugate
                self._evolution_operator = self._evolution_operator + sparse.kron(jump_operator,
                                                                                  jump_operator) - 1 / 2 * \
                                           sparse.kron(jump_operator.T @ jump_operator, sparse.identity(num_IS)) - \
                                           1 / 2 * sparse.kron(sparse.identity(num_IS), jump_operator.T @ jump_operator)

        elif self._evolution_operator is None:
            # TODO: generate the evolution operator for non-IS subspace states
            raise NotImplementedError
        return self.rates[0] * self._evolution_operator

    def liouvillian(self, state: State, apply_to=None):
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
        return State(out, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace, graph=self.graph)

    def jump_rate(self, state: State, apply_to=None):
        assert state.is_ket
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
                                           IS_subspace=state.IS_subspace, graph=state.graph))

        return np.asarray(jumped_states), np.asarray(jump_rates)

    def left_multiply(self, state: State, apply_to=None):
        # Evolve under the non-Hermitian Hamiltonian
        assert state.is_ket
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
        return State(out / 2, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace, graph=self.graph)

    def evolve(self, state: State, time):
        state_shape = state.shape
        state = np.reshape(state, (state_shape[0] ** 2, 1))
        out = sparse.linalg.expm_multiply(time * self.liouville_evolution_operator, state)
        return np.reshape(out, state_shape)

    def nh_evolve(self, state: State, time: float):
        """Non-hermitian time evolution."""
        if state.is_ket:
            return State(expm_multiply(-1j * time * self.nh_hamiltonian, state), is_ket=state.is_ket,
                         IS_subspace=state.IS_subspace, code=state.code, graph=self.graph)
        else:
            temp = expm(-1j * time * self.nh_hamiltonian)
            return State(temp @ state @ temp.conj().T, is_ket=state.is_ket, IS_subspace=state.IS_subspace,
                         code=state.code, graph=self.graph)


class SpontaneousEmission(LindbladJumpOperator):
    def __init__(self, transition: tuple = (0, 1), rates=(1,), code=qubit, IS_subspace=False,
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
        self._evolution_operator = None
        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            if code is not qubit:
                IS, nary_to_index, num_IS = graph.independent_sets_qudit(self.code)
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
                jump_operator = sparse.csc_matrix((entries, (rows, columns)), shape=(num_IS, num_IS))
                self._jump_operators.append(jump_operator)

        super().__init__(np.asarray(self._jump_operators), rates, code=code, graph=graph, IS_subspace=IS_subspace)

    @property
    def jump_operators(self):
        return np.sqrt(self.rates[0]) * self._jump_operators

    @property
    def evolution_operator(self, vector_space='hilbert'):
        if vector_space != 'hilbert' and vector_space != 'liouville':
            raise Exception('Attribute vector_space must be hilbert or liouville')
        if vector_space == 'liouville':
            if self._evolution_operator is None:
                if self.IS_subspace:
                    num_IS = self.jump_operators[0].shape[0]
                    self._evolution_operator = sparse.csc_matrix((num_IS ** 2, num_IS ** 2))
                    for jump_operator in self._jump_operators:
                        # Jump operator is real, so we don't need to conjugate
                        self._evolution_operator = self._evolution_operator + sparse.kron(jump_operator,
                                                                                          jump_operator) - \
                                                   1 / 2 * sparse.kron(jump_operator.T @ jump_operator,
                                                                       sparse.identity(num_IS)) - 1 / 2 * \
                                                   sparse.kron(sparse.identity(num_IS), jump_operator.T @ jump_operator)
                else:
                    raise NotImplementedError
            return self.rates[0] * self._evolution_operator

        else:
            # Vector space is hilbert
            if self._evolution_operator is None:
                if self.IS_subspace:
                    num_IS = self.jump_operators[0].shape[0]
                    out = np.zeros((num_IS, num_IS))
                    for j in range(len(self.jump_operators)):
                        out = out - self.jump_operators[j].conj().T @ self.jump_operators[j]
                    self._evolution_operator = out
                else:
                    raise NotImplementedError
            return 1j * self.rates[0] / 2 * self._evolution_operator

    def liouvillian(self, state: State, apply_to=None):
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
                                               self.jump_operators[0]) - 1 / 2 * \
                      self.code.left_multiply(state, [apply_to[i]], self.jump_operators[0].T @ self.jump_operators[0]) \
                      - 1 / 2 * self.code.right_multiply(state, [apply_to[i]],
                                                         self.jump_operators[0].T @ self.jump_operators[0])
        return State(out, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace, graph=state.graph)

    def jump_rate(self, state: State, apply_to=None):
        assert state.is_ket
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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


class LindbladPauliOperator(LindbladJumpOperator):
    def __init__(self, transition: tuple = (0, 1), rates=None, code=qubit, IS_subspace=False,
                 graph=None, pauli='X'):
        # jump_operators and weights are numpy arrays
        self.code = code
        self.rates = rates
        self.pauli = pauli
        self.transition = transition
        self.IS_subspace = IS_subspace
        if not self.IS_subspace:
            jump_operator = np.zeros((self.code.d, self.code.d), dtype=np.complex128)
            if pauli == 'X':
                jump_operator[self.transition[1], self.transition[0]] = 1
                jump_operator[self.transition[0], self.transition[1]] = 1
            elif pauli == 'Y':
                jump_operator[self.transition[1], self.transition[0]] = 1j
                jump_operator[self.transition[0], self.transition[1]] = -1j
            elif pauli == 'Z':
                jump_operator[self.transition[1], self.transition[1]] = -1
                jump_operator[self.transition[0], self.transition[0]] = 1
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
                IS, nary_to_index, num_IS = graph.independent_sets_qudit(self.code)
            else:
                # We have already solved for this information
                IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets
            self._jump_operators = []
            # For each atom, consider the states spontaneous emission can generate transitions between
            # Over-allocate space
            for j in range(graph.n):
                if self.pauli == 'Z':
                    entries = np.zeros(num_IS, dtype=int)
                    columns = np.arange(0, num_IS, 1, dtype=int)
                    rows = np.arange(0, num_IS, 1, dtype=int)
                    for k in IS:
                        if IS[k][2][j] == self.transition[0]:
                            entries[k] = 1
                        elif IS[k][2][j] == self.transition[1]:
                            entries[k] = -1

                elif self.pauli == 'X' or self.pauli == 'Y':
                    # For each IS, look at spin flips generated by the laser
                    # Over-allocate space
                    rows = np.zeros(graph.n * num_IS, dtype=int)
                    columns = np.zeros(graph.n * num_IS, dtype=int)
                    entries = np.zeros(graph.n * num_IS, dtype=np.complex128)
                    num_terms = 0
                    for i in IS:
                        if IS[i][2][j] == self.transition[1]:
                            # Flip spin at this location
                            # Get binary representation
                            temp = IS[i][2].copy()
                            temp[j] = self.transition[0]
                            flipped_temp = tools.nary_to_int(temp, base=code.d)
                            if flipped_temp in nary_to_index:
                                # This is a valid spin flip
                                rows[num_terms] = nary_to_index[flipped_temp]
                                columns[num_terms] = i
                                if self.pauli == 'X':
                                    entries[num_terms] = 1
                                elif self.pauli == 'Y':
                                    entries[num_terms] = -1j
                                num_terms += 1
                    # Cut off the excess in the arrays
                    columns = columns[:2 * num_terms]
                    rows = rows[:2 * num_terms]
                    entries = entries[:2 * num_terms]
                    # Populate the second half of the entries according to self.pauli
                    if self.pauli == 'X':
                        columns[num_terms:2 * num_terms] = rows[:num_terms]
                        rows[num_terms:2 * num_terms] = columns[:num_terms]
                        entries[num_terms:2 * num_terms] = entries[:num_terms]
                    elif self.pauli == 'Y':
                        columns[num_terms:2 * num_terms] = rows[:num_terms]
                        rows[num_terms:2 * num_terms] = columns[:num_terms]
                        entries[num_terms:2 * num_terms] = -1 * entries[:num_terms]
                else:
                    raise Exception('self.pauli must be X, Y, or Z')

                # Now, append the jump operator
                jump_operator = sparse.csc_matrix((entries, (rows, columns)), shape=(num_IS, num_IS))
                self._jump_operators.append(jump_operator)

        super().__init__(np.asarray(self._jump_operators), rates, code=code, graph=graph, IS_subspace=IS_subspace)

    @property
    def jump_operators(self):
        return np.sqrt(self.rates[0]) * self._jump_operators

    @property
    def evolution_operator(self, vector_space='hilbert'):
        if vector_space != 'hilbert' and vector_space != 'liouville':
            raise Exception('Attribute vector_space must be hilbert or liouville')
        if vector_space == 'liouville':
            if self._evolution_operator is None:
                if self.IS_subspace:
                    num_IS = self.jump_operators[0].shape[0]
                    self._evolution_operator = sparse.csc_matrix((num_IS ** 2, num_IS ** 2))
                    for jump_operator in self._jump_operators:
                        # Jump operator is real, so we don't need to conjugate
                        self._evolution_operator = self._evolution_operator + sparse.kron(jump_operator,
                                                                                          jump_operator) - \
                                                   1 / 2 * sparse.kron(jump_operator.T @ jump_operator,
                                                                       sparse.identity(num_IS)) - 1 / 2 * \
                                                   sparse.kron(sparse.identity(num_IS), jump_operator.T @ jump_operator)
                else:
                    raise NotImplementedError
            return self.rates[0] * self._evolution_operator

        else:
            # Vector space is hilbert
            if self._evolution_operator is None:
                if self.IS_subspace:
                    num_IS = self.jump_operators[0].shape[0]
                    out = np.zeros((num_IS, num_IS))
                    for j in range(len(self.jump_operators)):
                        out = out - self.jump_operators[j].conj().T @ self.jump_operators[j]
                    self._evolution_operator = out
                else:
                    raise NotImplementedError
            return 1j * self.rates[0] / 2 * self._evolution_operator

    def liouvillian(self, state: State, apply_to=None):
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
                                               self.jump_operators[0]) - 1 / 2 * \
                      self.code.left_multiply(state, [apply_to[i]], self.jump_operators[0].T @ self.jump_operators[0]) \
                      - 1 / 2 * self.code.right_multiply(state, [apply_to[i]],
                                                         self.jump_operators[0].T @ self.jump_operators[0])
        return State(out, is_ket=state.is_ket, code=state.code, IS_subspace=state.IS_subspace, graph=state.graph)

    def jump_rate(self, state: State, apply_to=None):
        assert state.is_ket
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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
        if apply_to is None:
            apply_to = list(range(state.number_physical_qudits))
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

