import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize
import scipy.integrate
from qsim.codes import rydberg, quantum_state
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.lindblad_master_equation import LindbladMasterEquation
from qsim.evolution import lindblad_operators, hamiltonian, quantum_channels
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim.graph_algorithms.graph import line_graph, degree_fails_graph, Graph, ring_graph
import numpy as np
from qsim.tools import tools
from qsim.codes import qubit
from qsim.codes.quantum_state import State
import scipy.sparse as sparse

"""
This script computes the scaling of the dissipation rate out of the MIS as a function of phi,
where cos(phi) = Omega_g/Omega, sin(phi)=Omega_r/Omega, for general graphs.
To compute this quantity, we first diagonalize the Hamiltonian and pick the highest
energy eigenstate (which will lead smoothly into the MIS). Then,
we compute \sum_{j != MIS} <j|L(|MIS><MIS|)|j>. We plot this quantity as a function of sin(phi).
"""


class EffectiveOperatorHamiltonian(object):
    def __init__(self, omega_g, omega_r, energies=(1,), graph: Graph = None, IS_subspace=True, code=qubit):
        # Just need to define self.hamiltonian
        assert IS_subspace
        self.energies = energies
        self.IS_subspace = IS_subspace
        self.graph = graph
        self.omega_r = omega_r
        self.omega_g = omega_g
        self.code = code
        assert self.code is qubit

        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            # We have already solved for this information
            IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets
            self.transition = (0, 1)
            self._hamiltonian_rr = np.zeros((num_IS, num_IS))
            self._hamiltonian_gg = np.zeros((num_IS, num_IS))
            self._hamiltonian_cross_terms = np.zeros((num_IS, num_IS))
            for k in IS:
                self._hamiltonian_rr[k, k] = np.sum(IS[k][2] == self.transition[0])
                self._hamiltonian_gg[k, k] = np.sum(IS[k][2] == self.transition[1])
            self._csc_hamiltonian_rr = sparse.csc_matrix(self._hamiltonian_rr)
            self._csc_hamiltonian_gg = sparse.csc_matrix(self._hamiltonian_gg)
            # For each IS, look at spin flips generated by the laser
            # Over-allocate space
            rows = np.zeros(graph.n * num_IS, dtype=int)
            columns = np.zeros(graph.n * num_IS, dtype=int)
            entries = np.zeros(graph.n * num_IS, dtype=float)
            num_terms = 0
            for i in IS:
                for j in range(len(IS[i][2])):
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
                            entries[num_terms] = 1
                            num_terms += 1
            # Cut off the excess in the arrays
            columns = columns[:2 * num_terms]
            rows = rows[:2 * num_terms]
            entries = entries[:2 * num_terms]
            # Populate the second half of the entries according to self.pauli
            columns[num_terms:2 * num_terms] = rows[:num_terms]
            rows[num_terms:2 * num_terms] = columns[:num_terms]
            entries[num_terms:2 * num_terms] = entries[:num_terms]
            # Now, construct the Hamiltonian
            self._csc_hamiltonian_cross_terms = sparse.csc_matrix((entries, (rows, columns)), shape=(num_IS, num_IS))
            try:
                self._hamiltonian_cross_terms = self._csc_hamiltonian_cross_terms.toarray()
            except MemoryError:
                self._hamiltonian_cross_terms = self._csc_hamiltonian_cross_terms

        else:
            # We are not in the IS subspace
            pass

    @property
    def hamiltonian(self):
        return self.energies[0] * (self.omega_g * self.omega_r * self._hamiltonian_cross_terms +
                                   self.omega_g ** 2 * self._csc_hamiltonian_gg +
                                   self.omega_r ** 2 * self._csc_hamiltonian_rr)

    def left_multiply(self, state: State):
        return self.hamiltonian @ state

    def right_multiply(self, state: State):
        return state @ self.hamiltonian


class EffectiveOperatorDissipation(lindblad_operators.LindbladJumpOperator):
    def __init__(self, omega_g, omega_r, rates=(1,), graph: Graph = None, IS_subspace=True, code=qubit):
        self.omega_g = omega_g
        self.omega_r = omega_r

        self.IS_subspace = IS_subspace
        self.transition = (0, 1)
        self.graph = graph
        # Construct jump operators
        if self.IS_subspace:
            # Generate sparse mixing Hamiltonian
            assert graph is not None
            assert isinstance(graph, Graph)
            if code is not qubit:
                IS, nary_to_index, num_IS = graph.independent_sets_code(self.code)
            else:
                # We have already solved for this information
                IS, nary_to_index, num_IS = graph.independent_sets, graph.binary_to_index, graph.num_independent_sets
            self._jump_operators_rg = []
            self._jump_operators_gg = []
            # For each atom, consider the states spontaneous emission can generate transitions between
            # Over-allocate space
            for j in range(graph.n):
                rows_rg = np.zeros(num_IS, dtype=int)
                columns_rg = np.zeros(num_IS, dtype=int)
                entries_rg = np.zeros(num_IS, dtype=int)
                rows_gg = np.zeros(num_IS, dtype=int)
                columns_gg = np.zeros(num_IS, dtype=int)
                entries_gg = np.zeros(num_IS, dtype=int)
                num_terms_gg = 0
                num_terms_rg = 0
                for i in IS:
                    if IS[i][2][j] == self.transition[0]:
                        # Flip spin at this location
                        # Get binary representation
                        temp = IS[i][2].copy()
                        temp[j] = self.transition[1]
                        flipped_temp = tools.nary_to_int(temp, base=code.d)
                        if flipped_temp in nary_to_index:
                            # This is a valid spin flip
                            rows_rg[num_terms_rg] = nary_to_index[flipped_temp]
                            columns_rg[num_terms_rg] = i
                            entries_rg[num_terms_rg] = 1
                            num_terms_rg += 1
                    elif IS[i][2][j] == self.transition[1]:
                        rows_gg[num_terms_gg] = i
                        columns_gg[num_terms_gg] = i
                        entries_gg[num_terms_gg] = 1
                        num_terms_gg += 1

                # Cut off the excess in the arrays
                columns_rg = columns_rg[:num_terms_rg]
                rows_rg = rows_rg[:num_terms_rg]
                entries_rg = entries_rg[:num_terms_rg]
                columns_gg = columns_gg[:num_terms_gg]
                rows_gg = rows_gg[:num_terms_gg]
                entries_gg = entries_gg[:num_terms_gg]
                # Now, append the jump operator
                jump_operator_rg = sparse.csc_matrix((entries_rg, (rows_rg, columns_rg)), shape=(num_IS, num_IS))
                jump_operator_gg = sparse.csc_matrix((entries_gg, (rows_gg, columns_gg)), shape=(num_IS, num_IS))

                self._jump_operators_rg.append(jump_operator_rg)
                self._jump_operators_gg.append(jump_operator_gg)
            self._jump_operators_rg = np.asarray(self._jump_operators_rg)
            self._jump_operators_gg = np.asarray(self._jump_operators_gg)
        else:
            #self._jump_operators_rg = []
            #self._jump_operators_gg = []
            op_rg = np.array([[[0, 0], [1, 0]]])
            op_gg = np.array([[[0, 0], [0, 1]]])
            """for i in range(self.graph.n):
                jump_operator_rg = tools.tensor_product(
                    [np.identity(2 ** i), op_rg, np.identity(2 ** (self.graph.n - i-1))])
                self._jump_operators_rg.append(jump_operator_rg)
                jump_operator_gg = tools.tensor_product(
                    [np.identity(2 ** i), op_gg, np.identity(2 ** (self.graph.n - i-1))])
                self._jump_operators_gg.append(jump_operator_gg)"""
            self._jump_operators_rg = op_rg
            self._jump_operators_gg = op_gg

        super().__init__(None, rates=rates, graph=graph, IS_subspace=IS_subspace, code=code)

    @property
    def jump_operators(self):
        return np.sqrt(self.rates[0]) * (self.omega_g * self._jump_operators_gg +
                                         self.omega_r * self._jump_operators_rg)


def jump_rate_scaling():
    graph = Graph(nx.star_graph(n=4))  # line_graph(n=3)
    energy = 1
    phis = np.arange(.001, .01, .001)
    rates = np.zeros(len(phis))
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=qubit)
    print(rydberg_hamiltonian_cost.hamiltonian)
    for d in range(len(phis)):
        print(d)
        laser = EffectiveOperatorHamiltonian(omega_g=np.cos(phis[d]), omega_r=np.sin(phis[d]), energies=(energy,),
                                             graph=graph)
        dissipation = EffectiveOperatorDissipation(omega_g=np.cos(phis[d]), omega_r=np.sin(phis[d]), rates=(energy,),
                                                   graph=graph)
        # Find the dressed states
        simulation = SchrodingerEquation(hamiltonians=[laser])
        eigval, eigvec = simulation.eig()
        eigval = [
            rydberg_hamiltonian_cost.approximation_ratio(State(eigvec[i].T, is_ket=True, graph=graph, IS_subspace=True))
            for i in range(eigval.shape[0])]
        print(eigval)
        optimal_eigval_index = np.argmax(eigval)
        optimal_eigval = eigvec[optimal_eigval_index]
        rate = 0
        for i in range(graph.n):
            # Compute the decay rate of the MIS into other states
            for j in range(eigvec.shape[0]):
                if j != optimal_eigval_index:
                    rate += np.abs(eigvec[j] @ dissipation.jump_operators[i] @ optimal_eigval.T) ** 2
        rates[d] = rate
    fit = np.polyfit(np.log(phis), np.log(rates), deg=1)

    plt.scatter(np.log(phis), np.log(rates), color='teal')
    plt.plot(np.log(phis), fit[0] * np.log(phis) + fit[1], color='k',
             label=r'$y=$' + str(np.round(fit[0], decimals=2)) + r'$x+$' + str(np.round(fit[1], decimals=2)))
    plt.legend()
    plt.xlabel(r'$\log(\phi)$')
    plt.ylabel(r'$\log(\rm{rate})$')
    plt.show()


# jump_rate_scaling()
def dephasing_performance():
    # adiabatic = SimulateAdiabatic(hamiltonian=[laser], noise = [dephasing, dissipation], noise_model='continuous',
    #                              graph=graph, IS_subspace=True, cost_hamiltonian=rydberg_hamiltonian_cost)
    # def schedule(t, tf):
    #    phi = (tf-t)/tf*np.pi/2
    #    laser.omega_g = np.cos(phi)
    #    laser.omega_r = np.sin(phi)
    #    dissipation.omega_g = np.cos(phi)
    #    dissipation.omega_r = np.sin(phi)
    # adiabatic.performance_vs_total_time(np.arange(5, 100, 5), schedule=schedule, verbose=True, plot=True, method='odeint')
    dephasing_rates = 10. ** (np.arange(-3, 0, .2))
    performance_3 = []
    performance_5 = []
    for j in [3, 5]:
        graph = line_graph(n=j)
        phi = .02
        laser = EffectiveOperatorHamiltonian(omega_g=np.cos(phi), omega_r=np.sin(phi), energies=(1,), graph=graph)
        dissipation = EffectiveOperatorDissipation(omega_g=np.cos(phi), omega_r=np.sin(phi), rates=(1,), graph=graph)
        dephasing = lindblad_operators.LindbladPauliOperator(pauli='Z', IS_subspace=True, graph=graph, rates=(.1,))
        rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=qubit)
        eq = LindbladMasterEquation(hamiltonians=[laser], jump_operators=[dissipation, dephasing])

        for i in range(len(dephasing_rates)):
            dissipation.rates = (1,)
            laser.energies = (1,)
            dephasing.rates = (dephasing_rates[i],)
            state = np.zeros(dissipation.nh_hamiltonian.shape)
            state[-1, -1] = 1
            state = State(state, is_ket=False, graph=graph, IS_subspace=True)
            ss = eq.steady_state(state)
            ss = ss[1][0]
            state = State(ss, is_ket=False, graph=graph, IS_subspace=True)
            print(rydberg_hamiltonian_cost.optimum_overlap(state))
            if j == 3:
                performance_3.append(rydberg_hamiltonian_cost.optimum_overlap(state))
            else:
                performance_5.append(rydberg_hamiltonian_cost.optimum_overlap(state))

    plt.scatter(dephasing_rates, performance_3, color='teal', label=r'$n=3$ line graph')
    plt.scatter(dephasing_rates, performance_5, color='purple', label=r'$n=5$ line graph')
    plt.ylabel(r'log(-log(optimum overlap))')
    plt.xlabel(r'$\log(\gamma\Omega^2/(\delta^2\Gamma_{\rm{dephasing}}))$')
    plt.legend()
    plt.show()


def imperfect_blockade_performance():
    graph = line_graph(n=5, return_mis=False)
    phi = .02
    laser = hamiltonian.HamiltonianDriver(pauli='X', energies=(np.cos(phi) * np.sin(phi),), graph=graph)
    energy_shift_r = hamiltonian.HamiltonianEnergyShift(index=0, energies=(np.sin(phi) ** 2,), graph=graph)
    energy_shift_g = hamiltonian.HamiltonianEnergyShift(index=1, energies=(np.cos(phi) ** 2,), graph=graph)
    dissipation = EffectiveOperatorDissipation(omega_g=np.cos(phi), omega_r=np.sin(phi), rates=(1,), graph=graph,
                                               IS_subspace=False)

    rydberg_hamiltonian = hamiltonian.HamiltonianMIS(graph, IS_subspace=False, code=qubit, energies=(0, 4,))
    rydberg_hamiltonian_cost_IS = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=qubit)
    rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=False, code=qubit, energies=(1, -4,))


    eq = LindbladMasterEquation(hamiltonians=[laser, energy_shift_g, energy_shift_r, rydberg_hamiltonian], jump_operators=[dissipation])
    state = np.zeros(rydberg_hamiltonian_cost.hamiltonian.shape)
    state[graph.independent_sets[np.argmax(rydberg_hamiltonian_cost_IS.hamiltonian)][0], graph.independent_sets[np.argmax(rydberg_hamiltonian_cost_IS.hamiltonian)][0]] = 1
    state = State(state, is_ket=False, graph=graph, IS_subspace=False)
    ss = eq.steady_state(state, k=80, use_initial_guess=True)
    print(ss[1].shape)
    ss = ss[1][0]
    print(np.diagonal(ss), rydberg_hamiltonian_cost.hamiltonian)
    state = State(ss, is_ket=False, graph=graph, IS_subspace=False)
    print(np.around(ss, decimals=3), rydberg_hamiltonian_cost.optimum_overlap(state))
    layout = np.zeros((2, 2))
    layout[0, 0] = 1
    layout[1, 1] = 1
    layout[0, 1] = 1
    adjacent_energy = 4
    diag_energy = adjacent_energy/8
    second_nearest_energy = adjacent_energy/64
    for i in range(layout.shape[0] - 1):
        for j in range(layout.shape[1] - 1):
            if layout[i, j] == 1:
                # There is a spin here
                pass


#imperfect_blockade_performance()


graph = line_graph(n=3, return_mis=False)
phi = np.pi/2
laser = EffectiveOperatorHamiltonian(omega_g=np.cos(phi), omega_r=np.sin(phi), graph=graph)
eq = SchrodingerEquation(hamiltonians=[laser])

state = State(np.ones((5, 1), dtype=np.complex128)/np.sqrt(5))
print(np.round(eq.eig(k='all')[1], decimals=3))

def performance_vs_alpha():
    # alpha = gamma * omega^2/(delta^2 T)
    graph = line_graph(n=3)
    gammas = np.array([1, 5, 10])
    times = np.array([1, 5, 10])
    deltas = np.array([20, 30])
    omegas = np.array([1, 5, 10])
    for (g, d, o) in zip(gammas, deltas, omegas):
        # Find the performance vs alpha
        laser = EffectiveOperatorHamiltonian(graph=graph)
        dissipation = EffectiveOperatorDissipation(graph=graph)
        rydberg_hamiltonian_cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True, code=qubit)
        adiabatic = SimulateAdiabatic(hamiltonian=[laser], noise = [dissipation], noise_model='continuous',
                                      graph=graph, IS_subspace=True, cost_hamiltonian=rydberg_hamiltonian_cost)
        def schedule(t, tf):
            phi = (tf-t)/tf*np.pi/2
            laser.omega_g = np.cos(phi)
            laser.omega_r = np.sin(phi)
            dissipation.omega_g = np.cos(phi)
            dissipation.omega_r = np.sin(phi)
        performance = adiabatic.performance_vs_total_time(times, schedule=schedule, verbose=True, method='odeint')[0]
        alphas = g * o**2/(d**2 * times)
        plt.scatter(alphas, performance)
    plt.show()
