import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult

from qsim.tools.tools import outer_product, is_valid_state
from qsim.codes import qubit
from qsim.evolution.hamiltonian import HamiltonianMIS, HamiltonianDriver, HamiltonianMaxCut
from qsim.graph_algorithms.graph import Graph
from qsim.codes.quantum_state import State
from qsim.schrodinger_equation import SchrodingerEquation
from qsim.lindblad_master_equation import LindbladMasterEquation


class SimulateAdiabatic(object):
    def __init__(self, graph: Graph, hamiltonian=None, noise_model=None, noise=None, code=None, cost_hamiltonian=None,
                 IS_subspace=False):
        """Noise_model is one of channel, continuous, monte_carlo, or None."""
        self.graph = graph
        self.IS_subspace = IS_subspace
        self.hamiltonian = hamiltonian
        self.noise_model = noise_model
        self.noise = noise
        self.N = self.graph.n
        # Depth of circuit
        if code is None:
            self.code = qubit
        else:
            self.code = code
        if cost_hamiltonian is None:
            cost_hamiltonian = HamiltonianMIS(graph, code=self.code)
        self.cost_hamiltonian = cost_hamiltonian

    def _num_from_time(self, time: float, method='RK45'):
        # Scale factor be 10 when time is one, should be one when time >= 10
        if method == 'trotterize':
            if self.noise_model == 'continuous' or self.noise_model is None:
                return int(time * 1000)
            elif self.noise_model == 'monte_carlo':
                return int(time * int(max(30 / time, 50)))
        else:
            if self.noise_model == 'continuous' or self.noise_model is None:
                return int(time * 10)
            elif self.noise_model == 'monte_carlo':
                return int(time * 50)

    def rydberg_MIS_schedule(self, t, tf, coefficients=None, verbose=False):
        if coefficients is None:
            coefficients = [1, 1]
        num_updates = 0
        for ham in self.hamiltonian:
            if isinstance(ham, HamiltonianDriver):
                # We don't want to update normal detunings
                if ham.pauli == 'X':
                    ham.energies = [coefficients[0] * np.sin(np.pi * t / tf) ** 2]
                    num_updates += 1
            elif self.IS_subspace and isinstance(ham, HamiltonianMIS):
                ham.energies = [coefficients[1] * (2 * t / tf - 1)]
                num_updates += 1
            elif isinstance(ham, HamiltonianMIS):
                ham.energies[0] = coefficients[1] * (2 * t / tf - 1)
                num_updates += 1
        if num_updates < len(self.hamiltonian) and verbose:
            print('Warning: not all Hamiltonian energies have been updated')
        return True

    def linear_schedule(self, t, tf, coefficients=None, verbose=False):
        if coefficients is None:
            coefficients = [1, 1]
        num_updates = 0
        for ham in self.hamiltonian:
            if isinstance(ham, HamiltonianDriver):
                if ham.pauli == 'X':
                    ham.energies = [coefficients[0] * (tf - t) / tf]
                    num_updates += 1
            elif self.IS_subspace and isinstance(ham, HamiltonianMIS):
                ham.energies = [coefficients[1] * t / tf]
                num_updates += 1
            elif isinstance(ham, HamiltonianMIS):
                ham.energies = [coefficients[1] * t / tf, coefficients[1] * t / tf]
                num_updates += 1
            elif isinstance(ham, HamiltonianMaxCut):
                ham.energies = [coefficients[1] * t / tf]
                num_updates += 1
        if num_updates < len(self.hamiltonian) and verbose:
            print('Warning: not all Hamiltonian energies have been updated')
        return True

    def run(self, time, schedule, num=None, initial_state=None, full_output=True, method='RK45', verbose=False,
            iterations=None):
        if (method == 'odeint' or method == 'trotterize') and num is None:
            num = self._num_from_time(time, method=method)
        if initial_state is None:
            # Begin with all qudits in the ground s
            initial_state = State(np.zeros((self.cost_hamiltonian.hamiltonian.shape[0], 1)), code=self.code,
                                  IS_subspace=self.IS_subspace, graph=self.graph)
            initial_state[-1, -1] = 1

        if self.noise_model is not None and self.noise_model != 'monte_carlo':
            initial_state = State(outer_product(initial_state, initial_state), IS_subspace=self.IS_subspace,
                                  code=self.code, graph=self.graph)

        if self.noise_model == 'continuous':
            # Initialize master equation
            if method == 'trotterize':
                master_equation = LindbladMasterEquation(hamiltonians=self.hamiltonian, jump_operators=self.noise)
                results, info = master_equation.run_trotterized_solver(initial_state, 0, time, num=num,
                                                                       schedule=lambda t: schedule(t, time),
                                                                       full_output=full_output, verbose=verbose)
            else:
                master_equation = LindbladMasterEquation(hamiltonians=self.hamiltonian, jump_operators=self.noise)
                results, info = master_equation.run_ode_solver(initial_state, 0, time, num=num,
                                                               schedule=lambda t: schedule(t, time), method=method,
                                                               full_output=full_output, verbose=verbose)
        elif self.noise_model is None:
            # Noise model is None
            # Initialize Schrodinger equation
            schrodinger_equation = SchrodingerEquation(hamiltonians=self.hamiltonian)
            if method == 'trotterize':
                results, info = schrodinger_equation.run_trotterized_solver(initial_state, 0, time, num=num,
                                                                            verbose=verbose, full_output=full_output,
                                                                            schedule=lambda t: schedule(t, time))
            else:
                results, info = schrodinger_equation.run_ode_solver(initial_state, 0, time, num=num, verbose=verbose,
                                                                    schedule=lambda t: schedule(t, time), method=method,
                                                                    full_output=full_output)

        else:
            assert self.noise_model == 'monte_carlo'
            # Initialize master equation
            master_equation = LindbladMasterEquation(hamiltonians=self.hamiltonian, jump_operators=self.noise)
            results, info = master_equation.run_stochastic_wavefunction_solver(initial_state, 0, time, num=num,
                                                                               full_output=full_output,
                                                                               schedule=lambda t: schedule(t, time),
                                                                               method=method, verbose=verbose,
                                                                               iterations=iterations)

        if len(results.shape) == 2:
            # The algorithm has output a single state
            out = [State(results, IS_subspace=self.IS_subspace, code=self.code, graph=self.graph)]
        elif len(results.shape) == 3:
            # The algorithm has output an array of states
            out = [State(res, IS_subspace=self.IS_subspace, code=self.code, graph=self.graph) for res in results]
        else:
            assert len(results.shape) == 4
            if self.noise_model != 'monte_carlo':
                raise Exception('Run output has more dimensions than expected')
            out = []
            for i in range(results.shape[0]):
                # For all iterations
                res = []
                for j in range(results.shape[1]):
                    # For all times
                    res.append(
                        State(results[i, j, ...], IS_subspace=self.IS_subspace, code=self.code, graph=self.graph))
                out.append(res)
        return out, info

    def performance_vs_time(self, time, schedule, num=None, metric='approximation_ratio', initial_state=None,
                            plot=False, verbose=False, method='RK45', iterations=None):
        # Convert metric and method to lists
        if isinstance(metric, str):
            metric = [metric]
        if isinstance(method, str):
            method = [method]
        metric_label = None
        min_performance = np.inf
        average_performance = []
        total_performance = {m: {} for m in method}
        all_info = {m: {} for m in method}
        colors = ['teal', 'purple', 'm', 'deepskyblue', 'deeppink', 'salmon', 'orange', 'r']
        scatter_label = None
        n = 0
        for l in range(len(method)):
            results, info = self.run(time, schedule, num=num, initial_state=initial_state, full_output=True,
                                     method=method[l], verbose=verbose, iterations=iterations)
            for m in range(len(metric)):
                if metric[m] != 'approximation_ratio' and metric[m] != 'optimum_overlap' and \
                        metric[m] != 'cost_function':
                    raise NotImplementedError('Metric must be approximation_ratio, cost_function or optimum_overlap.')
                else:
                    if metric[m] == 'cost_function':
                        metric_function = self.cost_hamiltonian.cost_function
                        metric_label = 'cost function'
                    elif metric[m] == 'approximation_ratio':
                        metric_function = self.cost_hamiltonian.approximation_ratio
                        metric_label = 'approximation ratio'
                    else:
                        # Metric function must be optimum overlap
                        metric_function = self.cost_hamiltonian.optimum_overlap
                        metric_label = 'optimum overlap'
                if isinstance(info, OdeResult):
                    times = info.t
                elif isinstance(info, dict):
                    times = info['t']
                else:
                    raise Exception('Times are not defined')
                # Input should have three dimensions if the noise model is not monte carlo
                # Otherwise, we should have four dimensions due to multiple iterations
                if self.noise_model == 'monte_carlo':
                    if iterations is None:
                        iterations = 1
                    performance = np.zeros((iterations, len(results[0])))
                    for (i, trial) in zip(range(iterations), results):
                        performance[i, ...] = np.array([metric_function(trial[j]) for j in range(len(trial))])
                    total_performance[method[l]][metric[m]] = performance
                    performance = np.mean(performance, axis=0)

                else:
                    performance = [metric_function(results[i]) for i in range(len(results))]
                    total_performance[method[l]][metric[m]] = performance
                average_performance.append(performance)
                info['results'] = results
                all_info[method[l]][metric[m]] = info
                if min(performance) < min_performance:
                    min_performance = min(performance)
                if verbose:
                    print('Final performance: ', str(performance[-1]))
                if plot:
                    if 'cost_function' not in metric:
                        plt.hlines(1, min(times) - 1, max(times) + 1, linestyles=':', colors='k')
                        plt.ylim(min_performance - .03, 1.03)
                    else:
                        plt.hlines(self.cost_hamiltonian.optimum, min(times) - 1, max(times) + 1, linestyles=':',
                                   colors='k')
                        plt.ylim(min_performance - .05 * self.cost_hamiltonian.optimum,
                                 self.cost_hamiltonian.optimum + .05 *
                                 self.cost_hamiltonian.optimum)
                    if len(method) > 1:
                        if len(metric) > 1:
                            scatter_label = metric_label + ', ' + method[l]
                        else:
                            scatter_label = method[l]
                    else:
                        if len(metric) > 1:
                            scatter_label = metric_label
                    plt.scatter(times, average_performance[n], color=colors[n], label=scatter_label)
                    plt.plot(times, average_performance[n], color=colors[n])
                    n += 1
        if plot:
            plt.xlim(-1, time + 1)
            plt.xlabel(r'annealing time $t$')
            if len(metric) == 1:
                plt.ylabel(metric_label)
            else:
                plt.ylabel('performance')
            if scatter_label is not None:
                plt.legend(loc='lower right')
            plt.show()

        return total_performance, all_info

    def performance_vs_total_time(self, time, schedule, num=None, metric='approximation_ratio', initial_state=None,
                                  plot=False, verbose=False, method='RK45', iterations=None, errorbar=False):
        # Convert metric and method to lists
        if isinstance(metric, str):
            metric = [metric]
        if isinstance(method, str):
            method = [method]
        metric_label = None
        min_performance = np.inf
        total_performance = {m: {} for m in method}
        all_info = {m: {} for m in method}
        if isinstance(time, int):
            all_performance = np.zeros((len(method), len(metric), len(np.linspace(0, 1, num=int(num)))))
        else:
            all_performance = np.zeros((len(method), len(metric), len(time)))
        if errorbar:
            stdev = np.zeros((len(method), len(metric), len(time)))
        colors = ['teal', 'purple', 'm', 'deepskyblue', 'deeppink', 'salmon', 'orange', 'r']
        scatter_label = None
        for l in range(len(method)):
            for t in range(len(time)):
                if verbose:
                    print('Solving time: ' + str(time[t]))
                results, info = self.run(time[t], schedule, num=num, initial_state=initial_state, full_output=False,
                                         method=method[l], iterations=iterations, verbose=verbose)
                # print(np.around(results[-1], decimals=3), is_valid_state(results[-1], is_ket=False))
                for m in range(len(metric)):
                    if t == 0:
                        total_performance[method[l]][metric[m]] = []
                    if metric[m] != 'approximation_ratio' and metric[m] != 'optimum_overlap' and metric[m] != \
                            'cost_function':
                        print('Metric must be approximation_ratio, cost_function or optimum_overlap. Metric ' +
                              metric[m] + ' will not be computed')
                    else:
                        if metric[m] == 'cost_function':
                            metric_function = self.cost_hamiltonian.cost_function
                        elif metric[m] == 'approximation_ratio':
                            metric_function = self.cost_hamiltonian.approximation_ratio
                        else:
                            metric_function = self.cost_hamiltonian.optimum_overlap

                        if self.noise_model == 'monte_carlo':
                            if errorbar:
                                print('Error bar not yet implemented for Monte Carlo trials. Error bar will default to '
                                      'zero.')
                            if iterations is None:
                                iterations = 1
                            performance_time = np.zeros(iterations)
                            for (p, trial) in zip(range(iterations), results):
                                performance_time[p, ...] = metric_function(trial)
                            total_performance[method[l]][metric[m]].append(performance_time)
                            all_performance[l, m, t] = np.mean(performance_time, axis=0)

                        else:
                            if errorbar:
                                if results[-1].is_ket:
                                    probabilities = (np.abs(results[-1]) ** 2).flatten().real
                                else:
                                    probabilities = np.diag(results[-1]).real
                                res = np.zeros(int(np.max(self.cost_hamiltonian.hamiltonian.real)) -
                                               int(np.min(self.cost_hamiltonian.hamiltonian.real)) + 1)
                                for i in range(results[-1].shape[0]):
                                    # For now, assume that the cost hamiltonian is diagonal
                                    # NOTE: non-integer valued hamiltonians will be binned as the integer floor of the float value
                                    if metric == 'cost_function':
                                        res[int(self.cost_hamiltonian.hamiltonian[i, i])] = res[int(
                                            self.cost_hamiltonian.hamiltonian[i, i])] + probabilities[i] * int(
                                            self.cost_hamiltonian.hamiltonian[i, i])
                                    else:
                                        res[int(self.cost_hamiltonian.hamiltonian[i, i])] = res[int(
                                            self.cost_hamiltonian.hamiltonian[i, i])] + probabilities[i]
                                vals = np.arange(len(res)) / (len(res) - 1)
                                x0 = np.sum(res * vals)
                                stdev[l, m, t] = np.sum(res * (vals - x0) ** 2) ** .5
                            total_performance[method[l]][metric[m]].append(metric_function(results[-1]))
                            all_performance[l, m, t] = metric_function(results[-1])
                        if all_performance[l, m, t] < min_performance:
                            min_performance = all_performance[l, m, t]

                        info['results'] = results
                        if t == 0:
                            all_info[method[l]][metric[m]] = {key: [] for key in info}
                        for key in info:
                            all_info[method[l]][metric[m]][key].append(info[key])

                    if verbose:
                        np.set_printoptions(threshold=np.inf)
                        print('Performance: ', all_performance[l, m, t])
        if 'cost_function' not in metric and plot:
            plt.hlines(1, min(time) - 1, max(time) + 1, linestyles=':', colors='k')
            print(min_performance)
            plt.ylim(min_performance - .03, 1.03)
        elif plot:
            plt.hlines(self.cost_hamiltonian.optimum, min(time) - 1, max(time) + 1, linestyles=':', colors='k')
            plt.ylim(min_performance - .05 * self.cost_hamiltonian.optimum, self.cost_hamiltonian.optimum + .05 *
                     self.cost_hamiltonian.optimum)
        if plot:
            n = 0
            for l in range(len(method)):
                for m in range(len(metric)):
                    if metric[m] != 'approximation_ratio' and metric[m] != 'optimum_overlap' and metric[m] != \
                            'cost_function':
                        raise NotImplementedError(
                            'Metric must be approximation_ratio, cost_function or optimum_overlap.')
                    else:
                        metric_label = metric[m].replace('_', ' ')
                        if len(method) > 1:
                            if len(metric) > 1:
                                scatter_label = metric_label + ', ' + method[l]
                            else:
                                scatter_label = method[l]
                        else:
                            if len(metric) > 1:
                                scatter_label = metric_label
                        if errorbar:
                            plt.errorbar(time, all_performance[l, m, :], yerr=stdev, color=colors[n])
                        plt.scatter(time, all_performance[l, m, :], color=colors[n], label=scatter_label)
                        plt.plot(time, all_performance[l, m, :], color=colors[n])
                        n += 1

            plt.xlim(min(time) - 1, max(time) + 1)
            plt.xlabel(r'total annealing time $T$')
            if len(metric) == 1:
                plt.ylabel(metric_label)
            else:
                plt.ylabel('performance')
            if scatter_label is not None:
                plt.legend(loc='lower right')
            plt.show()
        return total_performance, all_info

    def spectrum_vs_time(self, time, schedule, k=2, num=None, plot=False, which='S', hamiltonian=True, verbose=False):
        """Solves for the small (S) or large (L) energy sector."""
        if num is None:
            num = self._num_from_time(time)
        times = np.linspace(1e-2, time, num=num)
        if not hamiltonian and (self.noise_model is None or self.noise_model is 'monte_carlo'):
            print('No noise models found. Finding Hamiltonian spectrum')
            hamiltonian = True
        if self.noise_model is None or self.noise_model is 'monte_carlo' or hamiltonian:
            # Initialize Schrodinger equation
            schrodinger_equation = SchrodingerEquation(hamiltonians=self.hamiltonian)
            eigvals = np.zeros((len(times), k), dtype=np.float64)
            for i in range(len(times)):
                if verbose:
                    print(times[i])
                schedule(times[i], time)
                eigval, eigvec = schrodinger_equation.eig(which=which, k=k)
                if which == 'S':
                    eigvals[i] = eigval[0:k]
                elif which == 'L':
                    eigvals[i] = eigval[len(eigval) - k - 1:-1]
            if plot:
                plotted_eigvals = np.swapaxes(eigvals, 0, 1)
                for i in range(k):
                    plt.scatter(times, plotted_eigvals[i], color='teal')
                plt.ylabel('Energy')
                plt.xlabel('Time')
                plt.show()
        else:
            eigvals = np.zeros((len(times), self.cost_hamiltonian.shape[0], self.cost_hamiltonian.shape[0]),
                               dtype=np.complex128)
            raise NotImplementedError

        return eigvals

    def distribution_vs_total_time(self, time, schedule, num=None, metric='approximation_ratio', initial_state=None,
                                   plot=False, verbose=False, method='RK45', iterations=None):
        if metric != 'approximation_ratio' and metric != 'optimum_overlap' and metric != 'cost_function':
            raise NotImplementedError('Metric must be approximation_ratio, cost_function or optimum_overlap.')
        performance = np.zeros((len(time), int(np.max(self.cost_hamiltonian.hamiltonian.real) + 1)))
        j = 0
        for t in time:
            if verbose:
                print('Solving time: ' + str(t))
            results, info = self.run(t, schedule, num=num, initial_state=initial_state, full_output=False,
                                     method=method, verbose=verbose, iterations=iterations)
            if not self.noise_model == 'monte_carlo':
                res = np.zeros(int(np.max(self.cost_hamiltonian.hamiltonian.real)) -
                               int(np.min(self.cost_hamiltonian.hamiltonian.real)) + 1)
                if results[-1].is_ket:
                    probabilities = (np.abs(results[-1]) ** 2).flatten().real
                else:
                    probabilities = np.diag(results[-1]).real
                for i in range(results[-1].shape[0]):
                    # For now, assume that the cost hamiltonian is diagonal
                    # TODO: make this work for non-integer valued Hamiltonians
                    res[int(self.cost_hamiltonian.hamiltonian[i, i])] = res[int(
                        self.cost_hamiltonian.hamiltonian[i, i])] + probabilities[i]
            else:
                if iterations is None:
                    iterations = 1
                res = np.zeros(int(np.max(self.cost_hamiltonian.hamiltonian.real) - np.min(
                    self.cost_hamiltonian.hamiltonian.real) + 1))
                for k in range(iterations):
                    if results[k].is_ket:
                        probabilities = (np.abs(results[k]) ** 2).flatten().real
                    else:
                        probabilities = np.diag(results[k]).real
                    for i in range(len(probabilities)):
                        if self.cost_hamiltonian._is_diagonal:
                            res[int(self.cost_hamiltonian.hamiltonian[i, i])] = res[int(
                                self.cost_hamiltonian.hamiltonian[i, i])] + probabilities[i]
                res = res / iterations
            if verbose:
                print('Distribution', res)
            performance[j, ...] = res

            j += 1
        if verbose:
            print('Performance', performance.T)
        if plot:
            plt.imshow(performance.T, vmin=0, vmax=1, interpolation=None, extent=[min(time), max(time), 0, 1],
                       origin='lower', aspect='auto')
            plt.colorbar()

            plt.xlabel(r'Total annealing time $T$')
            plt.ylabel(r'Approximation ratio')
            plt.show()
        return performance

    def groundstate_ordering_vs_time(self, time, schedule, which='S', verbose=False):
        """In a typical adiabatic algorithm, the ground state will not have a level crossing. However, in graphs
        with degenerate ground states, it is possible that energy levels that adiabatically lead into the ground
        state have level crossings. Given a list of times, returns the eigenstates that will adiabatically lead
        into the ground state and their eigenenergies at each time."""
        # Start at later times
        self.graph.degeneracy = 10
        eigvals = np.zeros((len(time), self.graph.degeneracy))
        eigvecs = np.zeros((len(time), self.graph.degeneracy, self.cost_hamiltonian.hamiltonian.shape[0]),
                           dtype=np.complex128)
        indices = np.arange(self.graph.degeneracy)
        full_indices = np.zeros((len(time), self.graph.degeneracy))
        schrodinger_equation = SchrodingerEquation(hamiltonians=self.hamiltonian)
        threshold = .01

        def normalize_phase(eig):
            """Take in an array of eigenvectors, and normalize the phase such that the first element is positive
            and real"""
            where_nonzero = np.argwhere(np.absolute(eig) > 1e-9)[0, 0]
            eig = np.e ** (-1j * np.angle(eig[where_nonzero])) * eig
            # We can take the eigenvalues to be real for this Hamiltonian
            return eig / np.linalg.norm(eig)

        i = 0
        for t in np.flip(time):
            schedule(t, 1)
            if i == 0:
                eigval, eigvec = schrodinger_equation.eig(which=which, k=np.max(indices) + 1)
                # print(eigval, eigvec)
                for j in range(eigvec.shape[0]):
                    eigvec[j] = normalize_phase(eigvec[j])
                old_eigvec = eigvec
                eigvals[i, :] = eigval
                eigvecs[i, :] = eigvec
            else:
                eigval, eigvec = schrodinger_equation.eig(which=which, k=np.max(indices) + 1)
                # Now, check that the eigenvectors are macroscopically similar
                # First, you need to normalize the phase on the eigenvectors
                for j in range(eigvec.shape[0]):
                    eigvec[j] = normalize_phase(eigvec[j])
                # Now compute deviations in the dot product
                deviation = []
                for e in range(len(eigval)):
                    deviation.append(np.abs(np.dot(eigvec[e], old_eigvec[e])))
                deviation = 1 - np.array(deviation)
                where_different = np.argwhere(deviation > threshold).flatten()
                original_different = where_different.copy()
                if len(where_different) > 0:
                    if verbose:
                        print(where_different, indices, t)
                for j in range(len(where_different)):
                    # We need to check in nearby locations
                    # For now, assume that we have only jumped one location
                    if j != len(where_different) - 1 and where_different[j] != 0:
                        adjacent = True
                        # Check everything after it
                        check = where_different[j + 1:]
                        last = where_different[j]
                        for c in check:
                            if c != last + 1:
                                adjacent = False
                            last = c
                            dev = 1 - np.abs(np.dot(eigvec[where_different[j]], old_eigvec[c]))
                            if deviation[where_different[j]] - dev > .1 or dev < threshold:
                                if verbose:
                                    print('swapping', where_different[j], c, dev, deviation[where_different[j]])
                                # We now know that c should replace where_different[j]
                                temp = where_different.copy()
                                temp[j] = c
                                for k in np.argwhere(where_different == c):
                                    temp[k] = where_different[j]
                                where_different = temp.copy()
                                if verbose:
                                    print(indices, where_different)

                                break
                            else:
                                if verbose:
                                    print('decided against swapping', dev, deviation[where_different[j]],
                                          where_different[j], c)
                                    if adjacent and c == np.max(indices):
                                        print('all eigenstates are adjacent, consider looking at more spectrum',
                                              where_different[j], c)

                    else:
                        # Check if the last index is good or not, now that previous indices have been resolved
                        dev_new = 1 - np.abs(np.dot(old_eigvec[np.max(where_different)], eigvec[where_different[j]]))
                        if dev_new > threshold:
                            if verbose:
                                print('problems, need to look at more spectrum', where_different[j], dev_new,
                                      np.max(where_different))
                            if where_different[j] != 0:
                                for k in np.argwhere(indices == where_different[j]):
                                    if where_different[j] == np.max(
                                            indices):  # np.allclose(np.argwhere(indices == where_different[j]), np.array([[len(indices)-1]])):
                                        for l in [1, 2, 3]:
                                            new_eigval, new_eigvec = schrodinger_equation.eig(which=which,
                                                                                              k=np.max(indices) + 1 + l)
                                            dev = 1 - np.abs(np.dot(new_eigvec[where_different[j] + l],
                                                                    old_eigvec[where_different[j]]))
                                            if verbose:
                                                print('should we swap higher?', where_different[j], l, dev,
                                                      deviation[where_different[j]])
                                            if deviation[where_different[j]] - dev > .1:
                                                indices[k] = where_different[j] + l
                                                eigvec = new_eigvec
                                                eigval = new_eigval
                                                break
                        else:
                            if verbose:
                                print('happy times', dev_new)
                temp = indices.copy()
                for j in range(len(where_different)):
                    for k in np.argwhere(where_different[j] == indices):
                        # Swap that shit
                        temp[k] = original_different[j]
                indices = temp
                old_eigvec = eigvec
                # If they are not, solve for one more eigenvector
                # If the eigenvector is still not macroscopically similar to any eigenvector, lower the threshold for
                # comparison
                # Now populate eigvecs
            full_indices[i, :] = indices
            eigvecs[i, :] = eigvec[indices, :]
            eigvals[i, :] = eigval[indices]
            i += 1
        return np.flip(eigvecs, axis=0), np.flip(full_indices, axis=0), np.flip(eigvals, axis=0)

    def eigenstate_ordering_vs_time(self, time, schedule, verbose=False, return_eigvecs = False, k='all', which='S'):
        """In a typical adiabatic algorithm, the ground state will not have a level crossing. However, in graphs
        with degenerate ground states, it is possible that energy levels that adiabatically lead into the ground
        state have level crossings. Given a list of times, returns the eigenstates that will adiabatically lead
        into the ground state and their eigenenergies at each time."""
        # Start at earlier times
        if k == 'all':
            shape = self.hamiltonian[0].hamiltonian.shape[0]
            eigvals = np.zeros((len(time), shape))
            if return_eigvecs:
                eigvecs = np.zeros((len(time), shape, shape), dtype=np.complex128)
            indices = np.arange(shape)
            full_indices = np.zeros((len(time), shape))
            schrodinger_equation = SchrodingerEquation(hamiltonians=self.hamiltonian)
            threshold = .01

            def normalize_phase(eig):
                """Take in an array of eigenvectors, and normalize the phase such that the first element is positive
                and real"""
                where_nonzero = np.argwhere(np.absolute(eig) > 1e-9)[0, 0]
                eig = np.e ** (-1j * np.angle(eig[where_nonzero])) * eig
                # We can take the eigenvalues to be real for this Hamiltonian
                return eig / np.linalg.norm(eig)

            i = 0
            for t in time:
                print(t)
                schedule(t, 1)
                if i == 0:
                    eigval, eigvec = schrodinger_equation.eig(k=k)
                    eigvals[i, :] = eigval
                    for j in range(shape):
                        eigvec[j] = normalize_phase(eigvec[j])
                    old_eigvec = eigvec
                    if return_eigvecs:
                        eigvecs.append(eigvec)
                else:
                    eigval, eigvec = schrodinger_equation.eig(k='all')
                    eigvals[i, :] = eigval
                    # Now, check that the eigenvectors are macroscopically similar
                    # First, you need to normalize the phase on the eigenvectors
                    for j in range(shape):
                        eigvec[j] = normalize_phase(eigvec[j])
                    if return_eigvecs:
                        eigvecs[i, :] = eigvec
                    # Now compute deviations in the dot product
                    deviation = np.zeros(shape)
                    for e in range(len(eigval)):
                        deviation[e] = 1-np.abs(np.dot(eigvec[e], old_eigvec[e]))
                    where_different = np.argwhere(deviation > threshold).flatten()
                    original_different = where_different.copy()
                    if len(where_different) > 0:
                        if verbose:
                            print(where_different, indices, t)
                    for j in range(0, len(where_different)):
                        # We need to check in nearby locations
                        # For now, assume that we have only jumped one location
                        if j != len(where_different) - 1 and where_different[j] != 0:
                            # Don't worry about changes to the ground state
                            adjacent = True
                            # Check everything after it
                            # Only checking a few eigenvalues
                            check = where_different[j + 1:j+6]
                            last = where_different[j]
                            for c in check:
                                if c != last + 1:
                                    adjacent = False
                                last = c
                                dev = 1 - np.abs(np.dot(eigvec[where_different[j]], old_eigvec[c]))
                                if deviation[where_different[j]] - dev > .1 or dev < threshold:
                                    if verbose:
                                        print('swapping', t, where_different[j], c, dev, deviation[where_different[j]])
                                    # We now know that c should replace where_different[j]
                                    temp = where_different.copy()
                                    temp[j] = c
                                    # Now that the replacement is made, replace c with where_different[j]
                                    for k in np.argwhere(where_different == c):
                                        temp[k] = where_different[j]
                                    # Replace where_different
                                    # Try and do a handshake? Not implemented yet...
                                    where_different = temp
                                    if verbose:
                                        print(indices, where_different)

                                    break
                                else:
                                    if verbose:
                                        print('decided against swapping', dev, deviation[where_different[j]],
                                              where_different[j], c)
                                        if adjacent and c == np.max(indices):
                                            print('all eigenstates are adjacent and the end of the spectrum was reached',
                                                  where_different[j], c)
                    temp = indices.copy()
                    for j in range(len(where_different)):
                        for k in np.argwhere(where_different[j] == indices):
                            # Swap that shit
                            temp[k] = original_different[j]
                    indices = temp
                    old_eigvec = eigvec
                    # If they are not, solve for one more eigenvector
                    # If the eigenvector is still not macroscopically similar to any eigenvector, lower the threshold for
                    # comparison
                    # Now populate eigvecs, if desired
                full_indices[i, :] = indices
                if return_eigvecs:
                    eigvecs[i, :] = eigvec
                i += 1
            if return_eigvecs:
                return eigvecs, full_indices, eigvals
            else:
                return full_indices, eigvals

        else:
            # Start at later times
            eigvals = np.zeros((len(time), k))
            eigvecs = np.zeros((len(time), k, self.cost_hamiltonian.hamiltonian.shape[0]),
                               dtype=np.complex128)
            indices = np.arange(k)
            full_indices = np.zeros((len(time), k))
            schrodinger_equation = SchrodingerEquation(hamiltonians=self.hamiltonian)
            threshold = .01

            def normalize_phase(eig):
                """Take in an array of eigenvectors, and normalize the phase such that the first element is positive
                and real"""
                where_nonzero = np.argwhere(np.absolute(eig) > 1e-9)[0, 0]
                eig = np.e ** (-1j * np.angle(eig[where_nonzero])) * eig
                # We can take the eigenvalues to be real for this Hamiltonian
                return eig / np.linalg.norm(eig)

            i = 0
            for t in np.flip(time):
                schedule(t, 1)
                if i == 0:
                    eigval, eigvec = schrodinger_equation.eig(which=which, k=np.max(indices) + 1)
                    # print(eigval, eigvec)
                    for j in range(eigvec.shape[0]):
                        eigvec[j] = normalize_phase(eigvec[j])
                    old_eigvec = eigvec
                    eigvals[i, :] = eigval
                    eigvecs[i, :] = eigvec
                else:
                    eigval, eigvec = schrodinger_equation.eig(which=which, k=np.max(indices) + 1)
                    # Now, check that the eigenvectors are macroscopically similar
                    # First, you need to normalize the phase on the eigenvectors
                    for j in range(eigvec.shape[0]):
                        eigvec[j] = normalize_phase(eigvec[j])
                    # Now compute deviations in the dot product
                    deviation = []
                    for e in range(len(eigval)):
                        deviation.append(np.abs(np.dot(eigvec[e], old_eigvec[e])))
                    deviation = 1 - np.array(deviation)
                    where_different = np.argwhere(deviation > threshold).flatten()
                    original_different = where_different.copy()
                    if len(where_different) > 0:
                        if verbose:
                            print(where_different, indices, t)
                    for j in range(len(where_different)):
                        # We need to check in nearby locations
                        # For now, assume that we have only jumped one location
                        if j != len(where_different) - 1 and where_different[j] != 0:
                            adjacent = True
                            # Check everything after it
                            check = where_different[j + 1:]
                            last = where_different[j]
                            for c in check:
                                if c != last + 1:
                                    adjacent = False
                                last = c
                                dev = 1 - np.abs(np.dot(eigvec[where_different[j]], old_eigvec[c]))
                                if deviation[where_different[j]] - dev > .1 or dev < threshold:
                                    if verbose:
                                        print('swapping', where_different[j], c, dev, deviation[where_different[j]])
                                    # We now know that c should replace where_different[j]
                                    temp = where_different.copy()
                                    temp[j] = c
                                    for k in np.argwhere(where_different == c):
                                        temp[k] = where_different[j]
                                    where_different = temp.copy()
                                    if verbose:
                                        print(indices, where_different)

                                    break
                                else:
                                    if verbose:
                                        print('decided against swapping', dev, deviation[where_different[j]],
                                              where_different[j], c)
                                        if adjacent and c == np.max(indices):
                                            print('all eigenstates are adjacent, consider looking at more spectrum',
                                                  where_different[j], c)

                        else:
                            # Check if the last index is good or not, now that previous indices have been resolved
                            dev_new = 1 - np.abs(
                                np.dot(old_eigvec[np.max(where_different)], eigvec[where_different[j]]))
                            if dev_new > threshold:
                                if verbose:
                                    print('problems, need to look at more spectrum', where_different[j], dev_new,
                                          np.max(where_different))
                                if where_different[j] != 0:
                                    for k in np.argwhere(indices == where_different[j]):
                                        if where_different[j] == np.max(
                                                indices):  # np.allclose(np.argwhere(indices == where_different[j]), np.array([[len(indices)-1]])):
                                            for l in [1, 2, 3]:
                                                new_eigval, new_eigvec = schrodinger_equation.eig(which=which,
                                                                                                  k=np.max(
                                                                                                      indices) + 1 + l)
                                                dev = 1 - np.abs(np.dot(new_eigvec[where_different[j] + l],
                                                                        old_eigvec[where_different[j]]))
                                                if verbose:
                                                    print('should we swap higher?', where_different[j], l, dev,
                                                          deviation[where_different[j]])
                                                if deviation[where_different[j]] - dev > .1:
                                                    indices[k] = where_different[j] + l
                                                    eigvec = new_eigvec
                                                    break
                            else:
                                if verbose:
                                    print('happy times', dev_new)
                    temp = indices.copy()
                    for j in range(len(where_different)):
                        for k in np.argwhere(where_different[j] == indices):
                            # Swap that shit
                            temp[k] = original_different[j]
                    indices = temp
                    old_eigvec = eigvec
                    # If they are not, solve for one more eigenvector
                    # If the eigenvector is still not macroscopically similar to any eigenvector, lower the threshold for
                    # comparison
                    # Now populate eigvecs
                full_indices[i, :] = indices
                eigvecs[i, :] = eigvec[indices, :]
                eigvals[i, :] = eigval
                i += 1
            return np.flip(eigvecs, axis=0), np.flip(full_indices, axis=0)

