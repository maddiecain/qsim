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
                return time * 100
            elif self.noise_model == 'monte_carlo':
                return time * int(max(30 / time, 50))
        else:
            if self.noise_model == 'continuous' or self.noise_model is None:
                return time * 10
            elif self.noise_model == 'monte_carlo':
                return time * 50

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
        if method == 'odeint' or method == 'trotterize' and num is None:
            num = self._num_from_time(time, method=method)

        if initial_state is None:
            # Begin with all qudits in the ground s
            initial_state = State(np.zeros((self.cost_hamiltonian.hamiltonian.shape[0], 1)), code=self.code,
                                  IS_subspace=self.IS_subspace)
            initial_state[-1, -1] = 1

        if self.noise_model is not None and self.noise_model != 'monte_carlo':
            initial_state = State(outer_product(initial_state, initial_state), IS_subspace=self.IS_subspace,
                                  code=self.code)

        if self.noise_model == 'continuous':
            # Initialize master equation
            if method == 'trotterize':
                raise NotImplementedError
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

        elif self.noise_model == 'monte_carlo':
            if method == 'trotterize':
                raise NotImplementedError
            # Initialize master equation
            master_equation = LindbladMasterEquation(hamiltonians=self.hamiltonian, jump_operators=self.noise)
            results, info = master_equation.run_stochastic_wavefunction_solver(initial_state, 0, time, num=num,
                                                                               full_output=full_output,
                                                                               schedule=lambda t: schedule(t, time),
                                                                               method=method, verbose=verbose,
                                                                               iterations=iterations)

        if len(results.shape) == 2:
            # The algorithm has outputted a single state
            out = [State(results, IS_subspace=self.IS_subspace, code=self.code)]
        elif len(results.shape) == 3:
            # The algorithm has outputted an array of states
            out = [State(res, IS_subspace=self.IS_subspace, code=self.code) for res in results]
        elif len(results.shape) == 4:
            if self.noise_model != 'monte_carlo':
                raise Exception('Run output has more dimensions than expected')
            out = []
            for i in range(results.shape[0]):
                # For all iterations
                res = []
                for j in range(results.shape[1]):
                    # For all times
                    res.append(State(results[i, j, ...], IS_subspace=self.IS_subspace, code=self.code))
                out.append(res)
        return out, info

    def performance_vs_time(self, time, schedule, num=None, metric='approximation_ratio', initial_state=None,
                            plot=False, verbose=False, method='RK45', iterations=None):
        if metric != 'approximation_ratio' and metric != 'optimum_overlap' and metric != 'cost_function':
            raise NotImplementedError('Metric must be approximation_ratio, cost_function or optimum_overlap.')
        else:
            if metric == 'cost_function':
                metric_function = self.cost_hamiltonian.cost_function
                label = 'cost function'
            if metric == 'approximation_ratio':
                metric_function = self.cost_hamiltonian.approximation_ratio
                label = 'approximation ratio'
            if metric == 'optimum_overlap':
                metric_function = self.cost_hamiltonian.optimum_overlap
                label = 'optimum overlap'
        results, info = self.run(time, schedule, num=num, initial_state=initial_state, full_output=True,
                                 method=method, verbose=verbose, iterations=iterations)
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
            performance = np.mean(performance, axis=0)

        else:
            performance = [metric_function(results[i]) for i in range(len(results))]
        if verbose:
            print('Final performance: ', str(performance[-1]))
        if plot:
            if metric == 'approximation_ratio' or metric == 'optimum_overlap':
                plt.hlines(1, min(times) - 1, max(times) + 1, linestyles=':', colors='k')
                plt.ylim(min(performance) - .03, 1.03)
            else:
                plt.hlines(self.cost_hamiltonian.optimum, min(times) - 1, max(times) + 1, linestyles=':', colors='k')
                plt.ylim(min(performance) - .05 * self.cost_hamiltonian.optimum, self.cost_hamiltonian.optimum + .05 *
                         self.cost_hamiltonian.optimum)
            plt.xlim(min(times) - 1, max(times) + 1)
            plt.scatter(times, performance, color='teal')
            plt.plot(times, performance, color='teal')
            plt.hlines(1, 0, max(times), linestyles=':', colors='k')
            plt.xlabel(r'annealing time $t$')
            plt.ylabel(label)
            plt.show()
        return performance, info

    def performance_vs_total_time(self, time, schedule, num=None, metric='approximation_ratio', initial_state=None,
                                  plot=False, verbose=False, method='RK45', iterations=None, errorbar=False):
        if metric != 'approximation_ratio' and metric != 'optimum_overlap' and metric != 'cost_function':
            raise NotImplementedError('Metric must be approximation_ratio, cost_function or optimum_overlap.')
        else:
            if metric == 'cost_function':
                metric_function = self.cost_hamiltonian.cost_function
                label = 'cost function'
            if metric == 'approximation_ratio':
                metric_function = self.cost_hamiltonian.approximation_ratio
                label = 'approximation ratio'
            if metric == 'optimum_overlap':
                metric_function = self.cost_hamiltonian.optimum_overlap
                label = 'optimum overlap'
        performance = []
        stdev = []
        times = np.asarray(time)
        for t in times:
            if verbose:
                print('Solving time: ' + str(t))
            results, info = self.run(t, schedule, num=num, initial_state=initial_state, full_output=False,
                                     method=method, iterations=iterations, verbose=verbose)
            if self.noise_model == 'monte_carlo':
                if errorbar:
                    raise NotImplementedError
                if iterations is None:
                    iterations = 1
                performance_time = np.zeros(iterations)
                for (i, trial) in zip(range(iterations), results):
                    performance_time[i, ...] = metric_function(trial)
                performance.append(np.mean(performance_time, axis=0))

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
                    stdev.append(np.sum(res * (vals - x0) ** 2) ** .5)
                performance.append(metric_function(results[-1]))
            if verbose:
                np.set_printoptions(threshold=np.inf)
                print('Performance: ', performance[-1])
        if plot:
            if errorbar:
                plt.errorbar(time, performance, yerr=stdev, color='teal')
            plt.scatter(times, performance, color='teal')
            plt.plot(times, performance, color='teal')
            if metric == 'approximation_ratio' or metric == 'optimum_overlap':
                plt.hlines(1, min(times) - 1, max(times) + 1, linestyles=':', colors='k')
                plt.ylim(min(performance) - .03, 1.03)
            else:
                plt.hlines(self.cost_hamiltonian.optimum, min(times) - 1, max(times) + 1, linestyles=':', colors='k')
                plt.ylim(min(performance) - .05 * self.cost_hamiltonian.optimum, self.cost_hamiltonian.optimum + .05 *
                         self.cost_hamiltonian.optimum)
            plt.xlim(min(times) - 1, max(times) + 1)
            plt.xlabel(r'total annealing time $T$')
            plt.ylabel(label)
            plt.show()
        return performance

    def spectrum_vs_time(self, time, schedule, k=2, num=None, plot=False, which='S'):
        """Solves for the low (S) or high (L) energy sector."""
        if num is None:
            num = self._num_from_time(time)
        times = np.linspace(0, time, num=num)
        if self.noise_model is None or self.noise_model is 'monte_carlo':
            eigvals = np.zeros((len(times), self.cost_hamiltonian.shape[0]), dtype=np.complex128)
        else:
            eigvals = np.zeros((len(times), self.cost_hamiltonian.shape[0], self.cost_hamiltonian.shape[0]),
                               dtype=np.complex128)

        # Generate a LinearOperator from the Hamiltonian

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
