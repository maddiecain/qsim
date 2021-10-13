import matplotlib.pyplot as plt
import numpy as np
from qsim.evolution import hamiltonian
from qsim.graph_algorithms.graph import unit_disk_grid_graph, rydberg_graph
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic
from qsim import schrodinger_equation
import matplotlib.gridspec as gridspec
import scipy.sparse
import scipy.optimize
import pandas as pd
import sys
import scipy.interpolate


def loadfile(graph_index, size):
    graph_mask = np.reshape(np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 3:],
                            (size, size), order='F')[::-1, ::-1].T.astype(bool)
    MIS_size = np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 0].astype(int)
    degeneracy = np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 1].astype(int)
    graph_data = {'graph_index': graph_index, 'MIS_size': MIS_size, 'degeneracy': degeneracy,
                  'side_length': size,
                  'graph_mask': graph_mask, 'number_of_nodes': int(np.sum(graph_mask))}
    return graph_data


def find_ratio(tails_graph, graph, tf, graph_index=None, size=None):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    # print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    # print('Starting rydberg')
    if tails_graph is not None:
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    t_pulse_max = np.max(pulse[:, 0]) - 2 * 0.312

    def schedule(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)

    def schedule_old(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (-2 * np.pi * 11 * 2 * (1 / 2 - t / T),)  # (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)  # (2 * np.pi * 2 * amplitude,)

    def schedule_exp_optimized(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            t_pulse = (t - 0.312) / (T - 2 * 0.312) * t_pulse_max + 0.312
            driver.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], pulse[:, 1] / 2),)
            cost.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], -pulse[:, 2]),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)
        # print(t, cost.energies)

    def schedule_exp_linear(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (2 * np.pi * (-(11 + 15) / (T - 2 * .312) * (t - .312) + 15),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)

    # print(t, cost.energies)
    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    if tails_graph is None:
        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver], cost_hamiltonian=cost,
                               IS_subspace=True)
    else:
        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                               IS_subspace=True)
    # print('Starting evolution')
    ars = []
    probs = []
    for i in range(len(tf)):
        states, data = ad.run(tf[i], schedule_exp_linear, num=int(20 * tf[i]), method='odeint', full_output=False)
        cost.energies = (1,)
        ar = cost.approximation_ratio(states[-1])
        prob = cost.optimum_overlap(states[-1])
        # np.savez_compressed('{}x{}_{}_{}.npz'.format(size, size, graph_index, i), state=states[-1])
        print(tf[i], ar, prob)
        ars.append(ar)
        probs.append(prob)
    return ars, probs


def find_gap(graph, tails_graph, k=2, verbose=False):
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[4]

    print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    print('Starting cost')
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    if tails_graph is not None:
        print('Starting rydberg')
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    t_pulse_max = np.max(pulse[:, 0]) - 2 * 0.312
    max_detuning = np.max(pulse[:, 2])

    def schedule_old(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (-2 * np.pi * 11 * 2 * (1 / 2 - t / T),)  # (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)  # (2 * np.pi * 2 * amplitude,)

    def schedule_exp_optimized(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            t_pulse = (t - 0.312) / (T - 2 * 0.312) * t_pulse_max + 0.312
            driver.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], pulse[:, 1] / 2),)
            cost.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], -pulse[:, 2]),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * max_detuning,)
        # print(t, cost.energies)

    def schedule_exp_linear(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (2 * np.pi * (-(11 + 15) / (T - 2 * .312) * (t - .312) + 15),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)

    def gap(t):
        t = t[0]
        t = t * t_max
        schedule(t, t_max)
        eigval, eigvec = eq.eig(which='S', k=k)
        if verbose:
            print(np.abs(eigval[1] - eigval[0]), t / t_max)
        # print(t/t_max, np.abs(eigval[1] - eigval[0]))
        return np.abs(eigval[1] - eigval[0])

    if tails_graph is None:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])
    else:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])

    # Do minimization
    schedule = schedule_exp_linear
    if tails_graph is None:
        # Previously .67
        upper = 0.8
        # Previously .55
        lower = .55
        init = np.array([.7])
    else:
        # Default: .77
        upper = 0.77
        lower = .6
        # Default: .71
        init = np.array([.71])
    print('Starting gap')
    res_linear = scipy.optimize.minimize(gap, init, bounds=[(lower, upper)], method='L-BFGS-B')
    terminate = False
    while res_linear.x[0] == upper and not terminate:
        upper -= .01
        if upper <= lower:
            upper = .95
        res_linear = scipy.optimize.minimize(gap, init, bounds=[(lower, upper)], tol=1e-2)

    return res_linear.fun, res_linear.x[0]


def collect_gap_statistics(size):
    size_indices = np.array([5, 6, 7, 8, 9, 10])
    size_index = np.argwhere(size == size_indices)[0, 0]
    gaps = []
    ratios = []
    degeneracies = []
    locs = []
    graph_indices = []
    for index in range(38):
        xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
        # graph_index = graphs[index]
        graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])
        graph_indices.append(graph_index)
        print(graph_index)
        graph_data = loadfile(graph_index, size)
        grid = graph_data['graph_mask']
        # We want to have the file of hardest graphs, then import them
        graph = unit_disk_grid_graph(grid, periodic=False, generate_mixer=True)
        # tails_graph = rydberg_graph(grid, visualize=False)
        # print(np.arange(graph.num_independent_sets, 0, -1))
        # print(np.sum(2 ** np.arange(graph.n, 0, -1) * (1 - graph.independent_sets), axis=1))
        # print(1-graph.independent_sets)
        print('Degeneracy', graph.degeneracy)
        print('MIS size', graph.mis_size)
        print('Hilbert space size', graph.num_independent_sets)
        # tails_graph = rydberg_graph(grid, visualize=False)
        """gap_linear, loc_linear = find_gap(graph, tails_graph, k=2)
        # 0.8106922236509717 9.954314144613932
        # 0.8106958195062691 9.954314428767589
        print(gap_linear, loc_linear)
        gaps.append(gap_linear)
        locs.append(loc_linear)
        degeneracies.append(graph.degeneracy)
        is_sizes = np.sum(1 - graph.independent_sets, axis=1)
        ratio = np.sum(is_sizes == graph.mis_size - 1) / np.sum(is_sizes == graph.mis_size)
        ratios.append(ratio)
        print(gaps, degeneracies, ratios, locs)"""
        # print(index, graph_index, gap, loc, ratio, graph.degeneracy)
        # visualize_low_energy_subspace(graph)


def visualize_low_energy_subspace(graph, tails_graph, k=5):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    # print('Starting driver')
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[4]
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    # print('Starting rydberg')
    rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    t_pulse_max = np.max(pulse[:, 0]) - 2 * 0.312

    def schedule(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)

    def schedule_old(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (-2 * np.pi * 11 * 2 * (1 / 2 - t / T),)  # (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)  # (2 * np.pi * 2 * amplitude,)

    def schedule_exp_optimized(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            t_pulse = (t - 0.312) / (T - 2 * 0.312) * t_pulse_max + 0.312
            driver.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], pulse[:, 1] / 2),)
            cost.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], -pulse[:, 2]),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)
        # print(t, cost.energies)

    def schedule_exp_linear(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (2 * np.pi * (-(11 + 15) / (T - 2 * .312) * (t - .312) + 15),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)

    # print(t, cost.energies)
    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)

    def gap(t):
        schedule_exp_linear(t * t_max, t_max)
        eigval, eigvec = eq.eig(which='S', k=k)
        return np.abs(eigval - eigval[0]), eigvec

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])

    fig = plt.figure(tight_layout=True)
    k_cutoff = 5
    gs = gridspec.GridSpec(k_cutoff, k_cutoff)
    ax = fig.add_subplot(gs[:, 0:k_cutoff - 1])
    num = 50
    print('beginning computation')
    for (i, t) in enumerate(np.linspace(.5, .98, num)):
        print(i)
        g, eigvec = gap(t)
        print(g)
        if i == num - 1:
            probs = np.abs(eigvec) ** 2
            for l in range(k_cutoff):
                layout = grid.copy().flatten().astype(float)
                layout[layout == 0] = -5
                layout[layout == 1] = 0
                layout_temp = layout.copy()
                ax_im = fig.add_subplot(gs[k_cutoff - l - 1, -1])
                for j in range(probs.shape[1]):
                    layout_temp[layout == 0] = layout_temp[layout == 0] + (1 - graph.independent_sets[j]) * probs[l, j]
                ax_im.imshow(layout_temp.reshape(grid.shape))
                ax_im.set_axis_off()
        ax.scatter(np.ones(len(g)) * t, g, s=5, color='navy')
        ax.set_xlabel(r'$t/T$')
        ax.set_ylabel(r'Eigenenergy ($\Omega_{\max} = 1$)')
    plt.show()


def track_eigenstate_populations(graph, tails_graph, grid, k=2):
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[6]
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    # print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    # print('Starting rydberg')
    if tails_graph is not None:
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    t_pulse_max = np.max(pulse[:, 0]) - 2 * 0.312
    max_detuning = np.max(pulse[:, 2])

    def schedule_old(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (-2 * np.pi * 11 * 2 * (1 / 2 - t / T),)  # (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)  # (2 * np.pi * 2 * amplitude,)

    def schedule_exp_optimized(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            t_pulse = (t - 0.312) / (T - 2 * 0.312) * t_pulse_max + 0.312
            driver.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], pulse[:, 1] / 2),)
            cost.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], -pulse[:, 2]),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * max_detuning,)
        # print(t, cost.energies)

    def schedule_exp_linear(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (2 * np.pi * (-(11 + 15) / (T - 2 * .312) * (t - .312) + 15),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)

    def eigs(t):
        schedule_exp_linear(t, t_max)
        eigval, eigvec = eq.eig(which='S', k=k)
        return eigval - eigval[0], eigvec

    if tails_graph is not None:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])

        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                               IS_subspace=True)
    else:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])

        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver], cost_hamiltonian=cost,
                               IS_subspace=True)
    print('starting evolution')

    states, data = ad.run(t_max, schedule_exp_linear, num=int(60 * t_max), method='odeint', full_output=True)
    print('finished evolution')
    times = data['t']
    start_index = len(times) // 2
    times = times[start_index:]
    states = states[start_index:]
    populations = np.zeros((len(times), k))
    energies = np.zeros((len(times), k))
    for (i, time) in enumerate(times):
        print(i)
        eigval, eigvecs = eigs(time)
        populations[i] = (np.abs(eigvecs @ states[i]) ** 2).flatten()
        energies[i] = eigval / (2 * np.pi)
    populations = np.log10(populations)
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots()
    norm = plt.Normalize(-5, np.max(populations))
    for i in range(energies.shape[1]):
        points = np.array([times, energies[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        # Set the values used for colormapping
        lc.set_array(populations[:-1, i])
        lc.set_linewidth(1)
        line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.ax.set_ylabel(r'$\log_{10}(\rm{population})$')
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel(r'Eigenenergy (MHz)')
    ax.set_ylim(np.min(energies) - .3, np.max(energies))
    # ax.annotate(r'$r_{0\rightarrow j} = \sum_\mu |\langle j |c_\mu |0\rangle |^2$', xy=(0.4, 0.1), xycoords='data')
    plt.show()

    fig, ax = plt.subplots()
    deltas = np.zeros(len(times))
    print(times)
    for (i, time) in enumerate(times):
        schedule_exp_linear(time, t_max)
        deltas[i] = cost.energies[0] / (2 * np.pi)
    print(deltas)
    norm = plt.Normalize(-5, np.max(populations))
    for i in range(energies.shape[1]):
        points = np.array([deltas, energies[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        # Set the values used for colormapping
        lc.set_array(populations[:-1, i])
        lc.set_linewidth(1)
        line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.ax.set_ylabel(r'$\log_{10}(\rm{population})$')
    ax.set_xlim(np.max(deltas), np.min(deltas))
    ax.set_xlabel(r'$\Delta$ (MHz)')
    ax.set_ylabel(r'Eigenenergy (MHz)')
    ax.set_ylim(np.min(energies) - .3, np.max(energies))
    # ax.annotate(r'$r_{0\rightarrow j} = \sum_\mu |\langle j |c_\mu |0\rangle |^2$', xy=(0.4, 0.1), xycoords='data')
    plt.show()

    fig, ax = plt.subplots(3, 5)

    probs = np.abs(eigvecs) ** 2
    print(probs[:, :10])
    for l in range(k):
        i = l // 5
        j = l % 5
        layout = grid.copy().flatten().astype(float)
        layout[layout == 0] = -5
        layout[layout == 1] = 0
        layout_temp = layout.copy()
        for m in range(probs.shape[1]):
            layout_temp[layout == 0] = layout_temp[layout == 0] + (1 - graph.independent_sets[m]) * probs[l, m]
        ax[i][j].imshow(layout_temp.reshape(grid.shape))
        ax[i][j].set_axis_off()
        ax[i][j].text(-0.1, 1.05, '$\lambda${}'.format(str(l)), transform=ax[i][j].transAxes, size=10,
                      weight='bold')

    plt.show()


def track_eigenstate_composition(graph, tails_graph, num=1):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    # print('Starting driver')
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[4]
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    # print('Starting rydberg')
    if tails_graph is not None:
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    t_pulse_max = np.max(pulse[:, 0]) - 2 * 0.312

    def schedule(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)

    def schedule_old(t, T):
        # Linear ramp on the detuning, experiment-like ramp on the driver
        k = 50
        a = .95
        b = 3.1
        x = t / T
        amplitude = (
                            -1 / (1 + np.e ** (k * (x - a))) ** b - 1 / (1 + np.e ** (-k * (x - (1 - a)))) ** b + 1) / \
                    (-1 / ((1 + np.e ** (k * (1 / 2 - a))) ** b) - 1 / (
                            (1 + np.e ** (-k * (1 / 2 - (1 - a)))) ** b) + 1)
        cost.energies = (-2 * np.pi * 11 * 2 * (1 / 2 - t / T),)  # (2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2 * np.pi * 2 * amplitude,)  # (2 * np.pi * 2 * amplitude,)

    def schedule_exp_optimized(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            t_pulse = (t - 0.312) / (T - 2 * 0.312) * t_pulse_max + 0.312
            driver.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], pulse[:, 1] / 2),)
            cost.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], -pulse[:, 2]),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)
        # print(t, cost.energies)

    def schedule_exp_linear(t, T):
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 15,)
        elif .312 <= t <= T - .312:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (2 * np.pi * (-(11 + 15) / (T - 2 * .312) * (t - .312) + 15),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11,)

    # print(t, cost.energies)
    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)

    def eigs(t):
        schedule_exp_linear(t * t_max, t_max)
        if num == 0:
            eigval, eigvec = eq.eig(which='S', k=num+2)
        else:
            eigval, eigvec = eq.eig(which='S', k=num+1)

        return eigval, eigvec

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    if tails_graph is not None:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])
    else:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])
    #gap, loc = find_gap(graph, tails_graph)
    loc = 0.6403615396636326
    fig, ax = plt.subplots(1, num+2, sharex=True)

    print('Beginning computation')
    print(graph.mis_size)
    colors = ['blue', 'green', 'navy', 'orange', 'firebrick', 'purple', 'magenta', 'cornflowerblue', 'teal',
              'grey', 'cyan', 'limegreen', 'red', 'yellow', 'pink', 'orangered', 'salmon', 'violet']
    for (i, t) in enumerate(np.linspace(.6, .7, 50)):
        print(i)
        eigval, eigvec = eigs(t)
        ax[0].scatter(t, np.abs(eigval[0] - eigval[1]), color='k')
        for n in range(num + 1):
            if i == 0:
                ax[n+1].vlines(loc, 0, 1)
            vec = np.abs(eigvec[n])**2
            for j in range(graph.mis_size+1):
                population = np.sum(vec*(np.isclose(np.sum(1-graph.independent_sets, axis=1),j)))
                ax[n+1].scatter(t, population, color=colors[j])
    ax[0].set_xlabel(r'$t/T$')
    ax[1].set_xlabel(r'$t/T$')
    ax[1].set_ylabel(r'Gap')

    ax[0].set_ylabel(r'Population')
    for j in range(graph.mis_size):
        ax[0].scatter([],[],color=colors[j], label='IS size '+str(j))
    ax[0].legend()
    plt.show()


import networkx as nx


def count_maximal(graph_mask):
    graph = unit_disk_grid_graph(graph_mask)
    graph.generate_independent_sets()
    num_misminusone = len(np.argwhere(np.sum(1 - graph.independent_sets, axis=1) == graph.mis_size - 1))
    cliques = nx.algorithms.clique.find_cliques(nx.complement(graph.graph))
    num_maximal = 0
    for clique in cliques:
        size = len(clique)
        if size == graph.mis_size - 1:
            num_maximal += 1
    return num_misminusone, num_maximal, graph.degeneracy


gaps = []
locs = []
dts = []


def compare_annealing_schedules():
    fig, ax = plt.subplots(1, 2, sharex=True)
    n_points = 10
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)  # + .312 * 2
    probs_leo = np.array([0.03675858493628861])
    ratios_leo = np.array([0.850095867569476])
    times_leo = np.array([0.9187876675539848])
    ratios_maddie = np.array(
        [0.5872188740719826, 0.7500019839093216, 0.8552189842114166, 0.8945485343722768, 0.9019282778122899,
         0.9032200354212159, 0.9232604226144995, 0.9569637485679381, 0.9814859766664283, 0.9946943214195105])
    probs_maddie = np.array(
        [0.0032217228656843747, 0.03141104266513283, 0.07761120166475678, 0.13078151106804267, 0.12753771899729957,
         0.13052831370572185, 0.3100872816663417, 0.6129682495505786, 0.8334997647424613, 0.9522821051618593])
    ratios_exp_optimized = np.array(
        [0.6937452542855389, 0.8344103414753614, 0.8740343699638329, 0.8893686428993904, 0.8933369054638978,
         0.9011908746466736, 0.900462132059283, 0.9022041786902256, 0.9076768865454466, 0.917035664138586])
    probs_exp_optimized = np.array(
        [0.01124530482548392, 0.07446012129727977, 0.038744840556763166, 0.01735259369051035, 0.04120745580157461,
         0.1113976399718933, 0.10475899797263598, 0.12041107451352634, 0.16957758934842748, 0.2537594285021999])
    ratios_linear = np.array(
        [0.7863048641959948, 0.8572997328432775, 0.8898843660663831, 0.8987687767180945, 0.8985515483112597,
         0.8966572379599336, 0.8969606613030238, 0.8996494824460223, 0.9053560691624151, 0.915277681881363])
    probs_linear = np.array(
        [0.014854200678154648, 0.046936133414466195, 0.0849223729213073, 0.10164454954856618, 0.08938155189086139,
         0.0719630554138605, 0.07471134232765048, 0.09905850831328825, 0.1503826351818135, 0.23941144900869274])
    ax[0].scatter(times_exp, probs_maddie, label='Gap optimized', color='navy')
    ax[0].scatter(times_exp, probs_exp_optimized, label='Exp optimized', color='green')
    ax[0].scatter(times_exp, probs_linear, label='Linear', color='magenta')
    ax[1].scatter(times_exp, 1 - ratios_linear, label='Linear', color='magenta')
    ax[1].scatter(times_exp, 1 - ratios_maddie, color='navy')
    ax[0].scatter(times_leo, probs_leo, label='QAOA', color='orange')
    ax[0].plot(times_leo, probs_leo, color='orange')
    ax[1].scatter(times_leo, 1 - ratios_leo, color='orange')
    ax[1].plot(times_leo, 1 - ratios_leo, color='orange')
    ax[1].scatter(times_exp, 1 - ratios_exp_optimized, color='green')
    ax[0].plot(times_exp, probs_maddie, color='navy')
    ax[0].plot(times_exp, probs_exp_optimized, color='green')
    ax[1].plot(times_exp, 1 - ratios_maddie, color='navy')
    ax[1].plot(times_exp, 1 - ratios_exp_optimized, color='green')
    ax[0].plot(times_exp, probs_linear, color='magenta')
    ax[1].plot(times_exp, 1 - ratios_linear, color='magenta')
    ax[0].semilogx()
    ax[1].loglog()
    ax[0].set_xlabel('Total depth')
    ax[1].set_xlabel('Total depth')
    ax[0].set_ylabel('MIS probability')
    ax[1].set_ylabel('1-ratio')
    #ax[0].vlines(times_exp[6], 0, 1, color='k')
    #ax[1].vlines(times_exp[6], .5, .001, color='k')

    ax[0].legend()
    plt.show()

    fig, ax = plt.subplots(1, 2, sharex=True)
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)  # + .312 * 2
    probs_leo = np.array([0.03675858493628861])
    ratios_leo = np.array([0.850095867569476])
    times_leo = np.array([0.9187876675539848])
    ratios_maddie = np.array(
        [0.5768753658636498, 0.734892607858768, 0.8292658993839205, 0.8752504837727871, 0.8985046412634529, 0.9415628867929099, 0.9872245925873886])
    probs_maddie = np.array([0.001493831375768246, 0.015372714816018335, 0.039537540070439164, 0.06963481199935859, 0.14829108241357375, 0.4877380294039414, 0.8860993804250801])
    ratios_exp_optimized = np.array(
        [0.6756244607652274, 0.8090112318078814, 0.8586980893101519, 0.8848027026078742, 0.8998649948844744, 0.9004286593001408, 0.9086885929769424])
    probs_exp_optimized = np.array([0.005733294923230214, 0.029890723309834534, 0.01240750190786585, 0.044010978311958986, 0.11898186047675204, 0.1082699342339216, 0.1794732530650944])
    #ratios_linear = np.array(
    #    [0.7863048641959948, 0.8572997328432775, 0.8898843660663831, 0.8987687767180945, 0.8985515483112597,
    #     0.8966572379599336, 0.8969606613030238, 0.8996494824460223, 0.9053560691624151, 0.915277681881363])
    #probs_linear = np.array(
    #    [0.014854200678154648, 0.046936133414466195, 0.0849223729213073, 0.10164454954856618, 0.08938155189086139,
    #     0.0719630554138605, 0.07471134232765048, 0.09905850831328825, 0.1503826351818135, 0.23941144900869274])
    ax[0].scatter(times_exp, probs_maddie, label='Gap optimized', color='navy')
    ax[0].scatter(times_exp, probs_exp_optimized, label='Exp optimized', color='green')
    ax[1].scatter(times_exp, 1 - ratios_maddie, color='navy')
    ax[1].scatter(times_exp, 1 - ratios_exp_optimized, color='green')
    ax[0].plot(times_exp, probs_maddie, color='navy')
    ax[0].plot(times_exp, probs_exp_optimized, color='green')
    ax[1].plot(times_exp, 1 - ratios_maddie, color='navy')
    ax[1].plot(times_exp, 1 - ratios_exp_optimized, color='green')
    ax[0].semilogx()
    ax[1].loglog()
    ax[0].set_xlabel('Total depth')
    ax[1].set_xlabel('Total depth')
    ax[0].set_ylabel('MIS probability')
    ax[1].set_ylabel('1-ratio')
    # ax[0].vlines(times_exp[6], 0, 1, color='k')
    # ax[1].vlines(times_exp[6], .5, .001, color='k')

    ax[0].legend()
    plt.show()


#compare_annealing_schedules()

def critical_detuning_comparison():
    detunings_1 = np.concatenate([np.linspace(2, 8, 5), [5.203572566650173]])
    detunings_2 = np.concatenate([np.linspace(2, 8, 5), [7.280586067579559]])
    detunings_3 = np.concatenate([np.linspace(2, 8, 5), [5.898394603439648]])
    detunings_4 = np.concatenate([np.linspace(2, 8, 5), [7.383516127596532]])

    probs_1 = [[0.007186639711008398, 0.02679582285186702, 0.076544873813938, 0.18103178396756262, 0.4024836923536264, 0.647321126412186, 0.8567857427066692],
    [0.014400545650379432, 0.09240705181480163, 0.2136999932066448, 0.3674850290148115, 0.6153853878791313, 0.8407102409096786, 0.9624685045847856],
    [0.014180310709374422, 0.1212586059128046, 0.4158799881461088, 0.7453474822233768, 0.888835353962686, 0.9687450484419446, 0.9900368944376203],
    [0.005368004304025041, 0.054457110612650336, 0.2289963822861938, 0.4844169021623724, 0.6863400568881689, 0.8246858931584148, 0.944538369396123],
    [0.0009845588193858486, 0.012268444555575764, 0.06200125274890916, 0.1564709851628499, 0.31912664850951633, 0.5712629534736646, 0.7988169589649566],
    [0.013125836650401292, 0.11557251017885631, 0.416333832200632, 0.7663863236072153, 0.9081586803715801, 0.9745261522581807, 0.9886627440743851]]

    probs_2 = [[0.016703627511959036, 0.02100743493625355, 0.056619707928162843, 0.08625498233635148, 0.07823872704189007, 0.07251922042241975, 0.07445860715829188, 0.09361585483638846, 0.1357021131565131],
    [0.03321424585974909, 0.05982692688160364, 0.02031536366266278, 0.08663890040674732, 0.07339949063791613, 0.07216209744288724, 0.08810491167693478, 0.1241803762051952, 0.1882659200564603],
    [0.033048935660489866, 0.11180576300140717, 0.05408839489822237, 0.022444364907024485, 0.06781579809607528, 0.11357132755513522, 0.1196977550268624, 0.18453862827963013, 0.2825168401276489],
    [0.01591819876096013, 0.08472534938230433, 0.12479729066685044, 0.07466464981629742, 0.04077000632513535, 0.08637118836544955, 0.23337119091450503, 0.2937793745494568, 0.4358443519236693],
    [0.00447651765646492, 0.035802536108976386, 0.1039598501593759, 0.14229838678932788, 0.1379946878541734, 0.10434304265832645, 0.1366748821458483, 0.21353423907920976, 0.34325456917970826],
    [0.008633425706479124, 0.056179692772724395, 0.12326465748646787, 0.1293477415342329, 0.09123628191507568, 0.08848242183268093, 0.20724570376049312, 0.4323448000814015, 0.6506062631260177]]

    probs_3 = [[0.023689739386979235, 0.05096823966069981, 0.14039379445154945, 0.26015988552274116, 0.40582559966205556, 0.5404562713443539, 0.7003835025716258],
               [0.04174122354092795, 0.13252188260698541, 0.24076352176453902, 0.3545878578519954, 0.533844680904757,
                0.6793597771283164, 0.8339022293889466],
               [0.0328289304664268, 0.17481543318422982, 0.3348569182670805, 0.4872534108701274, 0.7309074427750968, 0.8678202932074139, 0.9522541583697066],
               [0.010964375910114716, 0.08664270169238139, 0.24046419112169007, 0.43599625593513225, 0.6863770117936299, 0.8718348216018633, 0.9335222395152349],
               [0.002082055914798844, 0.02335984333190648, 0.10377596456195982, 0.23520604005842527, 0.3823597491730406,
                0.5359063074193697, 0.7056350796877595],
               [0.018647958311212118, 0.12731389600402818, 0.3011265491799269, 0.4986558399836797, 0.7847235156818881, 0.9783909628974844, 0.9879456726674103]]

    probs_4 = [[0.0104140641059919, 0.01765812759807482, 0.05094595499696197, 0.0867777074398304, 0.08888536575175307, 0.10788421343186795, 0.1632897441744811, 0.26375924357548475],
               [0.020665240985470637, 0.04033811755636386, 0.03612818613390104, 0.10126900348451606, 0.1026257730109531, 0.15193788629159516, 0.22840160137600515, 0.3643838904203988],
               [0.019480842646699036, 0.07686444630680857, 0.04329398475809401, 0.0547429951260552, 0.23520461676957916, 0.1907736917849114, 0.34821570324327855, 0.506165660245684],
               [0.00851938533059389, 0.055258473884824105, 0.09158713297675963, 0.05166823505845166, 0.16069132788068147, 0.46389667742832486, 0.4870500400944564, 0.7075470837510854],
               [0.002135602306799587, 0.020010083691023008, 0.07268266277737306, 0.11494953157673611, 0.14551923858296023, 0.23805881941558185, 0.46910430109972073, 0.6802752479823319],
               [0.003954168051022381, 0.0317233643915049, 0.08917931771656168, 0.0997937218020668, 0.1310837129627334, 0.3585570245641468, 0.7305452740226175, 0.9050612642747103]]
    n_points = 7
    times_1 = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    n_points = 9
    times_2 = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    n_points = 7
    times_3 = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    n_points = 8
    times_4 = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    colors = ['blue', 'green', 'navy', 'orange', 'firebrick', 'purple', 'magenta', 'cornflowerblue', 'teal',
              'grey', 'cyan', 'limegreen', 'red', 'yellow', 'pink', 'orangered', 'salmon', 'violet']
    fig, ax = plt.subplots(1, 5)

    for i in range(len(probs_1)):
        if i < len(probs_1) -1:
            ax[1].scatter(times_1, probs_1[i], color=colors[i], label='Flatten at $\delta={}$'.format(np.round(detunings_1[i], 2)))
            ax[1].plot(times_1, probs_1[i], color=colors[i])
        else:
            ax[1].scatter(times_1, probs_1[i], color=colors[i],
                          label='Critical detuning $\delta={}$'.format(np.round(detunings_1[i], 2)))
            ax[1].plot(times_1, probs_1[i], color=colors[i], linestyle='dashed')

    for i in range(len(probs_2)):
        if i < len(probs_2) -1:
            ax[2].scatter(times_2, probs_2[i], color=colors[i])
            ax[2].plot(times_2, probs_2[i], color=colors[i])
        else:
            ax[2].scatter(times_2, probs_2[i], color=colors[i],
                          label='Critical detuning $\delta={}$'.format(np.round(detunings_2[i], 2)))
            ax[2].plot(times_2, probs_2[i], color=colors[i], linestyle='dashed')
    for i in range(len(probs_3)):
        if i < len(probs_3) -1:
            ax[3].scatter(times_3, probs_3[i], color=colors[i])
            ax[3].plot(times_3, probs_3[i], color=colors[i])
        else:
            ax[3].scatter(times_3, probs_3[i], color=colors[i],
                          label='Critical detuning $\delta={}$'.format(np.round(detunings_3[i], 2)))
            ax[3].plot(times_3, probs_3[i], color=colors[i], linestyle='dashed')

    for i in range(len(probs_4)):
        if i < len(probs_4) -1:
            ax[4].scatter(times_4, probs_4[i], color=colors[i])
            ax[4].plot(times_4, probs_4[i], color=colors[i])
        else:
            ax[4].scatter(times_4, probs_4[i], color=colors[i],
                          label='Critical detuning $\delta={}$'.format(np.round(detunings_4[i], 2)))
            ax[4].plot(times_4, probs_4[i], color=colors[i], linestyle='dashed')

    def schedule_cubic(t, T, plateau):
        cubic_ys = 2 * np.pi * np.array([11.5, plateau+.5, plateau, plateau-.5 , -11.5])
        cubic_xs = np.array([.312, (T / 2 - .312) / 1.35 + .312, T / 2, T - .312-(T / 2 - .312) / 1.35, T - .312])
        if t < .312:
            driver = (2 * np.pi * 2 * t / .312,)
            cost = (2 * np.pi * 11.5,)
        elif .312 <= t <= T - .312:
            driver = (2 * np.pi * 2,)
            #cost = (scipy.interpolate.CubicSpline(cubic_xs, cubic_ys,)(t),)
            cost = (scipy.interpolate.interp1d(cubic_xs, cubic_ys, kind='cubic')(t),)
        else:
            driver = (2 * np.pi * 2 * (T - t) / .312,)
            cost = (-2 * np.pi * 11.5,)
        return driver[0], cost[0]

    tmax = 3
    times_pulse = np.linspace(0, tmax, 100)
    for (p, plateau) in enumerate(-np.linspace(2, 10, 10)):
        for time in times_pulse:
            driver, cost = schedule_cubic(time, tmax, plateau)
            ax[0].scatter(time, -cost/ (2 * np.pi), color=colors[p])
            ax[0].scatter(time, driver / (2 * np.pi), color='pink')


    ax[0].scatter([], [], label='Rabi frequency', color='pink')
    ax[0].scatter([], [], label='Gap optimized', color='navy')
    ax[0].set_ylabel('Energy (MHz)')
    ax[0].set_xlabel('Time ($\mu$s)')
    ax[1].set_xlabel('Time ($\mu$s)')
    ax[2].set_xlabel('Time ($\mu$s)')
    ax[3].set_xlabel('Time ($\mu$s)')
    ax[4].set_xlabel('Time ($\mu$s)')


    ax[1].set_ylabel('MIS probability')
    ax[2].set_ylabel('MIS probability')
    ax[3].set_ylabel('MIS probability')
    ax[4].set_ylabel('MIS probability')

    ax[2].title.set_text('Graph 807, gap $={}$'.format(np.round(1.0058786679591663/(2*np.pi), 2)))
    ax[1].title.set_text('Graph 667, gap $={}$'.format(np.round(5.901738837773166/(2*np.pi), 2)))
    ax[3].title.set_text('Graph 776, gap $={}$'.format(np.round(4.476855347769174/(2*np.pi), 2)))
    ax[4].title.set_text('Graph 78, gap $={}$'.format(np.round(1.9899232828295226/(2*np.pi), 2)))

    ax[1].set_ylim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[3].set_ylim(0, 1)
    ax[4].set_ylim(0, 1)

    ax[1].set_xlim(.1, 13)
    ax[2].set_xlim(.1, 13)
    ax[3].set_xlim(.1, 13)
    ax[4].set_xlim(.1, 13)

    ax[1].semilogx()
    ax[2].semilogx()
    ax[3].semilogx()
    ax[4].semilogx()

    ax[1].legend(loc='lower right')
    ax[2].legend(loc='upper left')
    ax[3].legend(loc='lower right')
    ax[4].legend(loc='upper left')

    plt.show()

critical_detuning_comparison()
if __name__ == '__main__':
    import sys

    # index = int(sys.argv[1])
    rats = []
    for index in range(1,2):
        size = 7
        size_indices = np.array([5, 6, 7, 8, 9, 10, 11, 12])
        size_index = np.argwhere(size == size_indices)[0, 0]
        xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
        # print(repr(pd.read_excel(xls, 'Sheet1').to_numpy()[:20, size_index].astype(int)))
        graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])
        print('New', graph_index)
        # graph_index = index
        # graphs = np.array([188, 970, 91, 100, 72, 316, 747, 216, 168, 852, 7, 743, 32,
        #                   573, 991, 957, 555, 936, 342, 950])
        # graphs = np.array([189, 623, 354,  40, 323, 173, 661, 345, 813,  35, 162, 965, 336,
        #   667, 870,   1, 156, 901, 576, 346])

        # graph_index = graphs[index]
        # graph_index = 216
        # graph_index = 661
        # graph_index = 189
        graph_data = loadfile(graph_index, size)
        grid = graph_data['graph_mask']
        # misminusone, maximal, mis = count_maximal(graph_data['graph_mask'])
        xls = pd.ExcelFile('QuEra_graphs_degeneracies.xlsx')
        # print(repr(pd.read_excel(xls, 'Sheet1').to_numpy()[:20, size_index].astype(int)))
        #degeneracy = int(pd.read_excel(xls, 'MIS_degeneracy').to_numpy()[graph_index, size_index])
        #misminusone_degeneracy = int(pd.read_excel(xls, 'MIS-1_degeneracy').to_numpy()[graph_index, size_index])
        #maximalminusone_degeneracy = int(pd.read_excel(xls, 'maximal-1_degeneracy').to_numpy()[graph_index, size_index])
        #ratio = (pd.read_excel(xls, 'MIS-1__over_nonmaximal_ratio').to_numpy()[graph_index, size_index])
        #rats.append(ratio)
        #print(rats)
        #if index == 19:
        #    raise Exception
        #print(ratio, misminusone_degeneracy / (misminusone_degeneracy - maximalminusone_degeneracy))
        """if degeneracy != mis:
            print(graph_index, 'ERROR')
            print(degeneracy, mis)
        if misminusone_degeneracy!= misminusone:
            print(graph_index, 'ERROR')
            print(misminusone_degeneracy, misminusone)
        if maximalminusone_degeneracy!= maximal:
            print(graph_index, 'ERROR')
            print(maximalminusone_degeneracy, maximal)"""
        # print('Initializing graph')
        graph = unit_disk_grid_graph(grid, periodic=False, radius=1.51)
        graph.generate_independent_sets()
        print(graph.num_independent_sets)
        # print(np.log(graph.num_independent_sets), graph.degeneracy)
        # pass
        # print('Initializing tails graph')
        #tails_graph = rydberg_graph(grid, visualize=False)
        # print('starting function')
        # track_eigenstate_populations(graph, tails_graph, grid, k=15)

        track_eigenstate_composition(graph, None, num=1)
        #print('Final', gap, loc)
        # gaps.append(gap)
        # locs.append(loc)
        """times = np.linspace(10, 16, 7)
        times_long = np.linspace(10, 16, 10000)
        ratios, probs = find_ratio(None, graph, times)
        print(ratios, probs)
        dt = times_long[np.argmin(np.interp(times_long, times, np.abs(np.array(probs)-.9)))]
        dts.append(dt)
        print(dts)"""
        # print(gaps)
        # print(locs)
        # track_eigenstate_populations(graph, tails_graph, grid, 15)
