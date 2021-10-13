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
    print('Starting cost')
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
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
        upper = 0.7860824  # 0.77
        lower = 0.7856305  # 0.8161#.77#.6
        # Default: .71
        init = np.array([0.7860309])  # 0.74
    print('Starting gap')
    res_linear = scipy.optimize.minimize(gap, init, bounds=[(lower, upper)], method='L-BFGS-B')
    terminate = False
    while res_linear.x[0] == upper and not terminate:
        upper -= .01
        if upper <= lower:
            upper = .95
        res_linear = scipy.optimize.minimize(gap, init, bounds=[(lower, upper)], tol=1e-2)

    return res_linear.fun, res_linear.x[0]


def find_ground_first_excited(graph, tails_graph, t, k=2):
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[4]
    print('Starting cost')
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
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

    if tails_graph is None:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])
    else:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])

    # Do minimization
    schedule = schedule_exp_linear
    t = t * t_max
    schedule(t, t_max)
    eigval, eigvec = eq.eig(which='S', k=k)
    print(eigval[0], eigval[1])
    return eigval, eigvec
    # np.save('eigvec_{}.npy'.format(graph_index), eigvec)


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


size = 12
index = 0
size_indices = np.array([5, 6, 7, 8, 9, 10, 11, 12])
size_index = np.argwhere(size == size_indices)[0, 0]
xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
print(repr(pd.read_excel(xls, 'Sheet1').to_numpy()[:10, size_index].astype(int)))
gaps = []
locs = []
dts = []

for index in range(20):
    size = 7
    locs_notails_6 = np.array(
        [0.6354043718272815, 0.6353015471611612, 0.632269454494533, 0.6291213686378659, 0.6334695987119048,
         0.6287726138498437, 0.6320992450429324, 0.6287303853367949, 0.6274379022570347, np.inf,
         0.6170173651783496, 0.6308029057285616, np.inf, np.inf, 0.6157494497321381, 0.6246099627166594,
         0.6201706053474059, 0.627601395447675, 0.6167364634091927, np.inf])
    # xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
    # print(repr(pd.read_excel(xls, 'Sheet1').to_numpy()[:20, size_index].astype(int)))
    # graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])
    locs_7 = np.array(
        [0.7265036496277152, 0.7480795397419759, 0.7501160681277541, 0.7283436417782508, 0.7161896597170012,
         0.750875468376516, 0.6499999999999999, 0.7376863465734378, 0.7357162761418884, 0.7405728347818966,
         0.7231423513678857, 0.7210066666698024, 0.7107173688294144, 0.7240844082405429, 0.7595262897974576,
         0.7253202079003382, 0.7194599463412628, 0.7604139611843566, 0.7102416075728506, 0.7285607609781394,
         0.7310691332732819, 0.7471911433440447, 0.7390912620702788, 0.7184798282417322, 0.7337618912775123,
         0.6962885788468285, 0.7218528272008484, 0.749878029970634, 0.7206609232120385, 0.6887485236896556,
         0.7158657047278957, 0.7278519162920911, 0.735449028823428, 0.7016023773016111, 0.7361234146285848,
         0.7351237755884046, 0.697861342950187, 0.7398546759713864])
    graphs = np.array(
        [667, 557, 78, 312, 807, 776, 485, 980, 71, 50, 521, 773, 549, 523, 374, 515, 669, 344, 21, 107, 201,
         851, 736, 508, 286, 526, 385, 116, 20, 999, 357, 149, 872, 233, 528, 603, 912, 820])

    graphs = np.array([189, 623, 354, 40, 323, 173, 661, 345, 813, 35, 162, 965, 336,
                       667, 870, 1, 156, 901, 576, 346])

    graph_index = graphs[index]
    t = locs_7[index]
    # graph_index = 661
    # graph_index = 189
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    print('Initializing graph')
    graph = unit_disk_grid_graph(grid, periodic=False, radius=1.51)
    tails_graph = rydberg_graph(grid, visualize=False)
    # graph.generate_independent_sets()
    if not np.isinf(t):
        eigval, eigvec = find_ground_first_excited(graph, tails_graph, t, k=3)
        np.save('{}x{}_eigvec_{}.npy'.format(size, size, graph_index), eigvec)

"""for index in range(20):
    import sys
    #index = int(sys.argv[1])
    size = 6
    #size_indices = np.array([5, 6, 7, 8, 9, 10])
    #size_index = np.argwhere(size == size_indices)[0, 0]
    gaps_8 = np.array([0.018555264577685193, 0.3494778895114905, 1.7053519896727494, 1.7973674033959242,
                       1.3194361160893777, 0.3789663220178454, 2.155533273799165, 4.454764452556901,
                       0.677728932914647, 1.108900340765672, 1.674387563824439, np.inf,
                       np.inf, np.inf, np.inf, np.inf,
                       np.inf, np.inf, np.inf, np.inf])
    locs_8 = np.array([0.827956322052554, 0.8040219184932881, 0.7305878228970192, 0.7585904125924001,
                       0.732592062802305, 0.7737544889181655, 0.746496357369199, 0.7207227336013631,
                       0.7766741959601992, 0.7508037254169446, 0.7442339359689708, np.inf,
                       np.inf, np.inf, np.inf, np.inf,
                       np.inf, np.inf, np.inf, np.inf])
    locs_notails_6 = np.array(
        [0.6354043718272815, 0.6353015471611612, 0.632269454494533, 0.6291213686378659, 0.6334695987119048,
         0.6287726138498437, 0.6320992450429324, 0.6287303853367949, 0.6274379022570347, np.inf,
         0.6170173651783496, 0.6308029057285616, np.inf, np.inf, 0.6157494497321381, 0.6246099627166594,
         0.6201706053474059, 0.627601395447675, 0.6167364634091927, np.inf])
    #xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
    #print(repr(pd.read_excel(xls, 'Sheet1').to_numpy()[:20, size_index].astype(int)))
    #graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])

    graphs = np.array([188, 970, 91, 100, 72, 316, 747, 216, 168, 852, 7, 743, 32,
                       573, 991, 957, 555, 936, 342, 950])
    graphs = np.array(
        [667, 557, 78, 312, 807, 776, 485, 980, 71, 50, 521, 773, 549, 523, 374, 515, 669, 344, 21, 107, 201,
         851, 736, 508, 286, 526, 385, 116, 20, 999, 357, 149, 872, 233, 528, 603, 912, 820])

    #graphs = np.array([189, 623, 354,  40, 323, 173, 661, 345, 813,  35, 162, 965, 336,
    #   667, 870,   1, 156, 901, 576, 346])

    graph_index = graphs[index]
    t = locs_notails_6[index]
    #graph_index = 661
    #graph_index = 189
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    print('Initializing graph')
    graph = unit_disk_grid_graph(grid, periodic=False, radius=1.51)
    #print(np.log(graph.num_independent_sets), graph.degeneracy)
    #pass
    #print('Initializing tails graph')
    #tails_graph = rydberg_graph(grid, visualize=False)

    #print('starting function')
    #track_eigenstate_populations(graph, tails_graph, grid, k=15)
    if not np.isinf(t):
        eigval, eigvec = find_ground_first_excited(graph, None, t, k=2)
        np.save('{}x{}_eigvec_{}.npy'.format(size, size, graph_index), eigvec)
    #print('Final', gap, loc)
    #gaps.append(gap)
    #locs.append(loc)

    #print(gaps)
    #print(locs)
    #track_eigenstate_populations(graph, tails_graph, grid, 15)
"""
