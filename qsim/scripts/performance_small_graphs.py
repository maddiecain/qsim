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


def cost_reduction():
    pass


def cost_addition():
    pass


def loadfile(graph_index, size):
    graph_mask = np.reshape(np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 3:],
                            (size, size), order='F')[::-1, ::-1].T.astype(bool)
    MIS_size = np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 0].astype(int)
    degeneracy = np.loadtxt("mis_degeneracy_L%d.dat" % size)[graph_index, 1].astype(int)
    graph_data = {'graph_index': graph_index, 'MIS_size': MIS_size, 'degeneracy': degeneracy,
                  'side_length': size,
                  'graph_mask': graph_mask, 'number_of_nodes': int(np.sum(graph_mask))}
    return graph_data


def find_ratio(tails_graph, graph, tf):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    #print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    #print('Starting rydberg')
    rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2*np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    t_pulse_max = np.max(pulse[:, 0]) - 2 * 0.311

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

    def schedule_exp_optimized(t, T):
        if t < .311:
            driver.energies = (2 * np.pi * 2 * t / .311,)
            cost.energies = (2 * np.pi * 15,)
        elif .311 <= t <= T - .331:
            t_pulse = (t - 0.311) / (T - 2 * 0.311) * t_pulse_max + 0.311
            driver.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], pulse[:, 1] / 2),)
            cost.energies = (2 * np.pi * np.interp(t_pulse, pulse[:, 0], -pulse[:, 2]),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .311,)
            cost.energies = (-2 * np.pi * 11,)
        # print(t, cost.energies)

    def schedule_exp_linear(t, T):
        if t < .311:
            driver.energies = (2 * np.pi * 2 * t / .311,)
            cost.energies = (2 * np.pi * 15,)
        elif .311 <= t <= T - .331:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (2 * np.pi * (-(11 + 15) / (T - 2 * .311) * (t - .311) + 15),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .311,)
            cost.energies = (-2 * np.pi * 11,)
    # print(t, cost.energies)
    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)

    ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                           IS_subspace=True)
    #print('Starting evolution')
    ars = []
    for i in range(len(tf)):
        states, data = ad.run(tf[i], schedule_exp_optimized, num=int(200 * tf[i]), method='trotterize', full_output=False)
        cost.energies = (1,)
        ar = cost.approximation_ratio(states[-1])
        print(tf[i], ar)
        ars.append(ar)
    return ars


def visualize_low_energy_subspace(graph, k=5):
    print('initializing')
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    print('finished MIS')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    print('finished driver')
    # rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(1,))
    grid = graph.positions

    # random = hamiltonian.HamiltonianEnergyShift(IS_subspace=True, graph=graph, energies=(1e-6,))
    # random._hamiltonian = scipy.sparse.csc_matrix(
    #    (np.random.normal(loc=1e3, scale=1e3, size=graph.num_independent_sets), (np.arange(graph.num_independent_sets),
    #                                                                             np.arange(
    #                                                                                 graph.num_independent_sets))),
    #    shape=(graph.num_independent_sets, graph.num_independent_sets))

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
        cost.energies = (5 * 2 * (1 / 2 - x),)
        driver.energies = (amplitude,)

    def gap(t):
        schedule(t, 1)
        eigval, eigvec = eq.eig(which='S', k=k)
        return np.abs(eigval - eigval[0]), eigvec

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])
    fig = plt.figure(tight_layout=True)
    k_cutoff = 5
    gs = gridspec.GridSpec(k_cutoff, k_cutoff)
    ax = fig.add_subplot(gs[:, 0:k_cutoff - 1])
    num = 50
    print('beginning computation')
    for (i, t) in enumerate(np.linspace(.5, .95, num)):
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


def find_gap(graph, k=2):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    # rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(1,))
    grid = graph.positions
    random = hamiltonian.HamiltonianEnergyShift(IS_subspace=True, graph=graph, energies=(1e-6,))
    random._hamiltonian = scipy.sparse.csc_matrix(
        (np.random.normal(loc=1e3, scale=1e3, size=graph.num_independent_sets), (np.arange(graph.num_independent_sets),
                                                                                 np.arange(
                                                                                     graph.num_independent_sets))),
        shape=(graph.num_independent_sets, graph.num_independent_sets))

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
        cost.energies = (11 * 2 * (1 / 2 - x),)
        driver.energies = (2*amplitude,)

    def gap(t):
        t = t[0]
        schedule(t, 1)
        eigval, eigvec = eq.eig(which='S', k=k)
        print(t, np.abs(eigval[1] - eigval[0]))
        return np.abs(eigval[1] - eigval[0])

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, random])
    # Do minimization
    res = scipy.optimize.minimize(gap, np.array([.645]), bounds=[(.55, .7)], tol=1e-2)
    return res.fun, res.x[0]


def collect_gap_statistics(size):
    size_indices = np.array([5, 6, 7, 8, 9, 10])
    size_index = np.argwhere(size == size_indices)[0, 0]
    gaps = []
    ratios = []
    degeneracies = []
    for index in range(100):
        xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
        """graphs = np.array([667, 557,  78, 312, 807, 776, 485, 980,  71,  50, 521, 773, 549,
           523, 374, 515, 669, 344,  21, 107, 201, 851, 736, 508, 286, 526,
           385, 116,  20, 999, 357, 149, 872, 233, 528, 603, 912, 820, 155,
           744, 438, 931,  68, 610, 209, 876, 558, 809, 702, 194, 828, 437,
           470, 958, 359, 677, 185, 813, 715, 420, 153, 573, 394, 542, 688,
           863, 771, 325, 502, 795, 617, 722, 793, 182, 363, 984, 447, 506,
           673, 950, 329, 127, 492, 428, 343, 391, 812, 949,  69, 265, 276,
           564, 336, 966, 963, 219, 321, 144, 435, 696])"""

        # graph_index = graphs[index]
        graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])

        graph_data = loadfile(graph_index, size)
        grid = graph_data['graph_mask']
        # We want to have the file of hardest graphs, then import them
        graph = unit_disk_grid_graph(grid, periodic=False, generate_mixer=True)
        raise Exception
        # print(np.arange(graph.num_independent_sets, 0, -1))
        # print(np.sum(2 ** np.arange(graph.n, 0, -1) * (1 - graph.independent_sets), axis=1))
        # print(1-graph.independent_sets)
        print('Degeneracy', graph.degeneracy)
        print('MIS size', graph.mis_size)
        print('Hilbert space size', graph.num_independent_sets)
        # tails_graph = rydberg_graph(grid, visualize=False)
        gap, loc = find_gap(graph, k=2)
        gaps.append(gap)
        degeneracies.append(graph.degeneracy)
        is_sizes = np.sum(1 - graph.independent_sets, axis=1)
        ratio = np.sum(is_sizes == graph.mis_size - 1) / np.sum(is_sizes == graph.mis_size)
        ratios.append(ratio)
        print(gaps, degeneracies)

        # print(index, graph_index, gap, loc, ratio, graph.degeneracy)
        # visualize_low_energy_subspace(graph)


if __name__ == '__main__':
    # Evolve
    import sys
    tf = (2**np.linspace(-2.5, 2, 7)+2*.311)#*(2*np.pi)

    index = int(sys.argv[1])
    #time_index = index % len(tf)
    #index = index #/ len(tf)
    size = 7
    size_indices = np.array([5, 6, 7, 8, 9, 10])
    size_index = np.argwhere(size == size_indices)[0, 0]
    #xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
    #graph_index = (pd.read_excel(xls, 'Sheet1').to_numpy()[0:20, size_index]).astype(int)
    graph_indices = np.array([189, 623, 354,  40, 323, 173, 661, 345, 813,  35, 162, 965, 336,
       667, 870,   1, 156, 901, 576, 346])
    graph_index = graph_indices[index]

    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    graph = unit_disk_grid_graph(grid, periodic=False, visualize=False, generate_mixer=True)
    tails_graph = rydberg_graph(grid, visualize=False)
    ratio = find_ratio(tails_graph, graph, tf)
    print(ratio)

