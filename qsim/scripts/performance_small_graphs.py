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


def find_ratio(tails_graph, graph, tf, graph_index=None, size=None):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    # print('Starting driver')
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
        cost.energies = (-2*np.pi*11*2*(1/2-t/T),)#(2 * np.pi * (-(11 + 15) / T * t + 15),)
        driver.energies = (2*np.pi*2*amplitude,)#(2 * np.pi * 2 * amplitude,)

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
    ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                           IS_subspace=True)
    # print('Starting evolution')
    ars = []
    probs = []
    for i in range(len(tf)):
        states, data = ad.run(tf[i], schedule_exp_linear, num=int(10 * tf[i]), method='odeint', full_output=False)
        cost.energies = (1,)
        ar = cost.approximation_ratio(states[-1])
        prob = cost.optimum_overlap(states[-1])
        #np.savez_compressed('{}x{}_{}_{}_opt.npz'.format(size, size, graph_index, i), state=states[-1])
        print(tf[i], ar, prob)
        #ars.append(ar)
        #probs.append(prob)
    return ars, probs


def visualize_low_energy_subspace(graph, tails_graph, k=5):
    n_points = 7
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[4]
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    # print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    # print('Starting rydberg')
    if tails_graph is not None:
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
    pulse = np.loadtxt('for_AWG_{}.000000.txt'.format(6))
    max_detuning = np.max(pulse[:, 2])
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

    # print(t, cost.energies)
    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)

    def gap(t):
        schedule_exp_linear(t, t_max)
        eigval, eigvec = eq.eig(which='S', k=k)
        return np.abs(eigval - eigval[0]), eigvec

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    if tails_graph is None:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])

    else:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])

    fig, ax = plt.subplots(1, 1)
    num = 100
    print('beginning computation')
    for (i, t) in enumerate(np.linspace(.5, .97, num)*t_max):
        print(i)
        g, eigvec = gap(t)
        schedule_exp_linear(t, t_max)
        detuning = cost.energies[0]/(2*np.pi)
        g = g/(2*np.pi)
        ax.scatter(-np.ones(len(g)) * detuning, g, s=3, color='navy')
        ax.set_xlabel(r'Detuning (MHz)')
        ax.set_ylabel(r'Eigenenergy (MHz))')
    plt.show()


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
        upper = 0.7860824#0.77
        lower = 0.7856305#0.8161#.77#.6
        # Default: .71
        init = np.array([0.7860309])#0.74
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
        graph = unit_disk_grid_graph(grid, periodic=False)
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
        print(driver.energies[0]/(2*np.pi))
        print(cost.energies[0]/(2*np.pi))
        raise Exception

    if tails_graph is None:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver])
    else:
        eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, rydberg])

    # Do minimization
    schedule = schedule_exp_linear
    print(t, t_max, t*t_max)

    t = t * t_max
    schedule(t, t_max)
    eigval, eigvec = eq.eig(which='S', k=k)
    print(eigval[0], eigval[1])
    print('gap', (eigval[1]-eigval[0])/(2*np.pi))
    return eigval, eigvec
    #np.save('eigvec_{}.npy'.format(graph_index), eigvec)

"""size = 8
index = 0
size_indices = np.array([5, 6, 7, 8, 9, 10])
size_index = np.argwhere(size == size_indices)[0, 0]
xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
print(repr(pd.read_excel(xls, 'Sheet1').to_numpy()[:10, size_index].astype(int)))
graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])
n_points = 7
times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
t_max = times_exp[6]
graph_data = loadfile(graph_index, size)
grid = graph_data['graph_mask']
print(grid)
# We want to have the file of hardest graphs, then import them
graph = unit_disk_grid_graph(grid, periodic=False, radius=1.1)
graph.generate_independent_sets()
print('degeneracy', graph.degeneracy)
print('HS size', graph.num_independent_sets)

tails_graph = rydberg_graph(grid, visualize=False)
find_ratio(tails_graph, graph, [t_max])
#visualize_low_energy_subspace(graph, tails_graph, k=5)"""


if __name__ == '__main__':
    # Evolve
    import sys

    n_points = 10
    index = 0
    #tf = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2

    #index = int(sys.argv[1])
    # time_index = index % len(tf)
    # index = index #/ len(tf)
    size = 5
    size_indices = np.array([5, 6, 7, 8, 9, 10])
    size_index = np.argwhere(size == size_indices)[0, 0]
    xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
    graph_index = (pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index]).astype(int)
    #graph_indices = np.array([189, 623, 354, 40, 323, 173, 661, 345, 813, 35, 162, 965, 336,
    #                          667, 870, 1, 156, 901, 576, 346])
    #graph_index = graph_indices[index]
    times_exp = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points) + .312 * 2
    t_max = times_exp[4]
    print(t_max)
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    print(grid)
    # grid = np.ones((1, 10))
    graph = unit_disk_grid_graph(grid, periodic=False, visualize=False, radius=1.5)
    tails_graph = rydberg_graph(grid, visualize=False)
    relevant_times = np.linspace(0.312, t_max-0.312, 10000)
    deltas = np.linspace(15, -11, 10000)
    relevant_deltas = -np.array([2, 4, 6, 8])
    indices = []
    for delta in relevant_deltas:
        indices.append(np.argmin(np.abs(deltas-delta)))
    relevant_times = relevant_times[indices]/t_max
    print(relevant_times)
    #find_ground_first_excited(graph, tails_graph, .5)
    for t in relevant_times:
        find_ground_first_excited(graph, tails_graph, 0.8096760809588257)
    #res = visualize_low_energy_subspace(graph, tails_graph, k=2)
    #print(res)
    #ratio, probs = find_ratio(tails_graph, graph, tf, graph_index=graph_index, size=size)
    #print(ratio)
    #print(probs)
