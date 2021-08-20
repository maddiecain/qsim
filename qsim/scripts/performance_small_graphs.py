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
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(1,))

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
        driver.energies = (4 * amplitude,)

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)

    ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                           IS_subspace=True)

    final_state = ad.run(tf, schedule, method='odeint')[0][-1]
    cost.energies = (1,)
    ar = cost.approximation_ratio(final_state)
    return ar


def visualize_low_energy_subspace(graph, k=5):
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
    eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, random])
    fig = plt.figure(tight_layout=True)
    k_cutoff = 5
    gs = gridspec.GridSpec(k_cutoff, k_cutoff)
    ax = fig.add_subplot(gs[:, 0:k_cutoff - 1])
    num = 200
    for (i, t) in enumerate(np.linspace(.05, .95, num)):
        print(i)
        g, eigvec = gap(t)
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
        cost.energies = (5 * 2 * (1 / 2 - x),)
        driver.energies = (amplitude,)

    def gap(t):
        t = t[0]
        schedule(t, 1)
        eigval, eigvec = eq.eig(which='S', k=k)
        return np.abs(eigval[1] - eigval[0])

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    eq = schrodinger_equation.SchrodingerEquation(hamiltonians=[cost, driver, random])
    # Do minimization
    return scipy.optimize.minimize(gap, np.array([.63]), bounds=[(.5, .68)], tol=1e-2).fun


"""gaps = np.array([0.5020390838524253, 0.49406980906189446, 0.5578671193299023, 0.5682452892269829, 0.5564839733350588, 0.5551982927732428, 0.6197667567917655, 0.7096368857589059, 0.5525375459495105, 0.7309634399792699, 0.6890373439682342, 0.7050813597692294, 0.7051066962851582, 0.7146592090624981, 0.7250230280073069, 0.6786498194974975, 0.6625055636487698, 0.8785021342453643, 0.8145767848451175, 0.7038690448021434, 0.8784281926118798, 0.7206137688950331, 0.841583058127604, 0.7430021851145288, 0.7425451480430034, 0.7413106249240862, 0.7408932196924596, 0.5977275471880112, 0.8275948725226865, 0.6852943520829111, 0.8393314782629524, 0.8400803778070784, 0.8138045171292774, 0.7911154114349053, 0.8118002009920655, 0.6852374553005998, 0.8276171617588712, 0.8276685056095321, 0.6348678430839101, 0.6968779721318974, 0.715008637535222, 0.7336275320361825, 0.8072057132808244, 0.910128784719733, 0.7124284365965838, 0.712437896131954, 1.1328240737616717, 1.1418965577886855, 0.7993776021279224, 0.7827528712412537, 1.1408664131914765, 0.8144205107611171, 0.9868511192092964, 1.1416286234877937, 0.8036866647687511, 1.1394514075015056, 0.7806245763275967, 0.7571833189473764, 0.8362776429995513, 0.7867164462752037, 0.6360074928540236, 0.7356544085473153, 0.762529887615834, 0.8891864427643448, 1.140899044112249, 0.97476336746616, 0.8142365432286596, 1.149107884599852, 0.876712000417891, 0.7132383792767065, 1.129714243497144, 1.131473517173795, 0.7082766356957926, 0.7812759737940862, 0.8845863631400821, 0.7580294063116835, 1.1297322610287228, 1.1384763868430774, 0.7556774817338123, 0.7748748428642287, 1.0880981169523363, 0.9789711098750313, 1.1386336402337456, 1.1355791904849966, 1.1470671012034792, 0.944217878233351, 1.0959774739975732, 1.141808141117144, 1.1096187071615713, 1.0758793841547458, 1.1465369875301743, 0.7586312958530161, 1.1388819898113525, 0.7632140928767601, 1.0498850610703414, 0.773831768108332, 1.1096207964367046, 0.7586133421787125, 1.1452288103942099, 1.1100221019536018])
ratios = np.array([45.5, 43.5, 36.0, 36.0, 34.0, 34.0, 33.0, 32.0, 32.0, 30.0, 30.0, 29.5, 29.5, 29.0, 28.0, 28.0, 28.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 26.0, 26.0, 26.0, 26.0, 25.5, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 24.666666666666668, 24.5, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 22.5, 22.5, 22.5, 22.5, 22.333333333333332, 22.25, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 21.8, 21.5, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0])
plt.scatter(1/ratios, gaps,color='cornflowerblue')
plt.ylabel(r'adiabatic gap ($\Omega_{\max} = 1$)')
plt.xlabel(r'(# MIS)/(# MIS-1)')
plt.show()"""

size = 6
size_indices = np.array([5, 6, 7, 8, 9, 10])
size_index = np.argwhere(size == size_indices)[0, 0]
#gaps = []
#ratios = []
if __name__ == '__main__':
    import sys
    index = int(sys.argv[1])
    xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
    graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    # We want to have the file of hardest graphs, then import them
    graph = unit_disk_grid_graph(grid, periodic=False)
    # print(np.arange(graph.num_independent_sets, 0, -1))
    # print(np.sum(2 ** np.arange(graph.n, 0, -1) * (1 - graph.independent_sets), axis=1))
    # print(1-graph.independent_sets)
    #print('Degeneracy', graph.degeneracy)
    #print('MIS size', graph.mis_size)
    #print('Hilbert space size', graph.num_independent_sets)
    tails_graph = rydberg_graph(grid, visualize=False)
    gap = find_gap(graph, k=2)
    #gaps.append(gap)
    is_sizes = np.sum(1 - graph.independent_sets, axis=1)
    ratio = np.sum(is_sizes == graph.mis_size - 1) / np.sum(is_sizes == graph.mis_size)
    #ratios.append(ratio)
    print(gap, ratio, graph.degeneracy)
"""# Evolve
times = 2 * np.pi * 2 ** np.linspace(-2.5, 2, 7)
ratios = []

for time in times:
    print('Time', time)
    ratio = find_ratio(graph, time)
    print('Ratio', 1 - ratio)
    ratios.append(ratio)
    plt.scatter(time / 2 / np.pi, 1 - ratio, color='navy')

print(1 - np.array(ratios))
plt.loglog()
plt.ylabel(r'1-ratio')
plt.xlabel(r'total time ($\mu s$)')
plt.title(r'$5\times 5$ graphs, $r=1.50$ with no additions')
plt.show()

plt.clf()
# Results for time series
times = 2 * np.pi * np.linspace(.1, 4, 20)
ratios_410 = 1 - np.array(
    [0.6312698400365628, 0.6350631502841703, 0.7363947755015506, 0.7662979536725513, 0.8167691849746471,
     0.8623958274791159, 0.8913939185975078, 0.912168628992816, 0.9321780785078123, 0.9462223416029758,
     0.9565127845162442, 0.9662627545199562, 0.9728921938205923, 0.9781659289712348, 0.9827158152246354,
     0.9860167897451267, 0.9887534729034296, 0.9909437382177713, 0.9926390836244009, 0.9940074711333567])
plt.plot(times / 2 / np.pi, ratios_410, color='navy', label='410')
plt.scatter(times / 2 / np.pi, ratios_410, color='navy')
ratios_407 = 1 - np.array(
    [0.630063776878089, 0.6416610996304736, 0.7444381410820011, 0.7829055244953346, 0.8353697195102272,
     0.8817360513054348, 0.9132642942385995, 0.92883017789128, 0.9477625711705338, 0.9610277728044203,
     0.9699514991171989, 0.9775476589769881, 0.9833353927192476, 0.987173459261333, 0.9904535712504051,
     0.9927588688193951, 0.9944782876829874, 0.9958044215213879, 0.9967836134836626, 0.9975004790785146])
plt.plot(times / 2 / np.pi, ratios_407, color='cornflowerblue', label='407')
plt.scatter(times / 2 / np.pi, ratios_407, color='cornflowerblue')
ratios_396 = 1 - np.array(
    [0.570659624797462, 0.6308651114758622, 0.7424121339736333, 0.7870918781716714, 0.8284875558347364,
     0.8765767421749913, 0.89789528347941, 0.9269253219093562, 0.9443487385154993, 0.9568907893731936,
     0.9683788011764236, 0.9760734973226479, 0.9822823276485927, 0.9869070815629714, 0.990429418338397,
     0.9930128114995059, 0.9949157145006375, 0.9963084836091451, 0.9973188688684352, 0.9980537941090958])
plt.plot(times / 2 / np.pi, ratios_396, color='teal', label='396')
plt.scatter(times / 2 / np.pi, ratios_396, color='teal')
plt.legend()
plt.ylabel(r'1-ratio')
plt.xlabel(r'total time ($\mu s$)')
plt.title(r'$5\times 5$ graphs, $r=1.50$ with no additions')
plt.loglog()
plt.show()
"""