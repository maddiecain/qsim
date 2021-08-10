import matplotlib.pyplot as plt
import numpy as np
from qsim.evolution import hamiltonian
from qsim.graph_algorithms.graph import unit_disk_grid_graph, rydberg_graph
from qsim.graph_algorithms.adiabatic import SimulateAdiabatic


def find_ratio(graph, tf):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True)

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
    #schedule(0, 1)
    #print(cost.hamiltonian*2*np.pi)
    #print(driver.hamiltonian)

    ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                           IS_subspace=True)

    final_state = ad.run(tf, schedule, method='odeint')[0][-1]
    cost.energies = (1,)
    ar = cost.approximation_ratio(final_state)
    return ar

# THESE ARE MISLABELED
# 410
grid1 = np.array([[True, True, True, True, True],
                  [True, True, True, False, True],
                  [False, True, True, True, False],
                  [True, True, True, True, True],
                  [True, False, False, True, True]])
# 407
grid2 = np.array([[True, True, False, True, True],
                  [False, True, True, True, False],
                  [True, True, True, True, False],
                  [True, True, True, True, True],
                  [True, True, False, True, True]])
# 396
grid3 = np.array([[True, True, True, True, True],
                  [True, False, False, True, True],
                  [False, True, True, True, False],
                  [True, True, True, False, True],
                  [True, True, True, True, True]])

# 6x6 graphs
"""#667
grid1 = np.array([[False,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True,  True],
       [ True,  True, False,  True,  True,  True],
       [ True,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True, False],
       [ True,  True, False,  True,  True, False]])
#557
grid2 = np.array([[False, False,  True,  True,  True, False],
       [False,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True, False],
       [False,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True, False]])
#78
grid3 = np.array([[False, False,  True,  True,  True,  True],
       [False,  True, False,  True,  True,  True],
       [ True,  True,  True,  True,  True, False],
       [False,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True, False]])"""

graph = unit_disk_grid_graph(grid3, periodic=False)
print('MIS size', graph.mis_size)
print('Hilbert space size', graph.num_independent_sets)
tails_graph = rydberg_graph(grid3, visualize=False)
# Evolve
times = 2*np.pi*2**np.linspace(-2.5, 2, 7)
ratios = []

for time in times:
    print('Time', time)
    ratio = find_ratio(graph, time)
    print('Ratio', ratio)
    ratios.append(ratio)
    plt.scatter(time/2/np.pi, 1 - ratio, color='navy')

plt.loglog()
plt.ylabel(r'1-ratio')
plt.xlabel(r'total time ($\mu s$)')
plt.title(r'$5\times 5$ graphs, $r=1.50$ with no additions')
plt.show()

plt.clf()
# Results for time series
times = 2*np.pi*np.linspace(.1, 4, 20)
ratios_410 = 1-np.array([0.6312698400365628, 0.6350631502841703, 0.7363947755015506, 0.7662979536725513, 0.8167691849746471, 0.8623958274791159, 0.8913939185975078, 0.912168628992816, 0.9321780785078123, 0.9462223416029758, 0.9565127845162442, 0.9662627545199562, 0.9728921938205923, 0.9781659289712348, 0.9827158152246354, 0.9860167897451267, 0.9887534729034296, 0.9909437382177713, 0.9926390836244009, 0.9940074711333567])
plt.plot(times/2/np.pi, ratios_410, color='navy', label='410')
plt.scatter(times/2/np.pi, ratios_410, color='navy')
ratios_407 = 1-np.array([0.630063776878089, 0.6416610996304736, 0.7444381410820011, 0.7829055244953346, 0.8353697195102272, 0.8817360513054348, 0.9132642942385995, 0.92883017789128, 0.9477625711705338, 0.9610277728044203, 0.9699514991171989, 0.9775476589769881, 0.9833353927192476, 0.987173459261333, 0.9904535712504051, 0.9927588688193951, 0.9944782876829874, 0.9958044215213879, 0.9967836134836626, 0.9975004790785146])
plt.plot(times/2/np.pi, ratios_407, color='cornflowerblue', label='407')
plt.scatter(times/2/np.pi, ratios_407, color='cornflowerblue')
ratios_396 = 1-np.array([0.570659624797462, 0.6308651114758622, 0.7424121339736333, 0.7870918781716714, 0.8284875558347364, 0.8765767421749913, 0.89789528347941, 0.9269253219093562, 0.9443487385154993, 0.9568907893731936, 0.9683788011764236, 0.9760734973226479, 0.9822823276485927, 0.9869070815629714, 0.990429418338397, 0.9930128114995059, 0.9949157145006375, 0.9963084836091451, 0.9973188688684352, 0.9980537941090958])
plt.plot(times / 2 / np.pi, ratios_396, color='teal', label='396')
plt.scatter(times / 2 / np.pi, ratios_396, color='teal')
plt.legend()
plt.ylabel(r'1-ratio')
plt.xlabel(r'total time ($\mu s$)')
plt.title(r'$5\times 5$ graphs, $r=1.50$ with no additions')
plt.loglog()
plt.show()


