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
        driver.energies = (2 * amplitude,)

    # Uncomment this to print the schedule at t=0
    # schedule(0, 1)
    # print(cost.hamiltonian*2*np.pi)
    # print(driver.hamiltonian)
    if tails_graph is None:
        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver], cost_hamiltonian=cost,
                               IS_subspace=True)
    else:
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(1,))
        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                               IS_subspace=True)
    print('running')
    states, data = ad.run(tf, schedule, method='odeint')
    print(data)
    times = data['t']
    is_overlaps = np.zeros((graph.mis_size + 1, len(states)))
    cost.energies = (1,)
    for j in range(graph.mis_size + 1):
        projector = (cost._diagonal_hamiltonian == j)
        for i in range(len(states)):
            is_overlaps[j, i] = np.sum(np.abs(states[i]) ** 2 * projector)
    print(is_overlaps)
    for j in range(graph.mis_size + 1):
        plt.scatter(times, is_overlaps[j], label=str(j))
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
from scipy import interpolate
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y, kind='cubic')
xnew = np.arange(0, 9, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()

fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
times_exp = 2 ** np.linspace(-2.5, 2, 7)  # + .311 * 2
pulse = np.loadtxt('for_AWG_0.000000.txt')
ratios_line_18 = np.array([0.85636978, 0.93404579, 0.97442648, 0.98971751, 0.9950088, 0.99758132,
                           0.9988229, 0.99943188, 0.9997245, 0.99986662])
ratios_ring_18 = np.array([0.82140201, 0.89884519, 0.93969894, 0.95341189, 0.96378514,
                           0.97211422, 0.97861402, 0.9836049, 0.98740366, 0.99031313])
ratios_ring_19 = np.array([0.86703511, 0.94716423, 0.98230147, 0.99537633, 0.99811584,
                           0.99917558, 0.99965532, 0.99985681, 0.99994088, 0.99997698])
ratios_line_19 = np.array([0.81180318, 0.89278732, 0.93643566, 0.95185676, 0.96177094,
                           0.96967473, 0.9759006, 0.98084226, 0.98477305, 0.98789389])
ratios_ring_20 = np.array([0.82140187, 0.89802889, 0.93721277, 0.95080877, 0.95966849,
                           0.96746702, 0.97380099, 0.9789001, 0.9829763, 0.98625156])
ratios_line_20 = np.array([0.85287339, 0.93046327, 0.97099754, 0.9866697, 0.99268411,
                           0.99601385, 0.9978303, 0.99882542, 0.99936177, 0.99965323])
ratios_ring_21 = np.array([0.86247185, 0.94261942, 0.9802005, 0.99218954, 0.99638006,
                           0.99819805, 0.99913594, 0.99958842, 0.9998035, 0.99990864])
ratios_line_17 = np.array([0.81073667, 0.89208413, 0.9369285, 0.95405125, 0.96508608,
                           0.97354606, 0.97992921, 0.98476632, 0.98843963, 0.99122555])
ratios_line_15 = np.array([0.80940341, 0.89100949, 0.93871432, 0.95768451, 0.96974846,
                           0.97845186, 0.9846325, 0.98903565, 0.99217667, 0.99441943])
ratios_line_21 = np.array([0.81267579, 0.89333371, 0.93656791, 0.95067304, 0.95955762,
                           0.96679297, 0.97265934, 0.9774807, 0.98145248, 0.98472004])
ratios_line_23 = np.array([0.81340276, 0.89380291, 0.93689993, 0.95013945, 0.95817369,
                           0.96475423, 0.97017565, 0.97474525, 0.97861087, 0.98188111])
ratios_line_25 = np.array([0.81401818, 0.89420967, 0.9372477, 0.9499896, 0.95737503,
                           0.96338275, 0.96835154, 0.97260863, 0.9762813, 0.97945619])
ratios_line_rydberg_25 = np.array([0.68682793, 0.89344122, 0.92839035, 0.94673412, 0.95479584,
                                   0.96061771, 0.96571321, 0.97000527, 0.97358693, 0.97676314])
ratios_line_rydberg_23 = np.array([0.6876706, 0.8932507, 0.9284184, 0.94726082, 0.95563281])
# Correct pulse sequences
#ratios_21_optimized = np.array([0.4205557, 0.64163789, 0.7773311, 0.86705929, 0.91011691, 0.93389629, 0.95154214])
#ratios_23_optimized = np.array([0.42055570, 0.64163789, 0.77733110, 0.8653093, 0.90879235, 0.93345944, 0.95187553])
ratios_25_optimized = np.array([0.51864428, 0.74789354, 0.86540587, 0.93852973, 0.97278725, 0.99621505, 0.99637481])
ratios_21_linear = np.array([0.72038517, 0.83105992, 0.87525200, 0.90675623, 0.92993901, 0.94761466, 0.96546305])
ratios_25_linear = np.array([0.72285420, 0.83334918, 0.87705373, 0.90805359, 0.93082555, 0.94722681, 0.96189619])
ratios_25_linear_smooth = np.array([0.87682821, 0.89961852, 0.91698225, 0.93134324, 0.94238466, 0.95329563, 0.96483817])
ratios_exp_optimized = np.array([0.42818704, 0.2531182, 0.14880018, 0.09479478, 0.07501056,
                                 0.07799798, 0.08966179])
ratios_exp_linear = np.array([0.2331565, 0.17754241, 0.13804714, 0.11148272, 0.09503807,
                              0.09020094, 0.08277972])

ratios_27_optimized = np.array([0.52021872, 0.74954731, 0.86670179, 0.93909762, 0.96994163, 0.99530984, 0.99678150])
ratios_29_optimized = np.array([0.52158324, 0.75098058, 0.86782495, 0.93964521])

# plt.scatter(times, 1 - ratios_line_21, marker='.', color='purple', label=r'$n=21$ line')
# plt.scatter(times, 1 - ratios_line_23, marker='.', color='orange', label=r'$n=23$ line')
# plt.scatter(times, 1 - ratios_line_25, marker='.', color='black', label=r'$n=25$ line')
# plt.scatter(times, 1 - ratios_line_rydberg_25, marker='*', color='black', label=r'$n=25$ line Rydberg')
# plt.scatter(times[:len(ratios_line_rydberg_23)], 1 - ratios_line_rydberg_23, marker='*', color='orange',
#            label=r'$n=23$ line Rydberg')
# plt.scatter(times_exp, 1 - ratios_21_optimized, color='orange', label=r'$n=21$ optimized')
# plt.scatter(times_exp, 1 - ratios_23_optimized, color='red', label=r'$n=23$ optimized')
ax[0].scatter(times_exp, 1 - ratios_25_optimized, color='red', label=r'$n=25$ optimized')
ax[0].scatter(times_exp, 1 - ratios_27_optimized, color='red', label=r'$n=27$ optimized', marker='^')
ax[0].scatter(times_exp[:len(ratios_29_optimized)], 1 - ratios_29_optimized, color='red', label=r'$n=29$ optimized', marker='v')

# plt.scatter(times_exp, 1 - ratios_21_linear, color='orange', marker='*', label=r'$n=21$ linear')
# plt.scatter(times_exp, 1 - ratios_23_linear, color='red', marker='*', label=r'$n=23$ linear')
ax[1].scatter(times_exp, 1 - ratios_25_linear, color='purple', label=r'$n=25$ linear')
# plt.scatter(times_exp, 1 - ratios_21_linear, color='blue', marker='^', label=r'$n=21$ linear')
ax[0].scatter(times_exp, ratios_exp_optimized, color='red', marker='*', label=r'$n=25$ exp optimized')
ax[1].scatter(times_exp, ratios_exp_linear, color='purple', marker='*', label=r'$n=25$ exp optimized')

# plt.scatter(times_exp, 1 - ratios_25_linear_smooth, color='blue', marker='^', label=r'$n=25$ linear smooth')
res_optimized, err_optimized = scipy.optimize.curve_fit(lambda x, a, b: b * x ** a, times_exp[:4],
                                                        1 - ratios_25_optimized[:4])
res_linear, err_linear = scipy.optimize.curve_fit(lambda x, a, b: b * x ** a, times_exp[-5:], 1 - ratios_25_linear[-5:])
# res_linear_smooth, err_linear_smooth = scipy.optimize.curve_fit(lambda x, a, b: b * x ** a, times_exp[-3:], 1 - ratios_25_linear_smooth[-3:])

ax[0].plot(times_exp, res_optimized[1] * times_exp ** res_optimized[0],
         label=r'slope$=$' + str(np.round(res_optimized[0], 3)), color='k', linestyle='dashed')
ax[1].plot(times_exp, res_linear[1] * times_exp ** res_linear[0],
         label=r'slope$=$' + str(np.round(res_linear[0], 3)), color='k', linestyle=':')
# plt.plot(times_exp, res_linear_smooth[1] * times_exp ** res_linear_smooth[0],
#         label=r'slope$=$'+str(np.round(res_linear_smooth[0], 3)), color='k', linestyle='solid')
ax[0].set_xlabel('Total depth')
ax[1].set_xlabel('Total depth')

ax[0].set_ylabel(r'$\varepsilon$')
plt.loglog()
ax[0].legend(fontsize='small', frameon=False)
ax[1].legend(fontsize='small', frameon=False)

plt.show()
# Evolve
# import sys
# index = int(sys.argv[1])
index = 0
size = 6
size_indices = np.array([5, 6, 7, 8, 9, 10])
size_index = np.argwhere(size == size_indices)[0, 0]
xls = pd.ExcelFile('MIS_degeneracy_ratio.xlsx')
graph_index = int(pd.read_excel(xls, 'Sheet1').to_numpy()[index, size_index])

graph_data = loadfile(graph_index, size)
grid = graph_data['graph_mask']
times = 2 * np.pi * 2 ** np.linspace(-2.5, 2, 7)

graph = unit_disk_grid_graph(grid, periodic=False, visualize=False)
find_ratio(None, graph, 60)
