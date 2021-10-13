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


def find_ratio(tails_graph, graph, detuning_plateau, tf):
    cost = hamiltonian.HamiltonianMIS(graph, IS_subspace=True)
    print('Starting driver')
    driver = hamiltonian.HamiltonianDriver(IS_subspace=True, graph=graph)
    print('Starting rydberg')
    if tails_graph is not None:
        rydberg = hamiltonian.HamiltonianRydberg(tails_graph, graph, IS_subspace=True, energies=(2 * np.pi,))
        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver, rydberg], cost_hamiltonian=cost,
                               IS_subspace=True)
    else:
        ad = SimulateAdiabatic(graph=graph, hamiltonian=[cost, driver], cost_hamiltonian=cost,
                               IS_subspace=True)



    def schedule_cubic(t, T):
        cubic_ys = 2 * np.pi * np.array([11.5, detuning_plateau+.5, detuning_plateau, detuning_plateau-.5 , -11.5])
        cubic_xs = np.array([.312, (T / 2 - .312) / 1.35 + .312, T / 2, T - .312-(T / 2 - .312) / 1.35, T - .312])
        if t < .312:
            driver.energies = (2 * np.pi * 2 * t / .312,)
            cost.energies = (2 * np.pi * 11.5,)
        elif .312 <= t <= T - .312:
            driver.energies = (2 * np.pi * 2,)
            cost.energies = (scipy.interpolate.interp1d(cubic_xs, cubic_ys, kind='cubic')(t),)
        else:
            driver.energies = (2 * np.pi * 2 * (T - t) / .312,)
            cost.energies = (-2 * np.pi * 11.5,)

    print('Starting evolution')
    states, data = ad.run(tf, schedule_cubic, num=int(400 * tf), method='trotterize',
                          full_output=False)
    np.savez_compressed('{}x{}_{}_{}_{}_trotterize_highres.npz'.format(size, size, graph_index, np.round(np.abs(detuning_plateau), 2), np.round(np.abs(tf), 2)), state=states[-1])


if __name__ == '__main__':
    import sys

    size = 6
    # 4 or 0
    index = int(sys.argv[1])
    critical_detuning = -9.604213726908476#-6.019449429835163#-7.001#-8.495#
    critical_detunings = np.concatenate([-np.linspace(2, 10, 10), [critical_detuning]])
    #graph_index = 667
    graph_index = 807
    #graph_index = 173
    #graph_index = 336
    graph_data = loadfile(graph_index, size)
    grid = graph_data['graph_mask']
    print('Initializing graph')
    graph = unit_disk_grid_graph(grid, periodic=False, radius=1.1)
    tails_graph = rydberg_graph(grid, visualize=False)
    n_points = 7
    times = 2 ** np.linspace(-2.5, 4.5 / 6 * (n_points - 1) - 2.5, n_points)
    times = np.concatenate([times, [2.5]])#np.array([2.5])#
    times = times + .312 * 2
    time = times[index // len(critical_detunings)]
    # 8 times, 11 detunings = 0-87 array
    detuning = critical_detunings[index % len(critical_detunings)]
    print(time, detuning)
    #raise Exception
    find_ratio(tails_graph, graph, detuning, time)
