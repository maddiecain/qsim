import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import unit_disk_grid_graph

"""
For various lattices, compute the defect density as a function of ramp time. 
"""


def two_point_correlator(graph, state):
    """Compute the two point correlator for every pair of nodes in the graph."""
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v:
                # Compute two point correlator
                for i in range(state.shape[0]):
                    pass




graph = unit_disk_grid_graph(np.ones((3, 4)), IS=True)
two_point_correlator(graph)