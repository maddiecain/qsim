import matplotlib.pyplot as plt
import networkx as nx
import plot
from qsim import noise_models
from qsim.qaoa import simulate, optimize

# Construct a known graph
G = nx.Graph()

G.add_edge(0, 1, weight=1)
G.add_edge(0, 2, weight=1)
G.add_edge(2, 3, weight=1)
G.add_edge(0, 4, weight=1)
G.add_edge(1, 4, weight=1)
G.add_edge(3, 4, weight=1)
G.add_edge(1, 5, weight=1)
G.add_edge(2, 5, weight=1)
G.add_edge(3, 5, weight=1)

# Uncomment to visualize graph
# plot.draw_graph(G)

sim = simulate.SimulateQAOA(G, 10, 2, is_ket=False, noisy=True,
                            noise_model=noise_models.depolarize_single_qubit)

# Test that the calculated objective function and gradients are correct
print(optimize.optimize_instance_interp_heuristic(G,  verbose = True))