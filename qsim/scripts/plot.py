import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):
    nx.draw(G)
    plt.draw()
    plt.show()