import networkx as nx

def sample_graph():
    # Generate sample graph
    g = nx.Graph()

    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1)
    g.add_edge(2, 3, weight=1)
    g.add_edge(0, 4, weight=1)
    g.add_edge(1, 4, weight=1)
    g.add_edge(3, 4, weight=1)
    g.add_edge(1, 5, weight=1)
    g.add_edge(2, 5, weight=1)
    g.add_edge(3, 5, weight=1)
    return g
