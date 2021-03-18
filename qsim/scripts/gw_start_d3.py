import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def int_to_nary(n, size=None, base=2, pad_with=0):
    """Converts an integer :math:`n` to a size-:math:`\\log_2(n)` binary array.

    :param n: Integer to convert
    :type n: int
    :return: Binary array representing :math:`n`.
    """
    assert n >= 0
    if base == 2:
        # Use pad_with faster implementation, if possible
        if pad_with == 0:
            nary_repr = np.array(list(np.binary_repr(n, width=size)), dtype=int)
        else:
            nary_repr = np.array(list(np.binary_repr(n)), dtype=int)
            if not size is None:
                # Pad with padwith
                nary_repr = np.concatenate([np.ones(size - len(nary_repr), dtype=int) * pad_with, nary_repr])
    else:
        nary_repr = np.array(list(np.base_repr(n, base=base)), dtype=int)
        nary_repr = np.concatenate([np.ones(size - len(nary_repr), dtype=int) * pad_with, nary_repr])
    return nary_repr


def generate_SDP_graph(d, epsilon, visualize=False):
    diff = d - int(d * epsilon)
    nodes = np.zeros((2 ** d, d))
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(2 ** d))
    for i in range(2 ** d):
        binary = int_to_nary(i, size=d)
        nodes[i, :] = binary
    for i in range(2 ** d):
        for j in range(2 ** d):
            binary_i = int_to_nary(i, size=d)
            binary_j = int_to_nary(j, size=d)
            if np.sum(np.abs(binary_i - binary_j)) == diff:
                graph.add_edge(i, j)
    if visualize:
        nx.draw(graph, with_labels=True)
        plt.show()
    return graph


def generate_SDP_initial_state():
    sdp_initial_state = np.zeros(2 ** n)
    k = 0
    for i in range(2 ** n):
        binary = int_to_nary(i, size=n)
        hw = np.sum(binary[[0, 3, 5, 6]])
        if np.allclose(1 - binary[[0, 3, 5, 6]], binary[[7, 4, 2, 1]]):
            if not (np.allclose(np.array([1, 0, 0, 1, 0, 1, 1, 0]), binary) or np.allclose(
                    np.array([0, 1, 1, 0, 1, 0, 0, 1]), binary)):
                if hw % 2 == 0:
                    k += 1
                    # Print the cut value
                    sdp_initial_state[i] = np.sqrt(0.108173)
                else:
                    sdp_initial_state[i] = np.sqrt(0.0438699)
    sdp_initial_state = sdp_initial_state / np.linalg.norm(sdp_initial_state)
    return sdp_initial_state[np.newaxis, :].T


d = 3
n = 2 ** d
epsilon = 1 / 3
graph = generate_SDP_graph(d, epsilon, visualize=False)
print(generate_SDP_initial_state())
