import networkx as nx
from qsim.evolution import hamiltonian
from qsim.graph_algorithms import qaoa
import numpy as np
import matplotlib.pyplot as plt
from qsim.graph_algorithms.graph import Graph
from qsim import tools
from qsim.codes.quantum_state import State

def generate_SDP_graph(d, epsilon, visualize=False, le=False):
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(2**d))
    for i in range(2**d):
        for j in range(2**d):
            binary_i = 2*(tools.int_to_nary(i, size=d)-1/2)/np.sqrt(d)
            binary_j = 2*(tools.int_to_nary(j, size=d)-1/2)/np.sqrt(d)
            if le:
                if (1-np.dot(binary_i, binary_j))/2 > 1-epsilon+1e-5:
                    graph.add_edge(i, j, weight=2)
            else:
                if np.isclose((1-np.dot(binary_i, binary_j))/2, 1-epsilon):
                    graph.add_edge(i, j, weight=2)
    if visualize:
        nx.draw(graph, with_labels=True)
        plt.show()
    return Graph(graph)

def generate_SDP_initial_state():
    sdp_initial_state = np.zeros(2**graph.n)
    k = 0
    for i in range(2**graph.n):
        binary = tools.int_to_nary(i, size=graph.n)
        hw = np.sum(binary[[0, 3, 5, 6]])
        if np.allclose(1-binary[[0, 3, 5, 6]], binary[[7, 4, 2, 1]]):
            if not (np.allclose(np.array([1, 0, 0, 1, 0, 1, 1, 0]), binary) or np.allclose(np.array([0, 1, 1, 0, 1, 0, 0, 1]), binary)):
                if hw % 2 == 0:
                    k += 1
                    # Print the cut value
                    sdp_initial_state[i] = np.sqrt(0.108173)
                else:
                    sdp_initial_state[i] = np.sqrt(0.0438699)
    sdp_initial_state = sdp_initial_state/np.linalg.norm(sdp_initial_state)
    sdp_initial_state = State(sdp_initial_state[np.newaxis, :].T)
    return sdp_initial_state


def generate_SDP_output(d, num=100):
    state = np.zeros(2**(2**d))
    node_labels = np.zeros((2**d, d))
    for i in range(2 ** d):
        node_labels[i,:] = 2*(tools.int_to_nary(i, size=d)-1/2)/np.sqrt(d)

    def gaussian(vec):
        return np.exp(-np.abs(np.linalg.norm(vec))**2/2)/(2*np.pi)**(d/2)

    from scipy.special import erfcinv
    # Fraction of gaussian we are ok leaving out of the integration from the long range tails
    frac = 1e-5/(2**(2**d))
    sigma = erfcinv(frac)
    xmin = -sigma
    xmax = sigma
    xvals = np.linspace(xmin, xmax, num)
    dx = (xvals[1]-xvals[0])**d
    from itertools import product
    coordinates = product(xvals, repeat=d)
    total = num**d
    step = int(total/20)
    l = 0
    j = 0
    for rounding_vector in coordinates:
        if j == l*step:
            print(str(l*5)+'%')
            l += 1
        rounding_vector = np.array(rounding_vector)+np.random.normal(size=d, scale=.001)
        assignment = np.zeros(2 ** d)
        for i in range(2 ** d):
            #print(np.dot(rounding_vector, node_labels[i]), rounding_vector, node_labels[i])
            if np.dot(rounding_vector, node_labels[i]) < 0:
                assignment[i] = 1
            else:
                assignment[i] = 0
        state[tools.nary_to_int(assignment)] = state[tools.nary_to_int(assignment)] + gaussian(rounding_vector)*dx/2
        state[tools.nary_to_int(np.flip(assignment))] = state[tools.nary_to_int(np.flip(assignment))] + gaussian(rounding_vector)*dx/2

        j += 1
    # Even though it should already be normalized, our integral under
    state = state/np.sum(state)
    return state


def generate_legal_SDP_output(d, num=100):
    state = np.zeros(2**(2**d))
    node_labels = np.zeros((2**d, d))
    for i in range(2 ** d):
        node_labels[i,:] = 2*(tools.int_to_nary(i, size=d)-1/2)/np.sqrt(d)

    def gaussian(vec):
        return np.exp(-np.abs(np.linalg.norm(vec))**2/2)/(2*np.pi)**(d/2)

    from scipy.special import erfcinv
    # Fraction of gaussian we are ok leaving out of the integration from the long range tails
    frac = 1e-5/(2**(2**d))
    sigma = erfcinv(frac)
    xmin = -sigma
    xmax = sigma
    xvals = np.linspace(xmin, xmax, num)
    dx = (xvals[1]-xvals[0])**d
    from itertools import product
    coordinates = product(xvals, repeat=d)
    total = num**d
    step = int(total/20)
    l = 0
    j = 0
    for rounding_vector in coordinates:
        if j == l*step:
            print(str(l*5)+'%')
            l += 1
        rounding_vector = np.array(rounding_vector)+np.random.normal(size=d, scale=.001)
        assignment = np.zeros(2 ** d)
        for i in range(2 ** d):
            #print(np.dot(rounding_vector, node_labels[i]), rounding_vector, node_labels[i])
            if np.dot(rounding_vector, node_labels[i]) < 0:
                assignment[i] = 1
            else:
                assignment[i] = 0
        state[tools.nary_to_int(assignment)] = 1
        state[tools.nary_to_int(np.flip(assignment))] = 1

        j += 1
    # Even though it should already be normalized, our integral under
    state = state/np.sum(state)
    return state

#SDP_output = np.sqrt(generate_SDP_output(3, 50))
#print(SDP_output2[np.nonzero(SDP_output2)])
#np.save('d3_SDP_output', SDP_output)
SDP_output=np.load('d3_SDP_output.npy')
#SDP_output= SDP_output*np.random.uniform(.01, .99, size=len(SDP_output))
SDP_output = SDP_output/np.linalg.norm(SDP_output)
np.set_printoptions(threshold=np.inf)
print(SDP_output)
#print(SDP_output)
#SDP_output1 = generate_SDP_output(4, 20)
#print(np.linalg.norm(SDP_output1-SDP_output2))
#print(len(SDP_output[np.nonzero(SDP_output)]))
#print(SDP_output[rows, cols])
SDP_output = State(SDP_output[np.newaxis, :].T)
graph = generate_SDP_graph(4, 1/4, visualize=False)
cost = hamiltonian.HamiltonianMaxCut(graph, cost_function=True)
driver = hamiltonian.HamiltonianDriver()
qaoa = qaoa.SimulateQAOA(graph=graph, hamiltonian=[], cost_hamiltonian=cost)

#SDP_cf = cost.cost_function(SDP_output)
#plt.scatter(-1, cost.cost_function(SDP_output))
vanilla_cf = np.array([42.39230484541338, 48.26437904543421, 53.04760524625608, 58.034293475214454, 61.43733741238711, 62.1439072862153])/64
plt.scatter(np.arange(len(vanilla_cf)), 1-vanilla_cf)
#plt.semilogy()
plt.show()
count = 0
max_result = 0
print(max_result)
for p in range(6, 13):
    hamiltonian = [cost, driver]*p
    qaoa.hamiltonian = hamiltonian
    """params = [1.16874104, 2.90723173, 0.99061969, 3.2870658,  3.00033462, 3.77856328, 1.80869453, 5.2534133, 0.65767639,
              3.65537877, 1.9589525,  1.39462391, 4.14548547, 5.1316928]#,  5.5824009,  2.22808344]
    gammas = np.linspace(0, np.pi*2)
    betas = np.linspace(0, np.pi*2)
    results = np.zeros((len(gammas), len(betas)))
    for g in range(len(gammas)):
        for b in range(len(betas)):
            print(g)
            results[g, b] = qaoa.run(params+[gammas[g], betas[b]], SDP_output)
    plt.imshow(results)
    plt.show()"""
    for i in range(100):
        result = qaoa.find_parameters_minimize(verbose=False)
        if result['f_val'] > max_result:
            max_result = result['f_val']
            print('better', max_result)
        else:
            print('not better', result['f_val'])
        """result = qaoa.find_parameters_minimize(initial_state=SDP_output, verbose=False)
        if np.isclose(SDP_cf, result['f_val']) or SDP_cf<result['f_val']:
            if SDP_cf< result['f_val']:
                print('found a winner', result)
            count+=1
            print(result['f_val']-SDP_cf)
            print(result['params'])
            print(count)
        plt.scatter(i, result['f_val'])"""
    print(p, max_result)

plt.show()
ar_warm_start = np.array([0.76020043245, 0.8997364323093671, 0.9649373721434963, 0.9724929666329931, 0.972816710863865])
ar_vanilla = np.array([.75, 0.9243790248128766, 1])
plt.scatter(np.arange(len(ar_warm_start)), 1-ar_warm_start)
#plt.scatter(np.arange(len(ar_vanilla)), 1-ar_vanilla)
plt.loglog()
plt.show()


