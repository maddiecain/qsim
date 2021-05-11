import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

f = open('adjacency_1.txt')
graph = np.zeros((179,179))
i = 0
for line in f:
    line = line.split(' ')
    graph[i,...] = np.array(line)
    i += 1



plt.imshow(graph)
plt.show()