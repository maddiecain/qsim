import numpy as np
import matplotlib.pyplot as plt
import os
import re

times = np.arange(0, 1, .01)
i = 0
for j in range(len(times)):
    try:
        f = open(os.path.join(os.getcwd(), 'sk_avoided_crossing_n=18_p=3', 'sk_18_{}.out'.format(str(j))), 'r')

        for line in f:
            line = re.split(',', line)
            for k in range(len(line)):
                line[k] = line[k].replace('[', '')
                line[k] = line[k].replace(']', '')
                line[k] = float(line[k])
            eigvals = line
            for eigval in eigvals:
                if times[i] != 0 and eigval != -14:
                    plt.scatter(times[i], eigval, color='blue', s=2)
    except Exception as e:
        print('cannot open file', i, e)
    i += 1
plt.ylabel('energy')
plt.xlabel(r't/T')
plt.show()