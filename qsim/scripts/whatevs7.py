import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
betas = np.arange(15, 30, 1)
alphas = np.arange(0, 80, 5)
performances = np.zeros((len(alphas), len(betas)))
for i in range(len(betas)*len(alphas)):
    print(os.getcwd())
    try:
        f = open(os.path.join(os.getcwd(), 'n3', 'n3_hybrid_{}.out'.format(str(i))), 'r')
        for line in f:
            line = line.split(' ')
            if len(line) == 4:
                mode, alpha, beta, performance = line
                beta = float(beta)
                alpha = float(alpha)
                performance = float(performance)
                alpha_index = np.argwhere(np.isclose(alphas, alpha))
                in_list = True
                if len(alpha_index) != 0:
                    alpha_index = alpha_index[0][0]
                else:
                    print('extra data', mode, alpha_index, beta, performance)
                    in_list = False
                beta_index = np.argwhere(np.isclose(betas, beta))
                if len(beta_index) != 0:
                    beta_index = beta_index[0][0]
                else:
                    print('extra data', mode, alpha_index, beta, performance)
                    in_list = False
                if in_list:
                    performances[alpha_index, beta_index] = performance
    except:
        pass

#max_performance = np.max(performances, axis=1)
#optimal_beta = [betas[np.argwhere(performances == max_performance[i])[0][1]] for i in range(len(max_performance))]
plt.imshow(np.flip(performances, axis=(0)), cmap='rainbow', vmin=0, vmax=1, interpolation=None,
               extent=[min(betas), max(betas), min(alphas), max(alphas)], aspect='auto')
plt.colorbar()
plt.ylabel(r'$\frac{\Omega^2\gamma}{|\Delta|^2}$')
plt.xlabel(r'$\frac{\Omega^2\delta_e}{|\Delta|}$')
plt.show()