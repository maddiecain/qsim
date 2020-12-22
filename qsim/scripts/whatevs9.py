import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

times = np.arange(10, 200, 1)
performances = np.zeros(len(times))
for i in range(len(times)):
    try:
        f = open(os.path.join(os.getcwd(), 'n2/n2_hybrid_{}.out'.format(str(i))), 'r')
        for line in f:
            line = line.split(' ')
            #print(line)
            if len(line) == 5:
                t, o, d, g, performance = line
                t = float(t)
                o = float(o)
                d = float(d)
                g = float(g)
                performance = performance[1:-2]
                performance = float(performance)
                t_index = np.argwhere(np.isclose(times, t))
                in_list = True
                if len(t_index) != 0:
                    t_index = t_index[0][0]
                else:
                    print('extra data', t, performance)
                    in_list = False

                if in_list:
                    performances[t_index] = performance

    except:
        print('failed to find file')

# cut at the first zero
where_zero_first = np.argwhere(performances == 0)[0,0]
performances = performances[0:where_zero_first]
times = times[0:where_zero_first]
is_local_maxima = np.r_[True, performances[1:] < performances[:-1]] & np.r_[performances[:-1] < performances[1:], True]

local_maxima = np.log(-np.log(performances[is_local_maxima]))
times_local_maxima = np.log(times[is_local_maxima])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(times_local_maxima[0:4], local_maxima[0:4])
plt.plot(times_local_maxima, times_local_maxima*slope+intercept, label=str(slope))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(times_local_maxima[-4:], local_maxima[-4:])
plt.plot(times_local_maxima, times_local_maxima*slope+intercept, label=str(slope))
plt.plot(np.log(times), np.log(-np.log(performances)))
plt.legend()
plt.show()
