import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

n = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
reit = [-0.00473168,-0.03639462+0.j, -0.08484101277447886+0j, -0.13868272915102697+0j, -0.1955501446428799+0j, -0.2543476347619724+0j,
        -0.31447721049371347+0j, -0.37557923456633385, -0.4374203630362864, -0.4998406947299532, -0.5627261695801125]
ad = [-0.07511105,-0.19392685+0.j, -0.32423511678256456+0j, -0.46019132123765516+0j, -0.5992231418366185+0j, -0.74018775581,
      -0.8824810126722228, -1.0257427690715848, -1.1697409098327542, -1.3143167939305687, -1.4593572148575429]
reit = np.array(reit)
ad = np.array(ad)
n = np.array(n)
print(np.diff(reit), np.diff(ad))
plt.scatter(n[1:], -1*reit[1:], color='green', label='reit')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n[1:], -1*reit[1:])
print(slope, intercept)

plt.plot(n[1:], n[1:]*slope+intercept, color='green', label='slope= '+str(np.around(slope.real, decimals=4)))
plt.scatter(n[1:], -1*ad[1:], color='red', label='adiabatic')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n[1:], -1*ad[1:])
print(slope, intercept)
plt.xlabel(r'$n$')
plt.ylabel(r'constant prefactor on order $\alpha$ reduction in fidelity')
plt.plot(n[1:], n[1:]*slope+intercept, color='red', label='slope= '+str(np.round(slope.real, decimals=4)))
plt.legend()
plt.show()
ratios = np.array(ad).real/np.array(reit).real#np.diff(ad)/np.diff(reit)[:len(np.diff(ad))]#
print(ratios)
print(len(ratios), len(n))
plt.scatter(n[1:], ratios[1:], color='red')
plt.ylabel(r'ratio of adiabatic to reit first order corrections in $\alpha$')
plt.xlabel(r'$n$')
plt.ylim(0)
plt.hlines(1, 2, 21, linestyles=':')
plt.show()
#betas = np.arange(30, 46, 1)
#xis = np.arange(1000, 11000, 200)
fidelities = [0.9994142841297375, 0.9996184189167889, 0.999808522111377, 0.999909790044486, 0.9999466960059833, 0.9999579633451534, 0.9999665560744818, 0.9999803804848229, 0.999990837007162, 0.9999884707847753, 0.999979458065795, 0.999977974922668, 0.999984489220821, 0.9999895548121703, 0.9999901197126595, 0.9999896513060784, 0.9999896448955592, 0.9999884453686896, 0.9999873634232801, 0.9999881078255612, 0.9999887524285559, 0.9999878188615101, 0.9999868162549896, 0.9999872171724886, 0.9999876812688213, 0.9999871792494439, 0.9999866948899425, 0.9999868222086454, 0.9999866569388456, 0.9999860278539647, 0.9999856849996408, 0.9999859161898706, 0.9999857258960092, 0.9999850692538703, 0.999984787766854]
#fidelities = [0.9994112678826861, 0.9996150928992649, 0.99980487069132, 0.99990580040772, 0.9999423551089264, 0.9999532571666601, 0.9999614699404038, 0.9999748995118087, 0.9999849466254678, 0.9999821567388218, 0.9999727057423962, 0.9999707690244585, 0.9999768143889258, 0.9999813961895072, 0.9999814627069625, 0.9999804811283517, 0.9999799466642043, 0.9999782042987844, 0.9999765646218391, 0.9999767363578642, 0.999976793426035, 0.9999752575880818, 0.9999736378361903, 0.9999734066359159, 0.9999732237514817, 0.9999720599085287, 0.9999708988980714, 0.9999703346381681, 0.9999694629386124, 0.9999681126531752, 0.9999670336519122, 0.999966513777341, 0.9999655576217117, 0.9999641202846727]
#fidelities = [0.9991096965531964, 0.9992825567051411, 0.9994398083387246, 0.9995069319829729, 0.9995083779992414, 0.9994827713375162, 0.9994530094967554, 0.9994269803951672, 0.9993961137284684, 0.9993509965537664, 0.9992977371367195, 0.9992504890955624, 0.9992096702504208, 0.9991659239869208, 0.9991162005778238, 0.9990639453198373, 0.9990106945859732, 0.9989547085305445]
#fidelities = [0.9985834397792408, 0.980495772382922, 0.9433886090040569, 0.9274290096344515, 0.9357952981071747, 0.9545719074548834, 0.9718971604528459, 0.9836918864427806, 0.9907267359747188, 0.9947883770546668, 0.997025361287564, 0.9979279825849712, 0.9978019653958113, 0.9971668308864052, 0.9965944340290933, 0.996328494024964, 0.9962431120802397, 0.9961261485531194, 0.9958953239352072, 0.995582787971355, 0.9952255173447488, 0.9948572117942333, 0.9944858263735679, 0.9940889711432492, 0.9936581460048618, 0.9931914792247317, 0.9927265142458879, 0.9922680997062199, 0.9917976307347733, 0.991303142071523, 0.9907836421667267, 0.9902596933825274, 0.9897138244401155, 0.9891519753418785, 0.9885863857297179]
#fidelities = fidelities[5:]
#fidelities = [0.9638730961417439, 0.9691405469796317, 0.9736646359269011, 0.9774501240880478, 0.9805813351361425, 0.9831776989367331, 0.9853656844120322, 0.9872475732212352, 0.9888729109049907, 0.990249938869167, 0.9913849450841657, 0.9922995600609139, 0.9930228342100504, 0.9935872260747981, 0.9940342218721386, 0.9944044713424113, 0.9947217444135366, 0.9949900510067147, 0.995204302207876, 0.995358966907785, 0.9954507189283089, 0.9954841272119285, 0.9954733005112927, 0.9954345285610643]


print(len(fidelities))
betas = np.arange(20, 80, 1, dtype=np.float64)[:len(fidelities)]
print(len(betas))
x = np.log10(betas)
y = np.log10(-np.log10(np.array(fidelities)))
plt.scatter(x, y)
import scipy.stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[-10:], y[-10:])
print(slope, intercept)
plt.plot(x, x*slope+intercept, label=str(slope))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[4:10], y[4:10])
print(slope, intercept)
plt.plot(x, x*slope+intercept, label=str(slope))
plt.legend()
plt.show()
betas = np.arange(20, 40, 1)
xis = np.arange(250, 5000, 200)
performances = np.zeros((len(xis), len(betas)))
for i in range(len(betas)*len(xis)):
    try:
        #f = open(os.path.join(os.getcwd(), 'n3', 'n3_adiabatic_{}.out'.format(str(i))), 'r')
        f = open(os.path.join(os.getcwd(), 'n3_adiabatic.out'), 'r')
        for line in f:
            #print(line)
            line = line.split(' ')
            #print(line)
            if len(line) == 3:
                beta, xi, performance = line
                beta = float(beta)
                print(beta)
                xi = float(xi)
                #print(xi, beta)
                #print(beta, xi)
                performance = float(performance)
                xi_index = np.argwhere(np.isclose(xis, xi))
                in_list = True
                if len(xi_index) != 0:
                    xi_index = xi_index[0][0]
                else:
                    print('extra data', xi_index, beta, performance)
                    in_list = False
                beta_index = np.argwhere(np.isclose(betas, beta))
                if len(beta_index) != 0:
                    beta_index = beta_index[0][0]
                else:
                    print('extra data', xi_index, beta, performance)
                    in_list = False
                if in_list:
                    performances[xi_index, beta_index] = performance
            elif len(line) == 4:
                mode, xi, beta, performance = line
                beta = float(beta)
                print(beta)
                xi = float(xi)
                #print(xi, beta)
                #print(beta, xi)
                performance = float(performance)
                xi_index = np.argwhere(np.isclose(xis, xi))
                in_list = True
                if len(xi_index) != 0:
                    xi_index = xi_index[0][0]
                else:
                    print('extra data', xi_index, beta, performance)
                    in_list = False
                beta_index = np.argwhere(np.isclose(betas, beta))
                if len(beta_index) != 0:
                    beta_index = beta_index[0][0]
                else:
                    print('extra data', xi_index, beta, performance)
                    in_list = False
                if in_list:
                    performances[xi_index, beta_index] = performance
    except:
        print('failed to find file')
        #print(i)
print(performances.shape, performances[-1])
i = -1
plt.scatter(np.log(betas), np.log(-np.log(performances[i])))
print(np.log(-np.log(performances[i])))
x = np.log(betas)[-6:]
y = np.log(-np.log(performances[i]))[-6:]
index = ~(np.isinf(x) | np.isinf(y))
print(index)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[index], y[index])
print(slope, intercept)
plt.plot(x, x*slope+intercept, label=str(slope))
plt.legend()

plt.show()
plt.clf()
max_performance = np.max(performances, axis=1)
optimal_beta = [betas[np.argwhere(performances == max_performance[i])[0][1]] for i in range(len(max_performance))]
plt.scatter(np.log(xis), np.log(-np.log(max_performance)))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(xis)[0:14], np.log(-np.log(max_performance))[0:14])
#plt.plot(np.log(xis)[0:14], np.log(xis)[0:14]*slope+intercept, label=str(slope))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(xis)[:], np.log(-np.log(max_performance))[:])
plt.plot(np.log(xis)[:], np.log(xis)[:]*slope+intercept, label=str(slope))
plt.legend()
plt.show()
plt.clf()
plt.scatter(np.log(xis), np.log(optimal_beta))
#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(xis)[10:], np.log(optimal_beta)[10:])
#plt.plot(np.log(xis)[10:], np.log(xis)[10:]*slope+intercept, label=str(slope))
plt.legend()
plt.show()