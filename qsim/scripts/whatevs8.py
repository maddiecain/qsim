import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from qsim.graph_algorithms.graph import ring_graph, line_graph


ring_odd = [0.010731486819620769, 0.01313224478485521, 0.017118713707233217, 0.02158762971392806, 0.02754774984778865,
            0.03325256957453278, 0.040864182131257754, 0.048092636480067126, 0.05579763588231229, 0.06556633230178542]
ring_even = [0.012950893624301345, 0.05115426937604342, 0.1005415949400835, 0.15522033155622325, 0.21288972878345247,
             0.27244730709453113, 0.33180328100193807, 0.3929663689410151,  0.4667286148763289, 0.5201480279829628]
line_even = [0.010662442042852313, 0.016957472702683322, 0.02374938250983679, 0.03498899357236483, 0.043702391822540175,
             0.054663814324193705, 0.06593339301684226, 0.0750676714021293, 0.08894679704585753, 0.09750844885297337]
line_odd = [0.03639462, 0.08484101277447886, 0.13868272915102697, 0.1955501446428799, 0.2543476347619724,
            0.31447721049371347, 0.37557923456633385, 0.4374203630362864, 0.4998406947299532, 0.5627261695801125]
ring_n_odd = np.arange(3, 2*len(ring_odd)+3, 2)
ring_n_even = np.arange(4, 2*len(ring_even)+4, 2)
line_n_odd = np.arange(3, 2*len(line_odd)+3, 2)
line_n_even = np.arange(4, 2*len(line_even)+4, 2)
slope_ring_odd, int_ring_odd = np.polyfit(ring_n_odd, ring_odd, 1)
slope_ring_even, int_ring_even = np.polyfit(ring_n_even, ring_even, 1)
slope_line_odd, int_line_odd = np.polyfit(line_n_odd, line_odd, 1)
slope_line_even, int_line_even = np.polyfit(line_n_even, line_even, 1)

plt.scatter(ring_n_odd, ring_odd, label='odd ring (deg $n$, slope '+str(np.round(slope_ring_odd, 5))+')')
plt.scatter(ring_n_even, ring_even, label='even ring (deg 2, slope '+str(np.round(slope_ring_even, 5))+')')
plt.scatter(line_n_even, line_even, label=r'even line (deg $\frac{n}{2}+1$, slope '+str(np.round(slope_line_even, 5))+')')
plt.scatter(line_n_odd, line_odd, label='odd line (deg 1, slope '+str(np.round(slope_line_odd, 5))+')')
plt.ylabel('dissipative correction')
plt.xlabel(r'$n$')
plt.legend()
plt.show()


print(np.sum([4.24338406e-01, 3.21500841e-01, 3.14014855e-01,
 3.31627666e-02, 1.79732096e-01, 2.89387654e-02, 2.70738238e-02,
 8.55305488e-02, 6.49241178e-03, 1.98258228e-03]))
# What is going on here?
n_odd_torus = [6, 10, 14]
rates_odd_torus = [0.016271061859353512, 0.025270595180493512, 0.045396454248493326]
n_even_torus = [8, 12, 16, 20, 24]
rates_even_torus = [0.03641161509422947, 0.08381781086704676, 0.15316328607899515, 0.19485277599154976, 0.2910744122739734]
plt.scatter(n_odd_torus, rates_odd_torus)
plt.scatter(n_even_torus, rates_even_torus)
plt.show()


n=np.arange(3, 21, 2)
corr_hybrid = np.array([0.4689649753297138, 0.8050347793424018, 1.1294307138331496, 1.4573466729315718,
                        1.7875857961759019, 2.1194682311689546, 2.452570983712753, 2.7866134155996547,
                        3.1214001544576235])
corr_adiabatic = np.array([0.561569079499595, 0.8924593683713418, 1.227015532135043, 1.5637305502683205,
                           1.9018843015636409, 2.241064353541479, 2.5810110340276338, 2.921550395866910,
                           3.26256048552805])
plt.scatter(n, corr_adiabatic)
plt.scatter(n, corr_hybrid)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(n), np.log(corr_adiabatic))
print(slope, intercept)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(n), np.log(corr_hybrid))
print(slope, intercept)
print(corr_hybrid- corr_adiabatic)
#plt.loglog()
plt.show()



n = np.array([5, 7, 9, 11, 13, 15, 17])
m = np.array([ 2.90757773, 4.1770385, 5.21697679, 6.0971909, 6.8658034, 7.53113982, 8.12875574])
b = np.array([-10.83804585, -17.43937025, -25.52863279, -34.72000245, -44.8200431, -55.62969905])
b_shifted = np.array([-2.11531266, -0.73121624, 0.55625116, 1.86314295, 3.24058072, 4.61941954, 6.03945816])
"""plt.scatter(np.log(n), m)
res = np.polyfit(np.log(n), m, 1)
plt.plot(np.log(n), np.log(n)*res[0]+res[1])
plt.show()"""
"""print(np.diff(b_shifted))
plt.scatter(n, b_shifted)
plt.show()
res = np.polyfit(n, b_shifted, 1)
print(res)
plt.plot()"""
#plt.loglog()
#plt.scatter(n, m)
#plt.scatter(n, b)
#plt.show()


n_odd = np.arange(3, 23, 2)
n_even = np.arange(2, 22, 2)
reit_even = [0.00473168, 0.010662442042852313, 0.016957472702683322, 0.02374938250983679, 0.03498899357236483,
             0.043702391822540175, 0.054663814324193705, 0.06593339301684226, 0.0750676714021293, 0.08894679704585753]
ad_even = [0.07511105, 0.10954203450982979, 0.1644751755518889, 0.21920854799094225, 0.2764753155737447,
           0.3319969967490054, 0.38900249145598326, 0.44476043515277974, 0.5022977527828284, .56]
hybrid_even=[0, 0.006262531818490569, 0.0030127868957918655, 0.009845582637052938, 0.0176493922164758,
             0.02287351950202856, 0.030598243345535073, 0.03609680951698951]

hybrid = [0.03635259323994149, 0.08330018896954504, 0.12834430570068409, 0.17091713508988504, 0.2142658402270388,
          0.2581563204648457, 0.30244525320333504, 0.3470386280613774, 0.3918715594625951, 0.4368974642771301]
reit = [-0.03639462, -0.08484101277447886, -0.13868272915102697, -0.1955501446428799, -0.2543476347619724,
        -0.31447721049371347, -0.37557923456633385, -0.4374203630362864, -0.4998406947299532, -0.5627261695801125]
reit = -np.array(reit)
ad = [-0.19392685, -0.32423511678256456, -0.46019132123765516, -0.5992231418366185, -0.74018775581,
      -0.8824810126722228, -1.0257427690715848, -1.1697409098327542, -1.3143167939305687, -1.4593572148575429]
ad = -np.array(ad)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n_odd, ad)
plt.plot(n_odd, n_odd*slope+intercept, color='red', label='odd experiment, slope = '+str(np.round(slope.real, decimals=4)))
plt.scatter(n_odd, ad, color='red')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n_even, ad_even)
plt.plot(n_even, n_even*slope+intercept, color='salmon', label='even experiment, slope = '+str(np.around(slope.real, decimals=4)))
plt.scatter(n_even, ad_even, color='salmon')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n_odd, reit)
plt.plot(n_odd, n_odd*slope+intercept, color='green', label='odd STIRAP, slope = '+str(np.around(slope.real, decimals=4)))
plt.scatter(n_odd, reit, color='green')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n_even, reit_even)
plt.plot(n_even, n_even*slope+intercept, color='lightgreen', label='even STIRAP, slope = '+str(np.around(slope.real, decimals=4)))
plt.scatter(n_even, reit_even, color='lightgreen')


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n_odd[:len(hybrid)], hybrid)
plt.plot(n_odd, n_odd*slope+intercept, color='blue', label='odd hybrid, slope = '+str(np.around(slope.real, decimals=4)))
plt.scatter(n_odd[:len(hybrid)], hybrid, color='blue')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(n_even[:len(hybrid_even)], hybrid_even)
plt.plot(n_even, n_even*slope+intercept, color='lightblue', label='even hybrid, slope = '+str(np.around(slope.real, decimals=4)))
plt.scatter(n_even[:len(hybrid_even)], hybrid_even, color='lightblue')


plt.xticks(np.arange(2, 22))
plt.xlabel(r'$n$')
plt.ylabel(r'constant prefactor on order $\alpha$ reduction in fidelity')
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