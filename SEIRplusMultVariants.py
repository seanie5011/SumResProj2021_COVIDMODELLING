#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#this model will include multiple variants, vaccination, birth/death rates, imports, contact rates and immunity percentage

#---SEIR Code---#
icdict = {'s': 0.999, 'ex1': 0, 'i1': 0.0003, 'ex2': 0, 'i2': 0.0003, 'ex3': 0, 'i3': 0.0003, 'r': 0} 
pardict = {'beta1': 2.4/14, 'beta2': 4/14, 'beta3': 5/14, 'D': 10, 'L': 4.7, 'N': 1, 'rho': 0, 'mu': 0, 'delta1': 0, 'delta2': 0, 'delta3': 0, 'C': 1, 'p': 0}

ds_rhs = '(-beta1 * C * (i1/N) * s) - (beta2 * C * (i2/N) * s) - (beta3 * C * (i3/N) * s) - ((rho * p + mu) * s) + (mu * N)' #mu * (s/N) brings in a fraction of mu, so that it is total removed from all sections
de1_rhs = '(beta1 * C * (i1/N) * s) + delta1 - ((1/L) * ex1) - (mu * ex1)' #importing cases from outside country, disease still dies because everyone infected
di1_rhs = '((1/L) * ex1) - ((1/D) * i1) - (mu * i1)'
de2_rhs = '(beta2 * C * (i2/N) * s) + delta2 - ((1/L) * ex2) - (mu * ex2)' #importing cases from outside country, disease still dies because everyone infected
di2_rhs = '((1/L) * ex2) - ((1/D) * i2) - (mu * i2)'
de3_rhs = '(beta3 * C * (i3/N) * s) + delta3 - ((1/L) * ex3) - (mu * ex3)' #importing cases from outside country, disease still dies because everyone infected
di3_rhs = '((1/L) * ex3) - ((1/D) * i3) - (mu * i3)'
dr_rhs = '((1/D) * (i1 + i2 + i3)) + (rho * p * s) - (mu * r)' #multiply by p as this is the immunity rate, if 45% immunity, only 45% of people are vaccinated to recovered stage

vardict = {'s': ds_rhs, 'ex1': de1_rhs, 'i1': di1_rhs, 'ex2': de2_rhs, 'i2': di2_rhs, 'ex3': de3_rhs, 'i3': di3_rhs, 'r': dr_rhs} 

DSargs = dst.args() 
DSargs.name = 'SEIR'
DSargs.ics = icdict 
DSargs.pars = pardict 
DSargs.tdata = [0, 20] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'beta1': 3.32/10, 'beta2': 4/10, 'beta3': 5/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta1': 200/7, 'delta2': 100/7, 'delta3': 50/7}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': 4900000, 'ex1': 0, 'i1': 0, 'ex2': 0, 'i2': 0, 'ex3': 0, 'i3': 0, 'r': 0},
       tdata=[0, 100])

traj = DS.compute('demo')
pts = traj.sample()

#for j in range(len(pts['s'])): #ensures cant go to negative values, all pts have same length
#    if pts['s'][j] < 0:
#        pts['s'][j] = 0
#    if pts['r'][j] > DS.pars['N']:
#        pts['r'][j] = DS.pars['N']

#j = 0
#run = true
#while run == true:
#    if pts['s'][j+1] < 0 or j+1 >= len(pts['s']):
#        max_time = pts['t'][j+1]
#        run = false
#    j+=1

s_frac = pts['s']/DS.pars['N']
Reff1 = s_frac * DS.pars['beta1'] * DS.pars['D']
Reff2 = s_frac * DS.pars['beta2'] * DS.pars['D']
Reff3 = s_frac * DS.pars['beta3'] * DS.pars['D']

#np.savetxt('test1.txt', pts['i1'][1:-1:100], fmt='%10f', delimiter=',')
#np.savetxt('test2.txt', pts['i2'][1:-1:100], fmt='%10f', delimiter=',')

subtotal_cases = np.add(pts['i1'], pts['i2']) #adding current cases to previous cases (assuming no vaccination)
subtotal_cases1 = np.add(pts['i3'], pts['r'])
total_cases = np.add(subtotal_cases, subtotal_cases1)

#print(pts['s'][-1])
#print(pts['ex'][-1])
#print(pts['i'][-1])
#print(pts['r'][-1])

#---Plotting---#
plot1 = plt.figure(1)
plt.axline((0, 0), (250, 0), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5) #(0, (5, 10)) is a custom dashed line
plt.axline((0, 4900000), (250, 4900000), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5)
plt.plot(pts['t'], pts['s'], label='S')
plt.plot(pts['t'], pts['ex1'], label='E1')
plt.plot(pts['t'], pts['i1'], label='I1')
plt.plot(pts['t'], pts['ex2'], label='E2')
plt.plot(pts['t'], pts['i2'], label='I2')
plt.plot(pts['t'], pts['ex3'], label='E3')
plt.plot(pts['t'], pts['i3'], label='I3')
plt.plot(pts['t'], pts['r'], label='R')
#plt.plot(pts['t'], total_cases, label='Total Cases')
plt.xlim(0, 100) #limits max and min x-axis shown
#plt.ylim(0, 5000000)
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(pts['t'], Reff1, label='Reff1')
plt.plot(pts['t'], Reff2, label='Reff2')
plt.plot(pts['t'], Reff3, label='Reff3')
plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.xlabel('t')

plt.show()
