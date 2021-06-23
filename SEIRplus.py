#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#this model will include vaccination, birth/death rates, imports, contact rates and immunity percentage

#---SEIR Code---#
icdict = {'s': 0.999, 'ex': 0, 'i': 0.001, 'r': 0} 
pardict = {'beta': 2.4/14, 'D': 10, 'L': 4.7, 'N': 1, 'rho': 0, 'mu': 0, 'delta': 0, 'C': 1, 'p': 0}

ds_rhs = '(-beta * C * (i/N) * s) - ((rho * p + mu) * s) + (mu * N)' #mu * (s/N) brings in a fraction of mu, so that it is total removed from all sections
de_rhs = '(beta * C * (i/N) * s) + delta - ((1/L) * ex) - (mu * ex)' #importing cases from outside country, disease still dies because everyone infected
di_rhs = '((1/L) * ex) - ((1/D) * i) - (mu * i)'
dr_rhs = '((1/D) * i) + (rho * p * s) - (mu * r)' #multiply by p as this is the immunity rate, if 45% immunity, only 45% of people are vaccinated to recovered stage

vardict = {'s': ds_rhs, 'ex': de_rhs, 'i': di_rhs, 'r': dr_rhs} 

DSargs = dst.args() 
DSargs.name = 'SEIR'
DSargs.ics = icdict 
DSargs.pars = pardict 
DSargs.tdata = [0, 10] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'beta': 3/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 40/7}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': 4900000, 'ex': 0, 'i': 0, 'r': 0},
       tdata=[0, 10])

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
Reff = s_frac * DS.pars['beta'] * DS.pars['D']

subtotal_cases = np.add(np.add(pts['i'], pts['ex']), pts['r'])
#total_cases = np.add(subtotal_cases, pts['r']) #incliuding exposed in total cases
#total_cases = np.add(pts['i'], pts['r']) #adding current cases to previous cases (assuming no vaccination)

s_1 = pts['s'][-1]
ex_1 = pts['ex'][-1]
i_1 = pts['i'][-1]
r_1 = pts['r'][-1]

#set again to redo
DS.set(pars={'beta': 3/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 150/7, 'C': 1.2}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_1, 'ex': ex_1, 'i': i_1, 'r': r_1},
       tdata=[10, 20])

traj1 = DS.compute('demo')
pts1 = traj1.sample()

subtotal_cases1 = np.add(np.add(pts1['i'], pts1['ex']), pts1['r'])

s_2 = pts1['s'][-1]
ex_2 = pts1['ex'][-1]
i_2 = pts1['i'][-1]
r_2 = pts1['r'][-1]

#set again to redo
DS.set(pars={'beta': 3/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 100/7, 'C': 1}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_2, 'ex': ex_2, 'i': i_2, 'r': r_2},
       tdata=[20, 30])

traj2 = DS.compute('demo')
pts2 = traj2.sample()
#total imported cases by now is 414
subtotal_cases2 = np.add(np.add(pts2['i'], pts2['ex']), pts2['r'])

s_3 = pts2['s'][-1]
ex_3 = pts2['ex'][-1]
i_3 = pts2['i'][-1]
r_3 = pts2['r'][-1]

#set again to redo
DS.set(pars={'beta': 3/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 20/7, 'C': 0.2}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_3, 'ex': ex_3, 'i': i_3, 'r': r_3},
       tdata=[30, 90])

traj3 = DS.compute('demo')
pts3 = traj3.sample()
#total imported cases by now is 414
subtotal_cases3 = np.add(np.add(pts3['i'], pts3['ex']), pts3['r'])
total_cases = np.concatenate((subtotal_cases, subtotal_cases1, subtotal_cases2, subtotal_cases3))
total_time = np.concatenate((pts['t'], pts1['t'], pts2['t'], pts3['t']))

#---Plotting---#
plot1 = plt.figure(1)
#plt.axline((0, 0), (250, 0), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5) #(0, (5, 10)) is a custom dashed line
#plt.axline((0, 4900000), (250, 4900000), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5)
#plt.plot(pts['t'], pts['s'], label='S')
#plt.plot(pts['t'], pts['ex'], label='E')
#plt.plot(pts['t'], pts['i'], label='I')
#plt.plot(pts['t'], pts['r'], label='R')
plt.plot(total_time, total_cases, label='Total Cases')
#plt.xlim(0, 250) #limits max and min x-axis shown
#plt.ylim(0, 5000000)
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(pts['t'], Reff, label='Reff')
plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.xlabel('t')

plt.show()

