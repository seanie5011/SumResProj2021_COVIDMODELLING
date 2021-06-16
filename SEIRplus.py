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
DSargs.tdata = [0, 20] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'beta': 3/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 150/7}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': 4900000, 'ex': 0, 'i': 0, 'r': 0},
       tdata=[0, 17])

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

total_cases = np.add(pts['i'], pts['r']) #adding current cases to previous cases (assuming no vaccination)

#print(pts['s'][-1])
#print(pts['ex'][-1])
#print(pts['i'][-1])
#print(pts['r'][-1])

#---Plotting---#
plot1 = plt.figure(1)
#plt.axline((0, 0), (250, 0), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5) #(0, (5, 10)) is a custom dashed line
#plt.axline((0, 4900000), (250, 4900000), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5)
#plt.plot(pts['t'], pts['s'], label='S')
plt.plot(pts['t'], pts['ex'], label='E')
plt.plot(pts['t'], pts['i'], label='I')
#plt.plot(pts['t'], pts['r'], label='R')
#plt.plot(pts['t'], total_cases, label='Total Cases')
#plt.xlim(0, max_time) #limits max and min x-axis shown
#plt.ylim(0, 5000000)
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(pts['t'], Reff, label='Reff')
plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.xlabel('t')

plt.show()

