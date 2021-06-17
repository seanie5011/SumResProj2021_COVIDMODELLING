#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#This will include a second variant, the UK (alpha) variant

#---SIR Code---#
icdict = {'s': 0.999, 'ex1': 0, 'i1': 0.0003, 'ex2': 0, 'i2': 0.0003, 'ex3': 0, 'i3': 0.0003, 'r': 0} 
pardict = {'beta1': 2.4/14, 'beta2': 3.75/14, 'beta3': 5/14, 'D': 14, 'L': 4.7, 'N': 1}

ds_rhs = '-beta1 * (i1/N) * s - beta2 * (i2/N) * s - beta3 * (i3/N) * s' #divide by N to ensure it is a fraction
de1_rhs = 'beta1 * (i1/N) * s - ((1/L) * ex1)'
di1_rhs = 'beta1 * (i1/N) * s - ((1/D) * i1)'
de2_rhs = 'beta2 * (i2/N) * s - ((1/L) * ex2)'
di2_rhs = 'beta2 * (i2/N) * s - ((1/D) * i2)'
de3_rhs = 'beta3 * (i3/N) * s - ((1/L) * ex3)'
di3_rhs = 'beta3 * (i3/N) * s - ((1/D) * i3)'
dr_rhs = '(1/D) * (i1 + i2 + i3)'

vardict = {'s': ds_rhs, 'ex1': de1_rhs, 'i1': di1_rhs, 'ex2': de2_rhs, 'i2': di2_rhs, 'ex3': de3_rhs, 'i3': di3_rhs, 'r': dr_rhs} 

DSargs = dst.args() 
DSargs.name = 'SIR'
DSargs.ics = icdict 
DSargs.pars = pardict 
DSargs.tdata = [0, 20] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'N': 4900000},
       ics={'s': 4900000 - 2210, 'i1': 2000, 'i2': 200, 'i3': 10}, #original strain starts with 2000, alpha with 1
       tdata = [0, 100])

traj = DS.compute('demo')
pts = traj.sample()

s_frac = pts['s']/DS.pars['N']
Reff1 = s_frac * DS.pars['beta1'] * DS.pars['D']
Reff2 = s_frac * DS.pars['beta2'] * DS.pars['D']
Reff3 = s_frac * DS.pars['beta3'] * DS.pars['D']

subtotal_cases = np.add(pts['i1'], pts['i2']) #adding current cases to previous cases (assuming no vaccination)
subtotal_cases1 = np.add(pts['i3'], pts['r'])
total_cases = np.add(subtotal_cases, subtotal_cases1)

#---Plotting---#
plot1 = plt.figure(1)
#plt.plot(pts['t'], pts['s'], label='S')
#plt.plot(pts['t'], pts['ex1'], label='E1')
#plt.plot(pts['t'], pts['i1'], label='I1')
#plt.plot(pts['t'], pts['ex2'], label='E2')
#plt.plot(pts['t'], pts['i2'], label='I2')
#plt.plot(pts['t'], pts['ex3'], label='E3')
#plt.plot(pts['t'], pts['i3'], label='I3')
#plt.plot(pts['t'], pts['r'], label='R')
plt.plot(pts['t'], total_cases, label='Total Cases')
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(pts['t'], Reff1, label='Reff1')
plt.plot(pts['t'], Reff2, label='Reff2')
plt.plot(pts['t'], Reff3, label='Reff3')
plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--')
plt.legend()
plt.xlabel('t')

plt.show()

#to show on one plot all cases, can concatenate different sections instead of doing all at once
