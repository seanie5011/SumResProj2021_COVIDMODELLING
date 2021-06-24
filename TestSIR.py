#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#---SIR Code---#
icdict = {'s': 0.999, 'i': 0.001, 'r': 0} 
pardict = {'beta': 2.4/14, 'D': 10, 'N': 1}

ds_rhs = '-beta * (i/N) * s' #divide by N to ensure it is a fraction
di_rhs = 'beta * (i/N) * s - ((1/D) * i)'
dr_rhs = '(1/D) * i'

vardict = {'s': ds_rhs, 'i': di_rhs, 'r': dr_rhs} 

DSargs = dst.args() 
DSargs.name = 'SIR'
DSargs.ics = icdict 
DSargs.pars = pardict 
DSargs.tdata = [0, 20] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'beta': 3.332/10, 'N': 4900000},
       ics={'s': 4899900, 'i': 100},
       tdata = [0, 100])

traj = DS.compute('demo')
pts = traj.sample()

s_frac = pts['s']/DS.pars['N']
Reff = s_frac * DS.pars['beta'] * DS.pars['D']

s_1 = pts['s'][-1]
i_1 = pts['i'][-1]
r_1 = pts['r'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000}, 
       ics={'s': s_1, 'i': i_1, 'r': r_1},
       tdata=[100, 150])

traj1 = DS.compute('demo')
pts1 = traj1.sample()

s_frac1 = pts1['s']/DS.pars['N']
Reff1 = s_frac1 * DS.pars['beta'] * DS.pars['D']

#---Plotting---#
plot1 = plt.figure(1)
plt.plot(np.concatenate((pts['t'], pts1['t'])), np.concatenate((pts['s'], pts1['s'])), label='S')
plt.plot(np.concatenate((pts['t'], pts1['t'])), np.concatenate((pts['i'], pts1['i'])), label='I')
plt.plot(np.concatenate((pts['t'], pts1['t'])), np.concatenate((pts['r'], pts1['r'])), label='R')
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(np.concatenate((pts['t'], pts1['t'])), np.concatenate((Reff, Reff1)), label='Reff')
plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--')
plt.legend()
plt.xlabel('t')

plt.show()


