#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#---SEIR Code---#
icdict = {'s': 0.999, 'ex': 0, 'i': 0.001, 'r': 0} 
pardict = {'beta': 2.4/14, 'D': 14, 'L': 5, 'N': 1}

ds_rhs = '-beta * (i/N) * s' #divide by N to ensure it is a fraction
de_rhs = 'beta * (i/N) * s - ((1/L) * ex)'
di_rhs = '((1/L) * ex) - ((1/D) * i)'
dr_rhs = '(1/D) * i'

vardict = {'s': ds_rhs, 'ex': de_rhs, 'i': di_rhs, 'r': dr_rhs} 

DSargs = dst.args() 
DSargs.name = 'SEIR'
DSargs.ics = icdict 
DSargs.pars = pardict 
DSargs.tdata = [0, 20] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'beta': 5/14, 'N': 4900000},
       ics={'s': 4899999, 'i': 1},
       tdata = [0, 250])

traj = DS.compute('demo')
pts = traj.sample()

s_frac = pts['s']/DS.pars['N']
Reff = s_frac * DS.pars['beta'] * DS.pars['D']
#---Plotting---#
plot1 = plt.figure(1)
plt.plot(pts['t'], pts['s'], label='S')
plt.plot(pts['t'], pts['ex'], label='E')
plt.plot(pts['t'], pts['i'], label='I')
plt.plot(pts['t'], pts['r'], label='R')
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(pts['t'], Reff, label='Reff')
plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--')
plt.legend()
plt.xlabel('t')

plt.show()
