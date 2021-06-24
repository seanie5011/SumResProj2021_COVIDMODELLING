#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#this model will include vaccination, birth/death rates, imports, contact rates and immunity percentage

#---SEIR Code---#
icdict = {'s': 0.999, 'ex': 0, 'i': 0.001, 'r': 0} 
pardict = {'beta': 2.4/14, 'D': 10, 'L': 4.7, 'N': 1, 'rho': 0, 'mu': 0, 'delta': 0, 'C': 0.8, 'p': 0.9}

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

DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 50/7}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': 4900000, 'ex': 0, 'i': 0, 'r': 0},
       tdata=[0, 1000])

traj = DS.compute('demo')
pts = traj.sample()

#---Plotting---#
plt.plot(pts['t'], pts['s'], label='S')
plt.plot(pts['t'], pts['ex'], label='E')
plt.plot(pts['t'], pts['i'], label='I')
plt.plot(pts['t'], pts['r'], label='R')

plt.legend()
plt.xlabel('t')

plt.show()
