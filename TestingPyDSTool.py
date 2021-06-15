import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



# setting up initial conditions, parameters and variables
icdict = {'s': 4900000, 'l': 0, 'i': 0, 'r':0}
pardict = {'beta': 0.3332, 'c': 1, 'delta': 100/7, 'L': 4.7, 'alpha': 10, 'N': 4900000, 'rho': 0}

# defining RHS of differential equations of the system
ds_rhs = '- (beta * c * s * (i/N)) - rho'

dl_rhs = '(beta * c * s * (i/N)) + (delta) - ((1/L) * l)'

di_rhs = '((1/L) * l) - ((1/alpha) * i)'

dr_rhs = '((1/alpha) * i) + rho'

vardict = {'s': ds_rhs, 'l': dl_rhs, 'i': di_rhs, 'r': dr_rhs} # tells DSTool that there are 5 dynamic state var's, specified by the eqns in the strings dS/dI/dR_rhs

DSargs = dst.args()
DSargs.name = 'SEIR' # name the model
DSargs.ics = icdict # set initial condn's
DSargs.pars = pardict # set parameters
DSargs.tdata = [0, 20] # how long we expect to integrate for
DSargs.varspecs = vardict

#Generator
SEIR = dst.Generator.Vode_ODEsystem(DSargs)



# adjusting parameters and initial conditions, can be done independently without solving ODE's again using generator
SEIR.set(pars={'c': 1, 'delta': 200/7, 'N': 4900000, 'rho': 0},
        ics={'i': 0, 's': 4900000, 'r': 0},
        tdata=[0,10]) # t is in days starting on Match 1st 2020

# computing points for plotting 
traj1_SEIR = SEIR.compute('test')
pts1_SEIR = traj1_SEIR.sample()

# creating plot
plt.plot(pts1_SEIR['t'], pts1_SEIR['i'], label='I')
#plt.plot(pts1_SEIR['t'], pts1_SEIR['s'], label='S')
#plt.plot(pts_SEIR['t'], pts_SEIR['r'], label='R')
#plt.plot(pts_SIRV['t'], pts_SIRV['l'], label='V')
plt.legend()
plt.xlabel('t')
plt.title('28 Feb - 10 March 2020, (N = 4,900,000)')
plt.show()
