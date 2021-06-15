#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#---Tutorial Code---#
icdict = {'x': 1, 'y': 0.4} #initial conditions
pardict = {'k': 0.1, 'm': 0.5} #parameters

x_rhs = 'y'
y_rhs = '-k*x/m' #assigning to values previously shown, x and y change

vardict = {'x': x_rhs, 'y': y_rhs} #two dynamic state variables, value changes

DSargs = dst.args() #empty object in args class, need to add "dst" as this ensures it is coming from PyDSTool
DSargs.name = 'SHM' #name
DSargs.ics = icdict #assign
DSargs.pars = pardict #assign
DSargs.tdata = [0, 20] #determine how long we wanna integrate for
DSargs.varspecs = vardict #assign

DS = dst.Generator.Vode_ODEsystem(DSargs) #solver object

#DS.set(pars={'k': 0.3}, ics={'x': 0.4}) #changes parameters if wanted

traj = DS.compute('demo') #creates trajectory object
pts = traj.sample() #gives points from the trajectory

plt.plot(pts['t'], pts['x'], label='x')
plt.plot(pts['t'], pts['y'], label='y')
plt.legend()
plt.xlabel('t')
plt.show()

#---Showing Conseravtion of Energy---#
def KE(pts):
    return 0.5*DS.pars['m']*pts['y']**2

def PE(pts):
    return 0.5*DS.pars['k']*pts['x']**2

total_energy = KE(pts) + PE(pts) #returns array of all the total energies, all same values