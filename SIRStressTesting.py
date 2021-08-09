#---Import---#
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from decimal import Decimal as D

class SIR():
    def __init__(self, start, end, N, S, I, R, beta, D):
        self.T = end - start #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = (self.T / self.stepsize)
        self.wholenumber = 1
        self.t = np.arange(start,end)
        self.N = N
        self.S = np.array([S])
        self.I = np.array([I])
        self.R = np.array([R])
        self.DC = np.array([I])
        self.TC = np.array([I + R])
        self.beta = beta
        self.D = D
        self.Reff = np.array([self.beta * self.D])

    def calc(self):
        S, I, R, Reff, DC, TC = self.S[-1], self.I[-1], self.R[-1], self.Reff[-1], self.DC[-1], self.TC[-1] #makes sure always adding to last in sequence

        for i in range(int(self.numsteps)):
            S += self.stepsize * (-(self.beta * S * I/self.N))
            I += self.stepsize * ((self.beta * S * I/self.N) - ((1/self.D) * I))
            R += self.stepsize * (((1/self.D) * I))
            Reff = self.beta * self.D * S / self.N
            DC = (self.beta * S * I/self.N)
            TC = I + R

            if round(i * self.stepsize, 12) == self.wholenumber: #seperated by days 
                self.S = np.append(self.S, S)
                self.I = np.append(self.I, I)
                self.R = np.append(self.R, R)
                self.Reff = np.append(self.Reff, Reff)
                self.DC = np.append(self.DC, DC)
                self.TC = np.append(self.TC, TC)

                self.wholenumber += 1

        return self.t, self.N, self.S, self.I, self.R, self.Reff, self.DC, self.TC

    def reinitAdd(self, t, end, N, S, I, R, beta, D, Reff, DC, TC): #reinits but adds to pre-existing arrays, then calcs it, used as an extension
        self.T = end - t[-1] #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(t[0], end)
        self.N = N
        self.S = S
        self.I = I
        self.R = R
        self.DC = DC
        self.TC = TC
        self.beta = beta
        self.D = D
        self.Reff = Reff

        return self.calc()

    def plot(self):
        #---Plotting---#
        plot1 = plt.figure(1)
        plt.plot(self.t, self.S, color = "#1463E0", label='S')
        plt.plot(self.t, self.I, color = "#C40606", label='I')
        plt.plot(self.t, self.R, color = "#05A515", label='R')
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff, color = "#C40606", label='Variant 1')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot1 = plt.figure(3)
        plt.plot(self.t, self.DC, color = "#C40606", label='Variant 1')
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')

        plot1 = plt.figure(4)
        plt.plot(self.t, self.TC, color = "#C40606", label='Variant 1')
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')

        plt.show()

model = SIR(0, 140, 1000000, 999999, 1, 0, 3.332/10, 10)
model.calc()
model.plot()

#-stress testing fix for wholenumber floating point error
def func(number):
    sucess = False
    TimePeriod = number
    stepsize = 0.1 / TimePeriod
    numsteps = TimePeriod / stepsize
    wholenumber1 = 1
    wholenumber2 = 1

    for i in range(int(numsteps)):
        if i * stepsize == wholenumber1:
            wholenumber1 += 1
        if round(i * stepsize, 12) == wholenumber2:
            wholenumber2 += 1

    if wholenumber1 == TimePeriod:
        sucess1 = "T"
    else:
        sucess1 = "F"

    if wholenumber2 == TimePeriod:
        sucess2 = "T"
        sucess = True
    else:
        sucess2 = "F"
    
    print(f"{TimePeriod:3} - {sucess1} - {sucess2}")

    return sucess

#for j in range(1, 700): #definitely works up to 600
#    if func(j) == False:
#        print("\n", j)
#        break