#---Import---#
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#---Functions---#
def RoundUpOrdOfMag(x):
    OrderOfMag = 10**(math.floor(math.log10(x)))
    Answer = math.ceil(x / OrderOfMag) * OrderOfMag

    return Answer

#---Models---#
class SIR():
    def __init__(self, start, end, N, S, I, R, beta, D):
        self.T = end - start #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
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

            if i * self.stepsize == self.wholenumber: #seperated by days
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
        #some calcs:
        RN = RoundUpOrdOfMag(self.N)
        RDC = RoundUpOrdOfMag(np.amax(self.DC))

        #---Plotting---#
        plot1 = plt.figure(1)
        plt.plot(self.t, self.S, color = "#1463E0", label='S')
        plt.plot(self.t, self.I, color = "#C40606", label='I')
        plt.plot(self.t, self.R, color = "#05A515", label='R')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff, color = "#C40606", label='Variant 1')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'r', linestyle = '--')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, math.ceil(np.amax(self.Reff)))
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot1 = plt.figure(3)
        plt.plot(self.t, self.DC, color = "#C40606", label='Variant 1')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, RDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')

        plot1 = plt.figure(4)
        plt.plot(self.t, self.TC, color = "#C40606", label='Variant 1')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')
        print("Total Cases at end: ", self.TC[-1])

        plt.show()

class SEIR():
    def __init__(self, start, end, N, S, I, R, beta, D):
        self.T = end - start #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
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

            if i * self.stepsize == self.wholenumber: #seperated by days
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
        #some calcs:
        RN = RoundUpOrdOfMag(self.N)
        RDC = RoundUpOrdOfMag(np.amax(self.DC))

        #---Plotting---#
        plot1 = plt.figure(1)
        plt.plot(self.t, self.S, color = "#1463E0", label='S')
        plt.plot(self.t, self.I, color = "#C40606", label='I')
        plt.plot(self.t, self.R, color = "#05A515", label='R')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff, color = "#C40606", label='Variant 1')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'r', linestyle = '--')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, math.ceil(np.amax(self.Reff)))
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot1 = plt.figure(3)
        plt.plot(self.t, self.DC, color = "#C40606", label='Variant 1')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, RDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')

        plot1 = plt.figure(4)
        plt.plot(self.t, self.TC, color = "#C40606", label='Variant 1')
        plt.xlim(self.t[0], self.t[-1])
        plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')
        print("Total Cases at end: ", self.TC[-1])

        plt.show()

#---Running---#
model1 = SIR(0, 100, 4900000, 4899900, 100, 0, 3.332/10, 10) #start, end, N, S, I, R, beta, D
t, N, S, I, R, Reff, DC, TC = model1.calc() #return self.t, self.N, self.S, self.I, self.R, self.Reff
t, N, S, I, R, Reff, DC, TC = model1.reinitAdd(t, 150, N, S, I, R, 3.332/10, 10, Reff, DC, TC)#t, end, N, S, I, R, beta, D, Reff, DC, TC
model1.plot()