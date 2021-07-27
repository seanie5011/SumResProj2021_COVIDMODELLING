#---Import---#
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation

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
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')
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
    def __init__(self, start, end, N, S, E, I, R, V, beta, D, L, deltaweek, C, mupop, rhoweek, immunity):
        self.T = end - start #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(start,end)
        self.N = N
        self.S = np.array([S])
        self.E = np.array([E])
        self.I = np.array([I])
        self.R = np.array([R])
        self.V = np.array([V])
        self.DC = np.array([E + I + R])
        self.TC = np.array([E + I + R])
        self.beta = beta
        self.D = D
        self.L = L
        self.Reff = np.array([self.beta * self.D])
        self.delta = deltaweek / 7.0
        self.C = C
        self.mu = mupop / self.N
        self.rho = rhoweek / 7.0
        self.p = immunity

    def calc(self):
        S, E, I, R, V, Reff, DC, TC = self.S[-1], self.E[-1], self.I[-1], self.R[-1], self.V[-1], self.Reff[-1], self.DC[-1], self.TC[-1] #makes sure always adding to last in sequence

        for i in range(int(self.numsteps)):
            if S > 0: #as normal
                S += self.stepsize * (-(self.beta * self.C * S * I/self.N) + (self.mu * (self.N - S)) - (self.rho * self.p))
                V += self.stepsize * ((self.rho * self.p) - (self.mu * V))
            else: #ensures cant vaccinate or infect negative people, still need to add in birth / death rate
                S += self.stepsize * (self.mu * self.N)
                V += self.stepsize * (-(self.mu * V))
            E += self.stepsize * ((self.beta * self.C * S * I/self.N) - ((1/self.L) * E) + self.delta - (self.mu * E))
            I += self.stepsize * (((1/self.L) * E) - ((1/self.D) * I) - (self.mu * I))
            R += self.stepsize * (((1/self.D) * I) - (self.mu * R))
            Reff = self.beta * self.D * S / self.N
            DC = (self.beta * self.C * S * I/self.N) + self.delta - (self.mu * (self.N - S))
            TC = E + I + R
            self.N += self.stepsize * self.delta

            if i * self.stepsize == self.wholenumber: #seperated by days
                self.S = np.append(self.S, S)
                self.E = np.append(self.E, E)
                self.I = np.append(self.I, I)
                self.R = np.append(self.R, R)
                self.V = np.append(self.V, V)
                self.Reff = np.append(self.Reff, Reff)
                self.DC = np.append(self.DC, DC)
                self.TC = np.append(self.TC, TC)

                self.wholenumber += 1

        return self.t, self.N, self.S, self.E, self.I, self.R, self.V, self.Reff, self.DC, self.TC

    def reinitAdd(self, t, end, N, S, E, I, R, V, beta, D, L, Reff, DC, TC, deltaweek, C, mupop, rhoweek, immunity): #reinits but adds to pre-existing arrays, then calcs it, used as an extension
        self.T = end - t[-1] #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(t[0], end)
        self.N = N
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.V = V
        self.DC = DC
        self.TC = TC
        self.beta = beta
        self.D = D
        self.L = L
        self.Reff = Reff
        self.delta = deltaweek / 7.0
        self.C = C
        self.mu = mupop / self.N
        self.rho = rhoweek / 7.0
        self.p = immunity

        return self.calc()

    def plot(self):
        #some calcs:
        RN = RoundUpOrdOfMag(self.N)
        RDC = RoundUpOrdOfMag(np.amax(self.DC))

        #---Plotting---#
        plot1 = plt.figure(1)
        #plt.plot(self.t, self.S, color = "#1463E0", label='S')
        plt.plot(self.t, self.E, color = "#F2880A", label='E')
        plt.plot(self.t, self.I, color = "#C40606", label='I')
        #plt.plot(self.t, np.add(self.R, self.V), color = "#05A515", label='R + V')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff, color = "#C40606", label='Variant 1')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, math.ceil(np.amax(self.Reff)))
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot1 = plt.figure(3)
        plt.plot(self.t, self.DC, color = "#C40606", label='Variant 1')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')
        print(np.sum(self.DC))

        plot1 = plt.figure(4)
        plt.plot(self.t, self.TC, color = "#C40606", label='Variant 1')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')
        print("Total Cases at end: ", self.TC[-1])

        print(np.sum(self.DC) - self.TC[-1])
        print(self.E[-1] + self.I[-1] + self.R[-1] + self.S[-1] + self.V[-1])
        print(self.N)

        plt.show()

class SEIRMV():
    def __init__(self, start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity):
        self.T = end - start #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(start,end)
        self.N = N
        self.S = np.array([S])
        self.E1 = np.array([E1])
        self.I1 = np.array([I1])
        self.E2 = np.array([E2])
        self.I2 = np.array([I2])
        self.E3 = np.array([E3])
        self.I3 = np.array([I3])
        self.R = np.array([R])
        self.V = np.array([V])
        self.DC1 = np.array([E1 + I1])
        self.DC2 = np.array([E2 + I2])
        self.DC3 = np.array([E3 + I3])
        self.TDC = np.array([E1 + I1 + E2 + I2 + E3 + I3])
        self.TC = np.array([E1 + I1 + E2 + I2 + E3 + I3 + R])
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.D = D
        self.L = L
        self.Reff1 = np.array([self.beta1 * self.D])
        self.Reff2 = np.array([self.beta2 * self.D])
        self.Reff3 = np.array([self.beta3 * self.D])
        self.delta1 = deltaweek1 / 7.0
        self.delta2 = deltaweek2 / 7.0
        self.delta3 = deltaweek3 / 7.0
        self.C = C
        self.mu = mupop / self.N
        self.rho = rhoweek / 7.0
        self.p = immunity

    def calc(self):
        S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = self.S[-1], self.E1[-1], self.I1[-1], self.E2[-1], self.I2[-1], self.E3[-1], self.I3[-1], self.R[-1], self.V[-1], self.Reff1[-1], self.Reff2[-1], self.Reff3[-1], self.DC1[-1], self.DC2[-1], self.DC3[-1], self.TDC[-1], self.TC[-1] #makes sure always adding to last in sequence

        for i in range(int(self.numsteps)):
            if S > 0: #as normal
                S += self.stepsize * (-((self.beta1 * I1 + self.beta2 * I2 + self.beta3 * I3) * self.C * S / self.N) + (self.mu * (self.N - S)) - (self.rho * self.p))
                V += self.stepsize * ((self.rho * self.p) - (self.mu * V))
            else: #ensures cant vaccinate or infect negative people, still need to add in birth / death rate
                S += self.stepsize * (self.mu * self.N)
                V += self.stepsize * (-(self.mu * V))
            E1 += self.stepsize * ((self.beta1 * self.C * S * I1/self.N) - ((1/self.L) * E1) + self.delta1 - (self.mu * E1))
            I1 += self.stepsize * (((1/self.L) * E1) - ((1/self.D) * I1) - (self.mu * I1))
            E2 += self.stepsize * ((self.beta2 * self.C * S * I2/self.N) - ((1/self.L) * E2) + self.delta2 - (self.mu * E2))
            I2 += self.stepsize * (((1/self.L) * E2) - ((1/self.D) * I2) - (self.mu * I2))
            E3 += self.stepsize * ((self.beta3 * self.C * S * I3/self.N) - ((1/self.L) * E3) + self.delta3 - (self.mu * E3))
            I3 += self.stepsize * (((1/self.L) * E3) - ((1/self.D) * I3) - (self.mu * I3))
            R += self.stepsize * (((1/self.D) * (I1 + I2 + I3)) - (self.mu * R))
            Reff1 = self.beta1 * self.D * S / self.N
            Reff2 = self.beta2 * self.D * S / self.N
            Reff3 = self.beta3 * self.D * S / self.N
            DC1 = (self.beta1 * self.C * S * I1/self.N) + self.delta1 - (self.mu * (E1 + I1)) #keep an eye on the self.mu part
            DC2 = (self.beta2 * self.C * S * I2/self.N) + self.delta2 - (self.mu * (E2 + I2))
            DC3 = (self.beta3 * self.C * S * I3/self.N) + self.delta3 - (self.mu * (E3 + I3))
            TDC = DC1 + DC2 + DC3
            TC = E1 + I1 + E2 + I2 + E3 + I3 + R
            self.N += self.stepsize * (self.delta1 + self.delta2 + self.delta3)

            if i * self.stepsize == self.wholenumber: #seperated by days
                self.S = np.append(self.S, S)
                self.E1 = np.append(self.E1, E1)
                self.I1 = np.append(self.I1, I1)
                self.E2 = np.append(self.E2, E2)
                self.I2 = np.append(self.I2, I2)
                self.E3 = np.append(self.E3, E3)
                self.I3 = np.append(self.I3, I3)
                self.R = np.append(self.R, R)
                self.V = np.append(self.V, V)
                self.Reff1 = np.append(self.Reff1, Reff1)
                self.Reff2 = np.append(self.Reff2, Reff2)
                self.Reff3 = np.append(self.Reff3, Reff3)
                self.DC1 = np.append(self.DC1, DC1)
                self.DC2 = np.append(self.DC2, DC2)
                self.DC3 = np.append(self.DC3, DC3)
                self.TDC = np.append(self.TDC, TDC)
                self.TC = np.append(self.TC, TC)

                self.wholenumber += 1

        return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC

    def reinitAdd(self, t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity): #reinits but adds to pre-existing arrays, then calcs it, used as an extension
        self.T = end - t[-1] #T is the length of time, not end time
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(t[0], end)
        self.N = N
        self.S = S
        self.E1 = E1
        self.I1 = I1
        self.E2 = E2
        self.I2 = I2
        self.E3 = E3
        self.I3 = I3
        self.R = R
        self.V = V
        self.DC1 = DC1
        self.DC2 = DC2
        self.DC3 = DC3
        self.TDC = TDC
        self.TC = TC
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.D = D
        self.L = L
        self.Reff1 = Reff1
        self.Reff2 = Reff2
        self.Reff3 = Reff3
        self.delta1 = deltaweek1 / 7.0
        self.delta2 = deltaweek2 / 7.0
        self.delta3 = deltaweek3 / 7.0
        self.C = C
        self.mu = mupop / self.N
        self.rho = rhoweek / 7.0
        self.p = immunity

        return self.calc()

    def plot(self):
        #some calcs:
        RN = RoundUpOrdOfMag(self.N)
        RTDC = RoundUpOrdOfMag(np.amax(self.TDC))

        #---Plotting---#
        plot1 = plt.figure(1)
        #plt.plot(self.t, self.S, color = "#1463E0", label='S')
        plt.plot(self.t, self.E1, color = "#AB0000", label='E1')
        plt.plot(self.t, self.I1, color = "#E70000", label='I1')
        plt.plot(self.t, self.E2, color = "#AA0158", label='E2')
        plt.plot(self.t, self.I2, color = "#F50480", label='I2')
        plt.plot(self.t, self.E3, color = "#CE6600", label='E3')
        plt.plot(self.t, self.I3, color = "#FF7F00", label='I3')
        #plt.plot(self.t, np.add(self.R, self.V), color = "#05A515", label='R + V')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.Reff2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.Reff3, color = "#FF7F00", label='Variant 3')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, math.ceil(np.amax(self.Reff)))
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot3 = plt.figure(3)
        plt.plot(self.t, self.DC1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.DC2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.DC3, color = "#FF7F00", label='Variant 3')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RTDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases by Variant')
        print("Total Cases at end: DC1DC2DC3", np.sum(self.DC1) + np.sum(self.DC2) + np.sum(self.DC3))

        plot4 = plt.figure(4)
        plt.plot(self.t, self.TDC, color = "#9D009C", label='All Variants') #marker='o'
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RTDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')
        print("Total Cases at end: TDC", np.sum(self.TDC))

        plot5 = plt.figure(5)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xticks(self.monthspoints, self.monthslist)
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')
        print("Total Cases at end: TC", self.TC[-1])

        plot6 = plt.figure(6)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.yscale("log")
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases (Logarithmic)')

        print("Total Population:", self.N)

        plt.show()

class SEIRMVEx():
    def __init__(self, start, end, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, Ctarget, mupop, rhoweek, immunity):
        #self.monthslist = ["January", "February", "March", "April", "May", "June", "July", "August", "Septhember", "October", "November", "December"]
        self.monthslist = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "2021", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
        self.monthspoints = [0, 30, 62, 94, 124, 154, 184, 216, 245, 276, 306, 335, 365, 395, 425, 455, 485]
        
        self.T = end - start #T is the length of time, not end time
        self.timekeeper = start
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(start,end)
        self.N = N
        self.S = np.array([S])
        self.E1 = np.array([E1])
        self.I1 = np.array([I1])
        self.E2 = np.array([E2])
        self.I2 = np.array([I2])
        self.E3 = np.array([E3])
        self.I3 = np.array([I3])
        self.R = np.array([R])
        self.IV = np.array([IV])
        self.V = np.array([V])
        self.DC1 = np.array([E1 + I1])
        self.DC2 = np.array([E2 + I2])
        self.DC3 = np.array([E3 + I3])
        self.TDC = np.array([E1 + I1 + E2 + I2 + E3 + I3])
        self.VDC = np.array([0])
        self.VDCkeeper = 0
        self.TC = np.array([E1 + I1 + E2 + I2 + E3 + I3 + R])
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.D = D
        self.L = L
        self.Reff1 = np.array([self.beta1 * self.D])
        self.Reff2 = np.array([self.beta2 * self.D])
        self.Reff3 = np.array([self.beta3 * self.D])
        self.delta1 = deltaweek1 / 7.0
        self.delta2 = deltaweek2 / 7.0
        self.delta3 = deltaweek3 / 7.0
        self.Ctarget = Ctarget
        self.Ckeeper = np.array([self.Ctarget])
        self.C = self.Ckeeper[-1]
        self.mu = mupop / self.N
        self.rho = rhoweek / 7.0
        self.ogrho = self.rho
        self.rhokeeper = np.array([self.rho])
        self.p = 1 - immunity
        self.p2 = 1 - 0.5 * immunity

    def calc(self):
        S, E1, I1, E2, I2, E3, I3, R, IV, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, TC, self.C = self.S[-1], self.E1[-1], self.I1[-1], self.E2[-1], self.I2[-1], self.E3[-1], self.I3[-1], self.R[-1], self.IV[-1], self.V[-1], self.Reff1[-1], self.Reff2[-1], self.Reff3[-1], self.DC1[-1], self.DC2[-1], self.DC3[-1], self.TDC[-1], self.VDC[-1], self.TC[-1], self.Ckeeper[-1] #makes sure always adding to last in sequence
        if self.Ctarget < self.C:
            deltaC = abs(self.Ctarget - self.C) / 10 #tightening takes place over 10 days
        elif self.Ctarget > self.C:
            deltaC = abs(self.Ctarget - self.C) / 20 #loosening takes place over 20 days
        else:
            deltaC = 0

        for i in range(int(self.numsteps)):
            if self.N - (V + IV + self.VDCkeeper) <= 1100000: #if cant vaccinate
                self.rho = 0
            else:
                self.rho = self.ogrho #ensures goes back to vaccinating with number given for this section

            S += self.stepsize * (-((self.beta1 * I1 + self.beta2 * I2 + self.beta3 * I3) * self.C * S / self.N) + (self.mu * (self.N - S)) - (self.rho))
            IV += self.stepsize * ((self.rho) - ((self.beta1 * I1 + self.beta2 * I2 + self.beta3 * I3) * self.C * (IV * self.p2) / self.N) - (self.mu * IV))
            V += self.stepsize * ( - ((self.beta1 * I1 + self.beta2 * I2 + self.beta3 * I3) * self.C * (V * self.p) / self.N) - (self.mu * V))
            E1 += self.stepsize * ((self.beta1 * self.C * (S + IV * self.p2 + V * self.p) * I1/self.N) - ((1/self.L) * E1) + self.delta1 - (self.mu * E1))
            I1 += self.stepsize * (((1/self.L) * E1) - ((1/self.D) * I1) - (self.mu * I1))
            E2 += self.stepsize * ((self.beta2 * self.C * (S + IV * self.p2 + V * self.p) * I2/self.N) - ((1/self.L) * E2) + self.delta2 - (self.mu * E2))
            I2 += self.stepsize * (((1/self.L) * E2) - ((1/self.D) * I2) - (self.mu * I2))
            E3 += self.stepsize * ((self.beta3 * self.C * (S + IV * self.p2 + V * self.p) * I3/self.N) - ((1/self.L) * E3) + self.delta3 - (self.mu * E3))
            I3 += self.stepsize * (((1/self.L) * E3) - ((1/self.D) * I3) - (self.mu * I3))
            R += self.stepsize * (((1/self.D) * (I1 + I2 + I3)) - (self.mu * R))
            Reff1 = self.beta1 * self.C * self.D * (S + IV * self.p2 + V * self.p) / self.N
            Reff2 = self.beta2 * self.C * self.D * (S + IV * self.p2 + V * self.p) / self.N
            Reff3 = self.beta3 * self.C * self.D * (S + IV * self.p2 + V * self.p) / self.N
            DC1 = (self.beta1 * self.C * S * I1 /self.N) + self.delta1 - (self.mu * (E1 + I1)) #keep an eye on the self.mu part
            DC2 = (self.beta2 * self.C * S * I2 /self.N) + self.delta2 - (self.mu * (E2 + I2))
            DC3 = (self.beta3 * self.C * S * I3 /self.N) + self.delta3 - (self.mu * (E3 + I3))
            VDC = ((self.beta1 * I1 + self.beta2 * I2 + self.beta3 * I3) * self.C * (V * self.p + IV * self.p2) / self.N)  - (self.mu * (V + IV))
            TDC = DC1 + DC2 + DC3
            TC += self.stepsize * TDC
            self.N += self.stepsize * (self.delta1 + self.delta2 + self.delta3)

            if i * self.stepsize == self.wholenumber: #seperated by days
                if self.t[self.timekeeper + self.wholenumber] > 270: #ensure time period is over recovery period
                    S += (self.TDC[self.timekeeper + self.wholenumber - 270] - self.VDC[self.timekeeper + self.wholenumber - 270])  #add back to susceptible
                    R -= (self.TDC[self.timekeeper + self.wholenumber - 270] - self.VDC[self.timekeeper + self.wholenumber - 270])
                if self.t[self.timekeeper + self.wholenumber] > 14: 
                    V += self.rhokeeper[self.timekeeper + self.wholenumber - 14] + self.VDC[self.timekeeper + self.wholenumber - 14] 
                    IV -= self.rhokeeper[self.timekeeper + self.wholenumber - 14]
                    R -= self.VDC[self.timekeeper + self.wholenumber - 14]
                    self.VDCkeeper -= self.VDC[self.timekeeper + self.wholenumber - 14]

                if self.C < self.Ctarget:
                    self.C += deltaC
                elif self.C > self.Ctarget:
                    self.C -= deltaC

                self.S = np.append(self.S, S)
                self.E1 = np.append(self.E1, E1)
                self.I1 = np.append(self.I1, I1)
                self.E2 = np.append(self.E2, E2)
                self.I2 = np.append(self.I2, I2)
                self.E3 = np.append(self.E3, E3)
                self.I3 = np.append(self.I3, I3)
                self.R = np.append(self.R, R)
                self.IV = np.append(self.IV, IV)
                self.V = np.append(self.V, V)
                self.rhokeeper = np.append(self.rhokeeper, self.rho)
                self.Ckeeper = np.append(self.Ckeeper, self.C)
                self.Reff1 = np.append(self.Reff1, Reff1)
                self.Reff2 = np.append(self.Reff2, Reff2)
                self.Reff3 = np.append(self.Reff3, Reff3)
                self.DC1 = np.append(self.DC1, DC1)
                self.DC2 = np.append(self.DC2, DC2)
                self.DC3 = np.append(self.DC3, DC3)
                self.TDC = np.append(self.TDC, TDC)
                self.VDC = np.append(self.VDC, VDC)
                self.VDCkeeper += VDC
                self.TC = np.append(self.TC, TC)

                self.wholenumber += 1

        return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.IV, self.V, self.rhokeeper, self.Ckeeper, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.VDC, self.VDCkeeper, self.TC

    def reinitAdd(self, t, end, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, deltaweek1, deltaweek2, deltaweek3, Ctarget, Ckeeper, mupop, rhoweek, immunity): #reinits but adds to pre-existing arrays, then calcs it, used as an extension
        self.T = end - t[-1] #T is the length of time, not end time
        self.timekeeper = t[-1]
        self.stepsize = 0.1 / self.T
        self.numsteps = self.T / self.stepsize
        self.wholenumber = 1
        self.t = np.arange(t[0], end)
        self.N = N
        self.S = S
        self.E1 = E1
        self.I1 = I1
        self.E2 = E2
        self.I2 = I2
        self.E3 = E3
        self.I3 = I3
        self.R = R
        self.IV = IV
        self.V = V
        self.DC1 = DC1
        self.DC2 = DC2
        self.DC3 = DC3
        self.TDC = TDC
        self.VDC = VDC
        self.VDCkeeper = VDCkeeper
        self.TC = TC
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.D = D
        self.L = L
        self.Reff1 = Reff1
        self.Reff2 = Reff2
        self.Reff3 = Reff3
        self.delta1 = deltaweek1 / 7.0
        self.delta2 = deltaweek2 / 7.0
        self.delta3 = deltaweek3 / 7.0
        self.Ctarget = Ctarget
        self.Ckeeper = Ckeeper
        self.C = self.Ckeeper[-1]
        self.mu = mupop / self.N
        self.rho = rhoweek / 7.0
        self.ogrho = self.rho
        self.rhokeeper = rhokeeper
        self.p = 1 - immunity
        self.p2 = 1 - 0.5 * immunity

        return self.calc()

    def plot(self):
        #some calcs:
        RN = RoundUpOrdOfMag(self.N)
        RTDC = RoundUpOrdOfMag(np.amax(self.TDC))

        #---Plotting---#
        plot1 = plt.figure(1)
        plt.plot(self.t, self.S, color = "#1463E0", label='S')
        plt.plot(self.t, self.E1, color = "#AB0000", label='E1')
        plt.plot(self.t, self.I1, color = "#E70000", label='I1')
        plt.plot(self.t, self.E2, color = "#AA0158", label='E2')
        plt.plot(self.t, self.I2, color = "#F50480", label='I2')
        plt.plot(self.t, self.E3, color = "#CE6600", label='E3')
        plt.plot(self.t, self.I3, color = "#FF7F00", label='I3')
        plt.plot(self.t, np.add(np.add(self.R, self.V), self.IV), color = "#05A515", label='R + IV + V')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.Reff2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.Reff3, color = "#FF7F00", label='Variant 3')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, math.ceil(np.amax(self.Reff)))
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot3 = plt.figure(3)
        plt.plot(self.t, self.DC1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.DC2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.DC3, color = "#FF7F00", label='Variant 3')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RTDC)
        plt.legend()
        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases by Variant')
        print("Total Cases at end: DC1DC2DC3", np.sum(self.DC1) + np.sum(self.DC2) + np.sum(self.DC3))

        plot4 = plt.figure(4)
        plt.plot(self.t, self.TDC, color = "#9D009C", label='All Variants') #marker='o'
        plt.plot(self.t, self.VDC, color = "#1A5803", label='VDC')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RTDC)
        plt.legend()
        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')
        print("Total Cases at end: TDC", np.sum(self.TDC))

        plot5 = plt.figure(5)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')
        print("Total Cases at end: TC", self.TC[-1])

        plot6 = plt.figure(6)
        #for i in range(math.floor(self.t[-1] / 30)):
        #    if i + 3 > len(self.monthslist) - 1:
        #        plt.axline((self.t[0] + i * 30, 0), (self.t[0] + i * 30, np.amax(self.TC)), color = 'k', linestyle = '--', label = self.monthslist[i + 3 - 12])
        #    else:
        #        plt.axline((self.t[0] + i * 30, 0), (self.t[0] + i * 30, np.amax(self.TC)), color = 'k', linestyle = '--', label = self.monthslist[i + 3])
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.yscale("log")
        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases (Logarithmic)')

        plot7 = plt.figure(7)
        plt.plot(self.t, self.V, color = "#05A515", label='V')
        plt.plot(self.t, self.IV, color = "#C6BF00", label='IV')
        plt.plot(self.t, self.R, color = "#9D009C", label='R')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Vaccination Programme')

        plot8 = plt.figure(8)
        plt.plot(self.t, self.Ckeeper, color = "#9D009C", label='C')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Contact Rate')

        plot9 = plt.figure(9)
        Days = ['Jun28 (483)', 'Jul12 (497)', f'Most Recent ({self.t[-1]})']
        Variant1 = [self.DC1[483] * 100 / self.TDC[483], self.DC1[497] * 100 / self.TDC[497], self.DC1[-1] * 100 / self.TDC[-1]]
        Variant2 = [self.DC2[483] * 100 / self.TDC[483], self.DC2[497] * 100 / self.TDC[497], self.DC2[-1] * 100 / self.TDC[-1]]
        Variant3 = [self.DC3[483] * 100 / self.TDC[483], self.DC3[497] * 100 / self.TDC[497], self.DC3[-1] * 100 / self.TDC[-1]]
        plt.bar(Days, Variant1, 0.35, color = "#E70000", label='Variant 1')
        plt.bar(Days, Variant2, 0.35, bottom = Variant1, color = "#F50480", label='Variant 2')
        plt.bar(Days, Variant3, 0.35, bottom = Variant2, color = "#FF7F00", label='Variant 3')
        plt.legend() 
        plt.ylabel("Variant% of Total Cases")

        print("Total Population:", self.N)

        plt.show()

#---Running---#
#-Daniels-#
#model1 = SEIRMV(0, 10, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 200, 0, 0, 1, 157, 67000, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 27, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 75, 0, 0, 0.5, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 93, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 153, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 7, 0, 0, 0.3, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 217, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 100, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 235, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 279, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 298, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 1000, 0, 0.65, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 303, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 366, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 4.665/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 381, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 4.665/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.18, 157, 67000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 406, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 4.665/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.25, 157, 67000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity


#model1.plot()

#-FirstTrial-#

##March
#model2 = SEIRMV(0, 12, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 67000, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 30, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 120, 0, 0, 0.9, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 37, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 80, 0, 0, 0.7, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 44, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 60, 0, 0, 0.6, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 51, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.5, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 62, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 94, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 124, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 7, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##July
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 154, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.3, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##August
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 169, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 12, 0, 0, 0.35, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 184, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##September
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 201, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 50, 0, 0, 0.45, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 216, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##October
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 233, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 70, 0, 0, 0.45, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 245, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.35, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##November
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 276, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##December
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 302, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 500, 0, 0.6, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 306, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 300, 0, 0.55, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##January
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 311, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 15, 150, 0, 0.45, 157, 63000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 319, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 100, 0, 0.35, 157, 63000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 328, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 5, 50, 0, 0.25, 157, 63000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 335, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 20, 0, 0.2, 157, 63000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##February
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 365, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 5, 0, 0.16, 157, 75000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##March
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 395, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.2, 157, 110000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 410, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.2, 157, 120000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 425, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 20, 5, 0.25, 157, 150000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 436, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 50, 5, 0.25, 157, 200000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 440, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 80, 5, 0.4, 157, 300000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 455, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 100, 10, 0.4, 157, 120000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 485, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 9.031/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 50, 30, 0.45, 157, 200000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity

#-SecondTrial-#

##March
#model2 = SEIRMV(0, 12, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 67000, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 30, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 120, 0, 0, 0.9, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 37, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 80, 0, 0, 0.7, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 44, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 60, 0, 0, 0.6, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 51, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.5, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 62, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 94, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 124, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 7, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##July
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 154, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.3, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##August
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 169, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 12, 0, 0, 0.35, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 184, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##September
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 201, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 50, 0, 0, 0.45, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 216, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##October
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 233, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 70, 0, 0, 0.45, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 245, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.35, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##November
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 276, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##December
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 302, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 500, 0, 0.6, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 306, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 300, 0, 0.55, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##January
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 311, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 15, 150, 0, 0.45, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 319, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 100, 0, 0.35, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 328, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 5, 50, 0, 0.25, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 335, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 20, 0, 0.2, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##February
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 365, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 5, 0, 0.16, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##March
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 395, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.2, 157, 75000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 410, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.2, 157, 85000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 425, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 20, 5, 0.25, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 436, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 50, 5, 0.25, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 440, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 80, 5, 0.3, 157, 280000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 455, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 100, 10, 0.35, 157, 95000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 485, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 9.031/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 50, 30, 0.4, 157, 207000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity

##-ThirdTrial-#

##March
#model2 = SEIRMV(0, 12, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 67000, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 30, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 120, 0, 0, 0.9, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 37, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 80, 0, 0, 0.7, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 44, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 60, 0, 0, 0.6, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 51, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.5, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 62, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 94, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 124, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 7, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##July
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 154, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.3, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##August
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 169, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 12, 0, 0, 0.35, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 184, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##September
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 201, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 50, 0, 0, 0.45, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 216, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##October
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 233, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 70, 0, 0, 0.45, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 245, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 40, 0, 0, 0.35, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##November
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 276, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##December
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 302, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 1000, 0, 0.6, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 306, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 20, 300, 0, 0.55, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##January
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 311, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 15, 150, 0, 0.45, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 319, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 100, 0, 0.35, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 328, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 5, 50, 0, 0.25, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 335, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.2, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##February
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 365, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 5, 0, 0.16, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##March
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 395, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.16, 157, 75000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 410, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 10, 0, 0.2, 157, 85000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 425, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 20, 5, 0.2, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 436, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 50, 5, 0.25, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 440, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 80, 5, 0.25, 157, 280000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 455, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 80, 10, 0.3, 157, 95000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model2.reinitAdd(t, 485, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.35/10, 9.031/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 50, 30, 0.35, 157, 207000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity


#print("Total Vaccinated:", V[-1] / 0.9) #as vaccinated group is * 0.9 in calc, this gives number HSE use
#print("Susceptible left:", S[-1])
#print("Total Delta variant:", np.sum(DC3))
#print("Total Cases * 0.010712876:", TC[-1] * 0.010712876)
#model2.plot()

#-Experimental-#

#model3 = SEIRMVEx(0, 297, 1000000, 1000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 15000, 0.9) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, Ctarget, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model3.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.IV, self.V, self.rhokeeper, self.Ckeeper, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model3.reinitAdd(t, 353, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 100, 0, 0, 0.5, Ckeeper, 157, 50000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, Ctarget, Ckeeper, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model3.reinitAdd(t, 453, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 100, 0, 0, 0.8, Ckeeper, 157, 30000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, Ctarget, Ckeeper, mupop, rhoweek, immunity

#March
model3 = SEIRMVEx(0, 12, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 0, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 30, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 120, 0, 0, 0.9, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#April
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 37, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 80, 0, 0, 0.7, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 44, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 60, 0, 0, 0.6, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 51, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 40, 0, 0, 0.4, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 62, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 0, 0.3, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#May
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 94, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 0, 0.16, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#June
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 124, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 7, 0, 0, 0.16, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#July
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 154, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 25, 0, 0, 0.3, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#August
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 169, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 12, 0, 0, 0.35, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 184, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 25, 0, 0, 0.4, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#September
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 216, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 50, 0, 0, 0.6, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#October
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 233, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 70, 0, 0, 0.5, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 245, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 40, 0, 0, 0.2, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#November
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 276, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 20, 0, 0, 0.16, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#December
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 302, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 20, 1000, 0, 0.6, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 306, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 20, 700, 0, 0.5, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#January
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 321, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 10, 700, 0, 0.4, Ckeeper, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 336, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 100, 0, 0.16, Ckeeper, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#February
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 366, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 0, 0.16, Ckeeper, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#March
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 396, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 10, 0.16, Ckeeper, 157, 75000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#April
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 411, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 20, 0.2, Ckeeper, 157, 85000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 426, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 40, 0.25, Ckeeper, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#May
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 435, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 80, 0.25, Ckeeper, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 440, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 120, 0.25, Ckeeper, 157, 280000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 455, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 150, 0.3, Ckeeper, 157, 95000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#June
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 485, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 250, 0.35, Ckeeper, 157, 180000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#July
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 507, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 400, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 515, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 500, 0.7, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#August
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 545, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 500, 0.7, Ckeeper, 157, 120000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#September
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 575, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 600, 0.8, Ckeeper, 157, 90000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#October
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 605, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 600, 0.8, Ckeeper, 157, 80000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##November
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 635, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 400, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##December
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 665, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 400, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##January
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model3.reinitAdd(t, 696, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 400, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity

print("Total Vaccinated:", V[-1] + IV[-1] + VDCkeeper)
print("Total Not Vaccinated:", N - (V[-1] + IV[-1] + VDCkeeper))
print("Susceptible left:", S[-1])
print("Recovered left:", R[-1])
print("Sum TDC:", np.sum(TDC))
print("Total Delta variant:", np.sum(DC3))
print("Total Cases * 0.017120777:", TC[-1] * 0.017120777)
print("Total Vaccinated Cases * 0.010712876:", np.sum(VDC) * 0.010712876)
print(f"Jun28 (483): Delta {DC3[483]*100/TDC[483]}% and Alpha {DC2[483]*100/TDC[483]}%")
print(f"Jul12 (497): Delta {DC3[497]*100/TDC[497]}% and Alpha {DC2[497]*100/TDC[497]}%")
print(f"Most Recent ({t[-1]}): Delta {DC3[-1]*100/TDC[-1]}% and Alpha {DC2[-1]*100/TDC[-1]}%")
model3.plot()

#---Animation Plotting---#
#time = np.arange(0, 335)

#def animate(i):

#    plt.cla()

#    plt.xlabel("Days")
#    plt.ylabel("Population")

#    plt.plot(time[:i],I1[:i] + I2[:i],color="#C84134",label="Active Cases")

#    plt.legend()

#ani = animation.FuncAnimation(plt.gcf(),animate,interval=1)

#plt.show()