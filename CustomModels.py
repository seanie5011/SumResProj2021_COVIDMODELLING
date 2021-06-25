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

        plot1 = plt.figure(3)
        plt.plot(self.t, self.DC1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.DC2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.DC3, color = "#FF7F00", label='Variant 3')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RTDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases by Variant')
        print("Total Cases at end: DC1DC2DC3", np.sum(self.DC1) + np.sum(self.DC2) + np.sum(self.DC3))

        plot1 = plt.figure(4)
        plt.plot(self.t, self.TDC, color = "#9D009C", label='All Variants')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RTDC)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')
        print("Total Cases at end: TDC", np.sum(self.TDC))

        plot1 = plt.figure(5)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')
        plt.xlim(self.t[0], self.t[-1])
        #plt.ylim(0, RN)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')
        print("Total Cases at end: TC", self.TC[-1])

        print(self.N)

        plt.show()

#---Running---#
model1 = SEIRMV(0, 10, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 200, 0, 0, 1, 157, 67000, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 27, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 75, 0, 0, 0.5, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 93, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 153, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 7, 0, 0, 0.3, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 217, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 100, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 235, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 25, 0, 0, 0.4, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 279, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 0, 0, 0.2, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 298, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 10, 1000, 0, 0.65, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 303, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 5.644/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 366, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 4.665/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.16, 157, 67000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 381, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 4.665/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.18, 157, 67000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
t, N, S, E1, I1, E2, I2, E3, I3, R, V, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC = model1.reinitAdd(t, 406, N, S, E1, I1, E2, I2, E3, I3, R, V, 3.332/10, 4.665/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, 0, 0, 0, 0.25, 157, 67000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity


model1.plot()