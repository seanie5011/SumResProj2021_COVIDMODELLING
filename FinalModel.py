#---Import---#
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
import os, csv

#---Functions---#
def RoundUpOrdOfMag(x):
    OrderOfMag = 10**(math.floor(math.log10(x)))
    Answer = math.ceil(x / OrderOfMag) * OrderOfMag

    return Answer

#---Model---#
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
            if self.N - (V + IV + self.VDCkeeper) <= 850000: #if cant vaccinate
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
            DC1 = (self.beta1 * self.C * (S + IV * self.p2 + V * self.p) * I1 /self.N) + self.delta1 - (self.mu * (E1 + I1)) #keep an eye on the self.mu part
            DC2 = (self.beta2 * self.C * (S + IV * self.p2 + V * self.p) * I2 /self.N) + self.delta2 - (self.mu * (E2 + I2))
            DC3 = (self.beta3 * self.C * (S + IV * self.p2 + V * self.p) * I3 /self.N) + self.delta3 - (self.mu * (E3 + I3))
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
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Active cases')

        plot2 = plt.figure(2)
        plt.plot(self.t, self.Reff1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.Reff2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.Reff3, color = "#FF7F00", label='Variant 3')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')

        #plt.ylim(0, 4) #only use when on display mode
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3) 
        plt.xlim(self.t[0], self.t[-1])
        plt.margins(x=0, y=0)
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        plot3 = plt.figure(3)
        plt.plot(self.t, self.DC1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.DC2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.DC3, color = "#FF7F00", label='Variant 3')

        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        #plt.ylim(0, 7000) #only use when on display mode
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.margins(x=0, y=0)
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases by Variant')

        plot4 = plt.figure(4)
        plt.plot(self.t, self.TDC, color = "#9D009C", label='All Variants') #marker='o'
        #plt.plot(self.t, self.VDC, color = "#1A5803", label='VDC')
        #plt.text(30, 1300, "Wave 1: 1150") #Day 50
        #plt.text(210, 1900, "Wave 2: 1690") #Day 223
        #plt.text(300, 8600, "Wave 3: 8550") #Day 321
        #plt.text(450, 2300, "Final Date:\nJuly 22nd: 2000") #Day 506

        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        #plt.ylim(0, RTDC) #only use when on display mode
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.margins(x=0, y=0)
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')
        print("Total Cases at end: TDC", np.sum(self.TDC))

        plot5 = plt.figure(5)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')

        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3) 
        plt.margins(x=0, y=0)
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')

        plot6 = plt.figure(6)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')

        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.xlim(self.t[0], self.t[-1])
        plt.yscale("log")
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Total cases (Logarithmic)')

        plot7 = plt.figure(7)
        plt.plot(self.t, self.V, color = "#05A515", label='V')
        plt.plot(self.t, self.IV, color = "#C6BF00", label='IV')
        plt.plot(self.t, self.R, color = "#9D009C", label='R')

        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Vaccination Programme')

        plot8 = plt.figure(8)
        plt.plot(self.t, self.Ckeeper, color = "#9D009C", label='C')

        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3) 
        plt.xlim(self.t[0], self.t[-1])
        plt.margins(x=0, y=0)
        plt.ylim(0, 1)
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

        plot10 = plt.figure(10)
        plt.fill_between(self.t, (self.DC1) * 100 / self.TDC, 0, color = "#E70000", label = "Variant 1") 
        plt.fill_between(self.t, (self.DC1 + self.DC2) * 100 / self.TDC, (self.DC1) * 100 / self.TDC, color = "#F50480", label = "Variant 2")
        plt.fill_between(self.t,(self.DC1 + self.DC2 + self.DC3) * 100 / self.TDC,(self.DC1 + self.DC2) * 100 / self.TDC, color = "#FF7F00", label = "Variant 3")
        
        plt.plot(np.array([288, 296]), np.array([50, 50]), color = 'k', linestyle = '--')
        plt.text(270, 55, "Variant 2 50%") #Day 293
        plt.plot(np.array([480, 488]), np.array([50, 50]), color = 'k', linestyle = '--')
        plt.text(460, 55, "Variant 3 50%") #Day 483

        #plt.xticks(self.monthspoints, self.monthslist) #only use when on display mode
        plt.margins(x=0, y=0)
        plt.legend() 
        plt.ylabel("Variant% of Total Cases")


        plt.show()

    def dataplot(self):
        #--plots--#
        #Reff
        plot1 = plt.figure(1)
        plt.plot(self.t, self.Reff1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.Reff2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.Reff3, color = "#FF7F00", label='Variant 3')
        plt.axline((self.t[0], 1), (self.t[-1], 1), color = 'k', linestyle = '--')

        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3) 
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Effective Reproduction Number')

        #DC by Variant
        plot2 = plt.figure(2)
        plt.plot(self.t, self.DC1, color = "#E70000", label='Variant 1')
        plt.plot(self.t, self.DC2, color = "#F50480", label='Variant 2')
        plt.plot(self.t, self.DC3, color = "#FF7F00", label='Variant 3')

        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases by Variant')

        #TDC
        plot3 = plt.figure(3)
        plt.plot(self.t, self.TDC, color = "#9D009C", label='All Variants') #marker='o'

        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3)
        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.xlabel('Time in days')
        plt.ylabel('Daily cases')

        #TC
        plot4 = plt.figure(4)
        plt.plot(self.t, self.TC, color = "#C6BF00", label='All Variants')

        plt.xlim(self.t[0], self.t[-1])
        plt.legend()
        plt.grid(color = '#A4A4A4', linestyle = '-', linewidth = 0.3) 
        plt.xlabel('Time in days')
        plt.ylabel('Total cases')

        #--general--#
        plt.show()

#---Running---#

#--History--#
##March
#model = SEIRMVEx(0, 12, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 0, 0) #start, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.calc() #return self.t, self.N, self.S, self.E1, self.I1, self.E2, self.I2, self.E3, self.I3, self.R, self.V, self.Reff1, self.Reff2, self.Reff3, self.DC1, self.DC2, self.DC3, self.TDC, self.TC
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 30, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 120, 0, 0, 0.9, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 37, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 80, 0, 0, 0.7, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 44, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 60, 0, 0, 0.6, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 51, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 40, 0, 0, 0.4, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 62, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 0, 0.3, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 94, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 0, 0.16, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 124, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 7, 0, 0, 0.16, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##July
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 154, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 25, 0, 0, 0.3, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##August
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 169, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 12, 0, 0, 0.35, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 184, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 25, 0, 0, 0.4, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##September
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 216, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 50, 0, 0, 0.6, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##October
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 233, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 70, 0, 0, 0.5, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 245, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 40, 0, 0, 0.2, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##November
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 276, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 0/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 20, 0, 0, 0.16, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##December
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 302, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 20, 1000, 0, 0.6, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 306, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 20, 700, 0, 0.5, Ckeeper, 157, 0, 0)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##January
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 321, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 10, 700, 0, 0.4, Ckeeper, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 336, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 100, 0, 0.16, Ckeeper, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##February
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 366, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 0/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 0, 0.16, Ckeeper, 157, 38000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##March
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 396, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 10, 0.16, Ckeeper, 157, 75000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##April
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 411, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 20, 0.2, Ckeeper, 157, 85000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 426, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 40, 0.22, Ckeeper, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##May
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 435, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 80, 0.22, Ckeeper, 157, 165000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 440, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 120, 0.25, Ckeeper, 157, 280000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 455, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 150, 0.3, Ckeeper, 157, 95000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##June
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 485, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 200, 0.33, Ckeeper, 157, 170000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##July
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 507, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 200, 0.6, Ckeeper, 157, 180000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 515, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 220, 0.62, Ckeeper, 157, 180000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity

#-writing to file-#
#data = [t, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, TC]
#with open("HistoryEndJuly.csv", "w", newline="") as csvfile:
#            writer = csv.writer(csvfile, delimiter = ",")
#            for array in data:
#                writer.writerow(array.tolist()) #writes each row at a time

#-reading file-#
newdata = []
with open("HistoryEndJuly.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        newdata.append(row)

for x, list in enumerate(newdata):
    for y, string in enumerate(list):
        if x == 0:
            newdata[x][y] = int(string)
        else:
            newdata[x][y] = float(string)

#-reinitialise model after reading-#
model = SEIRMVEx(0, 12, 4900000, 4900000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.332/10, 0/10, 0/10, 10, 4.7, 100, 0, 0, 1, 157, 0, 0)
t, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, TC = np.array(newdata[0]), np.array(newdata[1]), np.array(newdata[2]), np.array(newdata[3]), np.array(newdata[4]), np.array(newdata[5]), np.array(newdata[6]), np.array(newdata[7]), np.array(newdata[8]), np.array(newdata[9]), np.array(newdata[10]), np.array(newdata[11]), np.array(newdata[12]), np.array(newdata[13]), np.array(newdata[14]), np.array(newdata[15]), np.array(newdata[16]), np.array(newdata[17]), np.array(newdata[18]), np.array(newdata[19]), np.array(newdata[20]), np.array(newdata[21])
N, VDCkeeper = 4910377.714300612, 6970.914876131299
t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 515, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 220, 0.62, Ckeeper, 157, 180000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity


#--Future--#
##August
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 545, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 500, 0.7, Ckeeper, 157, 120000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##September
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 575, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 600, 0.8, Ckeeper, 157, 90000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##October
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 605, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 600, 0.8, Ckeeper, 157, 80000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##November
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 635, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 400, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##December
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 665, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 400, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##January
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 696, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 500, 0.6, Ckeeper, 157, 210000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity
##test
#t, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC = model.reinitAdd(t, 1200, N, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, 3.332/10, 5.35/10, 9.095/10, 10, 4.7, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, VDCkeeper, TC, 0, 0, 500, 0.7, Ckeeper, 157, 90000, 0.9)#t, end, N, S, E1, I1, E2, I2, E3, I3, R, V, beta1, beta2, beta3, D, L, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, TC, deltaweek1, deltaweek2, deltaweek3, C, mupop, rhoweek, immunity

#--info--#
print("Total Population:", N)
print("Total Vaccinated:", V[-1] + IV[-1] + VDCkeeper)
print("Total Not Vaccinated:", N - (V[-1] + IV[-1] + VDCkeeper))
print("Sum TDC:", np.sum(TDC))
print(f"Jun28 (483): Delta {DC3[483]*100/TDC[483]}% and Alpha {DC2[483]*100/TDC[483]}%")
print(f"Jul12 (497): Delta {DC3[497]*100/TDC[497]}% and Alpha {DC2[497]*100/TDC[497]}%")
print(f"Most Recent ({t[-1]}): Delta {DC3[-1]*100/TDC[-1]}% and Alpha {DC2[-1]*100/TDC[-1]}%")

#--plotting--#
model.dataplot()

#--notes--#
#show difference in 1100000 and 850000 for non vaccinated, and 90% 92% 95% immunity

