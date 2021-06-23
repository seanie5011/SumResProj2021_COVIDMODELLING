#---Import---#
import PyDSTool as dst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#this model will include vaccination, birth/death rates, imports, contact rates and immunity percentage

#---SEIR Code---#
icdict = {'s': 0.999, 'ex': 0, 'i': 0.001, 'r': 0, 'daily_cases': 0} 
pardict = {'beta': 2.4/14, 'D': 10, 'L': 4.7, 'N': 1, 'rho': 0, 'mu': 0, 'delta': 0, 'C': 1, 'p': 0}

ds_rhs = '(-beta * C * (i/N) * s) - ((rho * p + mu) * s) + (mu * N)' #mu * (s/N) brings in a fraction of mu, so that it is total removed from all sections
de_rhs = '(beta * C * (i/N) * s) + delta - ((1/L) * ex) - (mu * ex)' #importing cases from outside country, disease still dies because everyone infected
di_rhs = '((1/L) * ex) - ((1/D) * i) - (mu * i)'
dr_rhs = '((1/D) * i) + (rho * p * s) - (mu * r)' #multiply by p as this is the immunity rate, if 45% immunity, only 45% of people are vaccinated to recovered stage
ddc_rhs = '((1/L) * ex)' #dailycases is just infections

vardict = {'s': ds_rhs, 'ex': de_rhs, 'i': di_rhs, 'r': dr_rhs, 'daily_cases': ddc_rhs} 

DSargs = dst.args() 
DSargs.name = 'SEIR'
DSargs.ics = icdict 
DSargs.pars = pardict 
DSargs.tdata = [0, 10] 
DSargs.varspecs = vardict

DS = dst.Generator.Vode_ODEsystem(DSargs)

DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 200/7}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': 4900000, 'ex': 0, 'i': 0, 'r': 0},
       tdata=[0, 10])

traj = DS.compute('demo')
pts = traj.sample()

#for j in range(len(pts['s'])): #ensures cant go to negative values, all pts have same length
#    if pts['s'][j] < 0:
#        pts['s'][j] = 0
#    if pts['r'][j] > DS.pars['N']:
#        pts['r'][j] = DS.pars['N']

#j = 0
#run = true
#while run == true:
#    if pts['s'][j+1] < 0 or j+1 >= len(pts['s']):
#        max_time = pts['t'][j+1]
#        run = false
#    j+=1

s_frac = pts['s']/DS.pars['N']
Reff = s_frac * DS.pars['beta'] * DS.pars['D']

subtotal_cases = np.add(pts['i'], pts['r'])
#total_cases = np.add(subtotal_cases, pts['r']) #incliuding exposed in total cases
#total_cases = np.add(pts['i'], pts['r']) #adding current cases to previous cases (assuming no vaccination)

s_1 = pts['s'][-1]
ex_1 = pts['ex'][-1]
i_1 = pts['i'][-1]
r_1 = pts['r'][-1]
dc_1 = pts['daily_cases'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 75/7, 'C': 0.5}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_1, 'ex': ex_1, 'i': i_1, 'r': r_1, 'daily_cases': dc_1},
       tdata=[10, 27])

traj1 = DS.compute('demo')
pts1 = traj1.sample()

s_frac1 = pts1['s']/DS.pars['N']
Reff1 = s_frac1 * DS.pars['beta'] * DS.pars['D']
subtotal_cases1 = np.add(pts1['i'], pts1['r'])

s_2 = pts1['s'][-1]
ex_2 = pts1['ex'][-1]
i_2 = pts1['i'][-1]
r_2 = pts1['r'][-1]
dc_2 = pts1['daily_cases'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 0/7, 'C': 0.2}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_2, 'ex': ex_2, 'i': i_2, 'r': r_2, 'daily_cases': dc_2},
       tdata=[27, 93])

traj2 = DS.compute('demo')
pts2 = traj2.sample()
#total imported cases by now is 414
s_frac2 = pts2['s']/DS.pars['N']
Reff2 = s_frac2 * DS.pars['beta'] * DS.pars['D']
subtotal_cases2 = np.add(pts2['i'], pts2['r'])

s_3 = pts2['s'][-1]
ex_3 = pts2['ex'][-1]
i_3 = pts2['i'][-1]
r_3 = pts2['r'][-1]
dc_3 = pts2['daily_cases'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 7/7, 'C': 0.3}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_3, 'ex': ex_3, 'i': i_3, 'r': r_3, 'daily_cases': dc_3},
       tdata=[93, 153])

traj3 = DS.compute('demo')
pts3 = traj3.sample()

s_frac3 = pts3['s']/DS.pars['N']
Reff3 = s_frac3 * DS.pars['beta'] * DS.pars['D']
subtotal_cases3 = np.add(pts3['i'], pts3['r'])

s_4 = pts3['s'][-1]
ex_4 = pts3['ex'][-1]
i_4 = pts3['i'][-1]
r_4 = pts3['r'][-1]
dc_4 = pts3['daily_cases'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 100/7, 'C': 0.4}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_4, 'ex': ex_4, 'i': i_4, 'r': r_4, 'daily_cases': dc_4},
       tdata=[153, 217])

traj4 = DS.compute('demo')
pts4 = traj4.sample()

s_frac4 = pts4['s']/DS.pars['N']
Reff4 = s_frac4 * DS.pars['beta'] * DS.pars['D']
subtotal_cases4 = np.add(pts4['i'], pts4['r'])

s_5 = pts4['s'][-1]
ex_5 = pts4['ex'][-1]
i_5 = pts4['i'][-1]
r_5 = pts4['r'][-1]
dc_5 = pts4['daily_cases'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 25/7, 'C': 0.4}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_5, 'ex': ex_5, 'i': i_5, 'r': r_5, 'daily_cases': dc_5},
       tdata=[217, 234])

traj5 = DS.compute('demo')
pts5 = traj5.sample()

s_frac5 = pts5['s']/DS.pars['N']
Reff5 = s_frac5 * DS.pars['beta'] * DS.pars['D']
subtotal_cases5 = np.add(pts5['i'], pts5['r'])

s_6 = pts5['s'][-1]
ex_6 = pts5['ex'][-1]
i_6 = pts5['i'][-1]
r_6 = pts5['r'][-1]
dc_6 = pts5['daily_cases'][-1]

#set again to redo
DS.set(pars={'beta': 3.332/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta': 10/7, 'C': 0.2}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_6, 'ex': ex_6, 'i': i_6, 'r': r_6, 'daily_cases': dc_6},
       tdata=[234, 278])

traj6 = DS.compute('demo')
pts6 = traj6.sample()

s_frac6 = pts6['s']/DS.pars['N']
Reff6 = s_frac6 * DS.pars['beta'] * DS.pars['D']
subtotal_cases6 = np.add(pts6['i'], pts6['r'])

s_7 = pts6['s'][-1]
ex_7 = pts6['ex'][-1]
i_7 = pts6['i'][-1]
r_7 = pts6['r'][-1]
dc_7 = pts6['daily_cases'][-1]

#2nd variant
vicdict = {'s': 0.999, 'ex1': 0, 'i1': 0.0003, 'ex2': 0, 'i2': 0.0003, 'r': 0, 'vdaily_cases': 0} 
vpardict = {'beta1': 2.4/14, 'beta2': 4/14, 'D': 10, 'L': 4.7, 'N': 1, 'rho': 0, 'mu': 0, 'delta1': 0, 'delta2': 0, 'C': 1, 'p': 0}

vds_rhs = '(-beta1 * C * (i1/N) * s) - (beta2 * C * (i2/N) * s) - ((rho * p + mu) * s) + (mu * N)' #mu * (s/N) brings in a fraction of mu, so that it is total removed from all sections
vde1_rhs = '(beta1 * C * (i1/N) * s) + delta1 - ((1/L) * ex1) - (mu * ex1)' #importing cases from outside country, disease still dies because everyone infected
vdi1_rhs = '((1/L) * ex1) - ((1/D) * i1) - (mu * i1)'
vde2_rhs = '(beta2 * C * (i2/N) * s) + delta2 - ((1/L) * ex2) - (mu * ex2)' #importing cases from outside country, disease still dies because everyone infected
vdi2_rhs = '((1/L) * ex2) - ((1/D) * i2) - (mu * i2)'
vdr_rhs = '((1/D) * (i1 + i2)) + (rho * p * s) - (mu * r)' #multiply by p as this is the immunity rate, if 45% immunity, only 45% of people are vaccinated to recovered stage
vddc_rhs = '((1/L) * (ex1 + ex2))' #dailycases is just infections


vvardict = {'s': vds_rhs, 'ex1': vde1_rhs, 'i1': vdi1_rhs, 'ex2': vde2_rhs, 'i2': vdi2_rhs, 'r': vdr_rhs, 'vdaily_cases': vddc_rhs} 

vDSargs = dst.args() 
vDSargs.name = 'vSEIR'
vDSargs.ics = vicdict 
vDSargs.pars = vpardict 
vDSargs.tdata = [0, 20] 
vDSargs.varspecs = vvardict

vDS = dst.Generator.Vode_ODEsystem(vDSargs)

vDS.set(pars={'beta1': 3.332/10, 'beta2': 5.644/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta1': 10/7, 'delta2': 1000/7, 'C': 0.65}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': s_7, 'ex1': ex_7, 'i1': i_7, 'ex2': 0, 'i2': 0, 'r': r_7, 'vdaily_cases': dc_7},
       tdata=[278, 297])

vtraj = vDS.compute('demo')
vpts = vtraj.sample()

vs_frac1 = vpts['s']/vDS.pars['N']
vReff1_1 = vs_frac1 * vDS.pars['beta1'] * vDS.pars['D']
vReff2_1 = vs_frac1 * vDS.pars['beta2'] * vDS.pars['D']
vsubtotal_cases = np.add(np.add(vpts['i1'], vpts['i2']), vpts['r'])

vs_1 = vpts['s'][-1]
vex1_1 = vpts['ex1'][-1]
vi1_1 = vpts['i1'][-1]
vex2_1 = vpts['ex2'][-1]
vi2_1 = vpts['i2'][-1]
vr_1 = vpts['r'][-1]
vdc_1 = vpts['vdaily_cases'][-1]

#redo
vDS.set(pars={'beta1': 3.332/10, 'beta2': 5.644/10, 'N': 4900000, 'rho': (67000/7)/4900000, 'mu': 157/4900000, 'delta1': 0/7, 'delta2': 0/7, 'C': 0.16}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': vs_1, 'ex1': vex1_1, 'i1': vi1_1, 'ex2': vex2_1, 'i2': vi2_1, 'r': vr_1, 'vdaily_cases': vdc_1},
       tdata=[297, 303])

vtraj1 = vDS.compute('demo')
vpts1 = vtraj1.sample()

vs_frac2 = vpts1['s']/vDS.pars['N']
vReff1_2 = vs_frac2 * vDS.pars['beta1'] * vDS.pars['D']
vReff2_2 = vs_frac2 * vDS.pars['beta2'] * vDS.pars['D']
vsubtotal_cases1 = np.add(np.add(vpts1['i1'], vpts1['i2']), vpts1['r'])

vs_2 = vpts1['s'][-1]
vex1_2 = vpts1['ex1'][-1]
vi1_2 = vpts1['i1'][-1]
vex2_2 = vpts1['ex2'][-1]
vi2_2 = vpts1['i2'][-1]
vr_2 = vpts1['r'][-1]
vdc_2 = vpts1['vdaily_cases'][-1]

#redo
vDS.set(pars={'beta1': 3.332/10, 'beta2': 4.665/10, 'N': 4900000, 'rho': (63000/7)/4900000, 'mu': 157/4900000, 'delta1': 0/7, 'delta2': 0/7, 'C': 0.16, 'p': 0.9}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': vs_2, 'ex1': vex1_2, 'i1': vi1_2, 'ex2': vex2_2, 'i2': vi2_2, 'r': vr_2, 'vdaily_cases': vdc_2},
       tdata=[303, 365])

vtraj2 = vDS.compute('demo')
vpts2 = vtraj2.sample()

vs_frac3 = vpts2['s']/vDS.pars['N']
vReff1_3 = vs_frac3 * vDS.pars['beta1'] * vDS.pars['D']
vReff2_3 = vs_frac3 * vDS.pars['beta2'] * vDS.pars['D']
vsubtotal_cases2 = np.add(np.add(vpts2['i1'], vpts2['i2']), vpts2['r'])

vs_3 = vpts2['s'][-1]
vex1_3 = vpts2['ex1'][-1]
vi1_3 = vpts2['i1'][-1]
vex2_3 = vpts2['ex2'][-1]
vi2_3 = vpts2['i2'][-1]
vr_3 = vpts2['r'][-1]
vdc_3 = vpts2['vdaily_cases'][-1]

#redo
vDS.set(pars={'beta1': 3.332/10, 'beta2': 4.665/10, 'N': 4900000, 'rho': (63000/7)/4900000, 'mu': 157/4900000, 'delta1': 0/7, 'delta2': 0/7, 'C': 0.18, 'p': 0.9}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': vs_3, 'ex1': vex1_3, 'i1': vi1_3, 'ex2': vex2_3, 'i2': vi2_3, 'r': vr_3, 'vdaily_cases': vdc_3},
       tdata=[365, 380])

vtraj3 = vDS.compute('demo')
vpts3 = vtraj3.sample()

vs_frac4 = vpts3['s']/vDS.pars['N']
vReff1_4 = vs_frac4 * vDS.pars['beta1'] * vDS.pars['D']
vReff2_4 = vs_frac4 * vDS.pars['beta2'] * vDS.pars['D']
vsubtotal_cases3 = np.add(np.add(vpts3['i1'], vpts3['i2']), vpts3['r'])

vs_4 = vpts3['s'][-1]
vex1_4 = vpts3['ex1'][-1]
vi1_4 = vpts3['i1'][-1]
vex2_4 = vpts3['ex2'][-1]
vi2_4 = vpts3['i2'][-1]
vr_4 = vpts3['r'][-1]
vdc_4 = vpts3['vdaily_cases'][-1]

#redo
vDS.set(pars={'beta1': 3.332/10, 'beta2': 4.665/10, 'N': 4900000, 'rho': (63000/7)/4900000, 'mu': 157/4900000, 'delta1': 0/7, 'delta2': 0/7, 'C': 0.25, 'p': 0.9}, #mu decided by rounding ((14371*4)/365)/4900000, amount of births gotten from quarterly report by CSO, delta is 1000 people per week
       ics={'s': vs_4, 'ex1': vex1_4, 'i1': vi1_4, 'ex2': vex2_4, 'i2': vi2_4, 'r': vr_4, 'vdaily_cases': vdc_4},
       tdata=[380, 406])

vtraj4 = vDS.compute('demo')
vpts4 = vtraj4.sample()

vs_frac5 = vpts4['s']/vDS.pars['N']
vReff1_5 = vs_frac5 * vDS.pars['beta1'] * vDS.pars['D']
vReff2_5 = vs_frac5 * vDS.pars['beta2'] * vDS.pars['D']
vsubtotal_cases4 = np.add(np.add(vpts4['i1'], vpts4['i2']), vpts4['r'])

vs_5 = vpts4['s'][-1]
vex1_5 = vpts4['ex1'][-1]
vi1_5 = vpts4['i1'][-1]
vex2_5 = vpts4['ex2'][-1]
vi2_5 = vpts4['i2'][-1]
vr_5 = vpts4['r'][-1]
vdc_5 = vpts4['vdaily_cases'][-1]

total_s = np.concatenate((pts['s'], pts1['s'], pts2['s'], pts3['s'], pts4['s'], pts5['s'], pts6['s'], vpts['s'], vpts1['s'], vpts2['s'], vpts3['s'], vpts4['s']))
total_ex1 = np.concatenate((pts['ex'], pts1['ex'], pts2['ex'], pts3['ex'], pts4['ex'], pts5['ex'], pts6['ex'], vpts['ex1'], vpts1['ex1'], vpts2['ex1'], vpts3['ex1'], vpts4['ex1']))
total_i1 = np.concatenate((pts['i'], pts1['i'], pts2['i'], pts3['i'], pts4['i'], pts5['i'], pts6['i'], vpts['i1'], vpts1['i1'], vpts2['i1'], vpts3['i1'], vpts4['i1']))
total_ex2 = np.concatenate((vpts['ex2'], vpts1['ex2'], vpts2['ex2'], vpts3['ex2'], vpts4['ex2']))
total_i2 = np.concatenate((vpts['i2'], vpts1['i2'], vpts2['i2'], vpts3['i2'], vpts4['i2']))
total_r = np.concatenate((pts['r'], pts1['r'], pts2['r'], pts3['r'], pts4['r'], pts5['r'], pts6['r'], vpts['r'], vpts1['r'], vpts2['r'], vpts3['r'], vpts4['r']))
total_dc = np.concatenate((pts['daily_cases'], pts1['daily_cases'], pts2['daily_cases'], pts3['daily_cases'], pts4['daily_cases'], pts5['daily_cases'], pts6['daily_cases'], vpts['vdaily_cases'], vpts1['vdaily_cases'], vpts2['vdaily_cases'], vpts3['vdaily_cases'], vpts4['vdaily_cases']))
total_Reff1 = np.concatenate((Reff, Reff1, Reff2, Reff3, Reff4, Reff5, Reff6, vReff1_1, vReff1_2, vReff1_3, vReff1_4, vReff1_5))
total_Reff2 = np.concatenate((vReff2_1, vReff2_2, vReff2_3, vReff2_4, vReff2_5))
total_cases = np.concatenate((subtotal_cases, subtotal_cases1, subtotal_cases2, subtotal_cases3, subtotal_cases4, subtotal_cases5, subtotal_cases6, vsubtotal_cases, vsubtotal_cases1, vsubtotal_cases2, vsubtotal_cases3, vsubtotal_cases4))
total_time = np.concatenate((pts['t'], pts1['t'], pts2['t'], pts3['t'], pts4['t'], pts5['t'], pts6['t'], vpts['t'], vpts1['t'], vpts2['t'], vpts3['t'], vpts4['t']))
vtotal_time = np.concatenate((vpts['t'], vpts1['t'], vpts2['t'], vpts3['t'], vpts4['t']))

all_daily_cases = []
for j in range(len(total_time[1:-1:100])):
    if j != 0:
        temp_list = total_dc[1:-1:100][j] - total_dc[1:-1:100][j-1]
        all_daily_cases.append(temp_list)
    else:
        all_daily_cases.append(total_dc[0])

print(np.sum(all_daily_cases))
print(total_cases[-1])

#---Plotting---#
plot1 = plt.figure(1)
#plt.axline((0, 0), (250, 0), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5) #(0, (5, 10)) is a custom dashed line
#plt.axline((0, 4900000), (250, 4900000), color = 'r', linestyle = (0, (5, 10)), linewidth = 0.5)
#plt.plot(total_time, total_s, label='S')
#plt.plot(total_time, total_ex1, label='E1')
plt.plot(total_time, total_i1, label='I1')
#plt.plot(vtotal_time, total_ex2, label='E2')
plt.plot(vtotal_time, total_i2, label='I2')
#plt.plot(total_time, total_r, label='R')
#plt.xlim(0, 250) #limits max and min x-axis shown
#plt.ylim(0, 5000000)
plt.legend()
plt.xlabel('t')

plot2 = plt.figure(2)
plt.plot(total_time, total_Reff1, label='Reff1')
plt.plot(vtotal_time, total_Reff2, label='Reff2')
#plt.axline((0, 1), (100, 1), color = 'r', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.xlabel('t')

plot3 = plt.figure(3)
plt.plot(total_time, total_cases, label='Total Cases')
plt.legend()
plt.xlabel('t')

plot4 = plt.figure(4)
plt.plot(total_time[1:-1:100], all_daily_cases, label='Daily Cases')
plt.legend()
plt.xlabel('t')

plt.show()
