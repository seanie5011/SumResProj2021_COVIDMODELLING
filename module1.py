import numpy as np

npadded = np.array([4, 5])
npadded = np.append(npadded, 6)
print(npadded)

npadded1 = np.array([7, 8])
npadded1 = np.append(npadded1, 9)
print(npadded1)

npadded2 = np.array([10, 11])
npadded2 = np.append(npadded2, 12)
print(npadded2)

npdatadata = [npadded, npadded1, npadded2]

npdata = np.array([0, 1, 2])
for i in range(len(npdatadata)):
    if i == 0:
        npdata = np.append(np.array([npdata]), np.array([npdatadata[i]]), axis = 0) #np.array([[4,5,6,7]])
    else:
        npdata = np.append(npdata, np.array([npdatadata[i]]), axis = 0)
    print(npdata)

np.savetxt("helpme.csv", npdata, delimiter=',')

A = np.genfromtxt("helpme.csv", delimiter=',')
print(A)

#data = [t, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, TC] #t, N, VDCkeeper
#for i in range(len(data)):
#    if i != 0:
#        npdata = np.append(npdata, np.array([data[i]]), axis = 0)
#    else:
#        npdata = np.array([data[0]])
#np.savetxt("CustomModelsData.csv", npdata, delimiter = ',')
#print(t, ",", N, ",", VDCkeeper)

#A = np.genfromtxt("CustomModelsData.csv", delimiter = ',')
#t, S, E1, I1, E2, I2, E3, I3, R, IV, V, rhokeeper, Ckeeper, Reff1, Reff2, Reff3, DC1, DC2, DC3, TDC, VDC, TC = A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9], A[10], A[11], A[12], A[13], A[14], A[15], A[16], A[17], A[18], A[19], A[20], A[21]
#N, VDCkeeper = 4910973.428583844 , 1645.9299761053821

