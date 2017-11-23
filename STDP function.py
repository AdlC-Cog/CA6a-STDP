import numpy as np
import matplotlib.pyplot as plt

#%%
def Change_function(Apos,Aneg,taupos,tauneg,time):
    F = np.zeros(len(time))
    for i,t in enumerate(time):
        if t<0:
            F[i] = Apos*exp(t/taupos)
            
        else:
            F[i] = Aneg*exp(-t/tauneg)
    return F

Apos = 0.005 # in %
Aneg = -1.05*Apos # in%
taupos = tauneg = 20 #ms
deltat = 0.1
time = np.arange(-100.0, 100, deltat)


plt.plot(time,Change_function(Apos,Aneg,taupos,tauneg,time)*100)
plt.xlabel(r'$\Delta$t (ms)')
plt.ylabel('STDP (%)')


#%%

#%%

import numpy as np

spikes = []
num_cells = 200
delay = 10000
frequency = 10
num_spikes_per_cell = delay/1000*frequency + delay/100000*frequency 
time = np.arange(0.0, delay, deltat)
deltat = 0.1

def ismember(B,A):
    B_unique_sorted, B_idx = np.unique(B, return_index=True)
    B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)
    return B_in_A_bool

for i in range(num_cells):
    isi = np.random.poisson(frequency, num_spikes_per_cell).astype(float)
    spikes.append(ismember(time,np.cumsum(isi*10,dtype=float)))
