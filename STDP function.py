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
def neuron(Vrest, Vt, taum, time, deltat, gex, gin, Eex, Ein):
    count = 1
    V = np.zeros(len(time))
    V[0] = Vrest
    for t in range(len(time)-1):
        V[t+1] = V[t] + deltat/taum*((-V[t] + Vrest) +gex*(Eex-V[t])+gin*(Ein-V[t]))
        if V[t+1] > Vt:
            V[t+1] = -60
            count +=1
    print 'The firing rate is :'+str(count*1000/time[-1])+' spike/s'   # firing rate ~ 1/T*1000/time[-1]
    return V

taum = 20
Vrest = -70
Eex = 0
Ein = -70
Vt = -54
gex = 0.5
gin = 0.5
deltat = 0.1

time = np.arange(0.0, 1000, deltat)

plt.plot(time, neuron(Vrest, Vt, taum, time, deltat, gex, gin, Eex, Ein))
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
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
