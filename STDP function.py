import numpy as np
import matplotlib.pyplot as plt

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

def ismember(B,A):
    B_unique_sorted, B_idx = np.unique(B, return_index=True)
    B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)
    return B_in_A_bool

def spike_trains(num_cells,frequency,delay):
    spikes = []
    num_spikes_per_cell = delay/1000*frequency + delay/100000*frequency 
    
    
    for i in range(num_cells):
        isi = np.random.poisson(frequency, num_spikes_per_cell).astype(float)
        spikes.append(ismember(time,np.cumsum(isi*10,dtype=float)))
    return spikes
    

#%%
def neuron(Vrest, Vt, taum, tau_ex, tau_in, tau_minus, tau_plus, time, deltat, gex_0, gin_0, g_in_bar, M, Pa, Eex, Ein):
    
    #generate spike trains for inhibitory and excitatory neuron populations with function spike_trains
    N = 1000 #nb of excitatory synapses
    inhibitory_spike_trains = spike_trains(200,10,delay)
    excitatory_spike_trains = spike_trains(N,10,delay)
    
    count = 1
    V = np.zeros(len(time))
    V[0] = Vrest
    g_ex = np.zeros(len(time))
    g_ex[0]= gex_0
    g_in = np.zeros(len(time))
    g_in[0]= gin_0
    M = np.zeros(len(time))
    M[0] = 0
    Pa
        
        
    
    
    for t in range(len(time)-1):
        
        g_ex[t+1]= g_ex[t] + deltat*(-g_ex[t]/tau_ex)
        g_in[t+1]= g_in[t] + deltat*(-g_in[t]/tau_in)
        
        for inhb_neurons in inhibitory_spike_trains:
            if inhb_neurons[t+1] == True:
                g_in[t+1] += g_in_bar
                
        V[t+1] = V[t] + deltat/taum*((-V[t] + Vrest) +g_ex[t]*(Eex-V[t])+g_in[t]*(Ein-V[t]))
        if V[t+1] > Vt:
            V[t+1] = -60
            count +=1
            
    print ('The firing rate is :'+str(count*1000/time[-1])+' spike/s')   # firing rate ~ 1/T*1000/time[-1]
    
    return V

delay = 10000
taum = 20
tau_ex = 5
tau_in = 5
Vrest = -70
Eex = 0
Ein = -70
Vt = -54
gex_0 = 0.015
gin_0 = 0.05
deltat = 0.1
g_in_bar = 0.05
time = np.arange(0.0, delay, deltat)

plt.plot(time, neuron(Vrest, Vt, taum,tau_ex, tau_in, time, deltat, gex_0, gin_0,g_in_bar, Eex, Ein))
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
#%%


