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
