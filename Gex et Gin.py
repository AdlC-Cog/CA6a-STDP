import numpy as np
import matplotlib.pyplot as plt

###Initialisation
tau_ex = 5
tau_in = 5 

###Time variables
deltat = 0.1
duree = 10
t = np.linspace (0, duree, int(duree//deltat))

###g_ex and g_in arrays
g_ex = np.zeros( len(t) )
g_ex[0]= 0.015
g_in = np.zeros( len(t) )
g_in[0]=0.05

###g_ex and g_in propagation
for k in range(int(duree//deltat)-1):
    g_ex[k+1]= g_ex[k] + deltat*(-g_ex[k]/tau_ex)
    
for k in range(int(duree//deltat)-1):
    g_in[k+1]= g_in[k] + deltat*(-g_in[k]/tau_in)


###Visualisation
plt.plot(t, g_in, label = 'Simulation')

plt.xlabel('$t$ (ms)')
plt.ylabel('$g_in$ (ms)')

plt.legend(loc = 4)

plt.show()