#!/usr/bin/env python
# coding: utf-8

# # M2Pi NRC Swarming ODE Model

# In[1]:


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# In[27]:


q,m,eda = 3.52*10**(-6),1.12*10**(-5),1
def sgn (v):
    if v>0:
        return (1)
        if v<0:
            return (-1)
def model (z,t):
    v = z0[0]
    x = z0[1]
    dvdt = (-sgn(v)*(q/m)*v**2)-x*(eda/m)
    dxdt = v
    dzdt = [dvdt,dxdt]
    return dzdt

z0=[0,0]

t = np.linspace(0,100,100)

z = odeint(model,z0,t)

plt.plot(t,z[:,0])
plt.ylabel('velocity (m/s)')
plt.xlabel('time (s)')
#plt.legend(loc='best')
plt.show()


# In[ ]:




