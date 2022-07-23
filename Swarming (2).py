#!/usr/bin/env python
# coding: utf-8

# # M2Pi NRC Swarming ODE Model

# In[1]:


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.optimize import fsolve


# In[2]:


def swarm_model (z,t):
    # params
    q = 3.52*10**(-6)
    m = 1.12*10**(-5)
    eta = 1
    
    # assign each ODE to a vector element
    x = z[0]
    v = z[1]
    
    #define ODEs
    
    dxdt = v
    dvdt = (-np.sign(v)*(q/m)*v**2)-x*(eta/m)
    
    return[dxdt,dvdt]

# initial conditions
z0 = [0,0]

# declare a time vector (time window)
t = np.linspace(0,20,100)
z = odeint(swarm_model,z0,t)

x = z[:,0]
v = z[:,1]

# plot the results

plt.plot(t,z[:,1])
plt.ylabel('velocity (m/s)')
plt.xlabel('time (s)')
#plt.legend(loc='best')
plt.show()

plt.plot(t,z[:,0])
plt.ylabel('position')
plt.xlabel('time (s)')
#plt.legend(loc='best')
plt.show()


# In[8]:


def swarm_model (z,t):
    # params
    q = 3.52*10**(-6)
    m = 1.12*10**(-5)
    eta = 1
    
    # assign each ODE to a vector element
    x = z[0]
    v = z[1]
    
    #define ODEs
    
    dxdt = v
    dvdt = (-np.sign(v)*(q/m)*v**2)-x*(eta/m)
    
    return[dxdt,dvdt]

# initial conditions
z0 = [1,1]

# declare a time vector (time window)
t = np.linspace(0,20,1000)
z = odeint(swarm_model,z0,t)

x = z[:,0]
v = z[:,1]

# plot the results

plt.plot(t,z[:,1])
plt.ylabel('velocity (m/s)')
plt.xlabel('time (s)')
#plt.legend(loc='best')
plt.savefig('v_vs_t_swarming.png',dpi=300)
plt.show()

plt.plot(t,z[:,0])
plt.ylabel('position (m)')
plt.xlabel('time (s)')
#plt.legend(loc='best')
plt.savefig('x_vs_t_swarming.png',dpi=300)
plt.show()


# ## Steady-state analysis

# In[4]:


# give it an initial guess. We know that the SS should be 0,0, 
# so let's give it that guess.

w0 = [0,0]

SS = fsolve(swarm_model, w0,t)

print(SS)


# # Model with Wind

# In[20]:


def swarm_model (z,t):
    # params
    # q = 3.52*10**(-6)
    # m = 1.12*10**(-5)
    # eta = 1
    w = 1
    
    # assign each ODE to a vector element
    x = z[0]
    v = z[1]
    
    #define ODEs
    
    dxdt = v+w
    dvdt = (-np.sign(v+w)*(v+w)**2)-x
    
    return[dxdt,dvdt]

# initial conditions
z0 = [0,0]

# declare a time vector (time window)
t = np.linspace(0,200,1000)
z = odeint(swarm_model,z0,t)

x = z[:,0]
v = z[:,1]

# plot the results

plt.plot(t,z[:,1])
plt.ylabel('velocity (m/s)')
plt.xlabel('time (s)')
plt.savefig('v_vs_t_swarming_wind_1.png',dpi=300)
#plt.legend(loc='best')
plt.show()

plt.plot(t,z[:,0])
plt.ylabel('position')
plt.xlabel('time (s)')
plt.savefig('x_vs_t_swarming_wind_1.png',dpi=300)
#plt.legend(loc='best')
plt.show()


# In[19]:


def swarm_model (z,t):
    # params
    # q = 3.52*10**(-6)
    # m = 1.12*10**(-5)
    # eta = 1
    w = -1
    
    # assign each ODE to a vector element
    x = z[0]
    v = z[1]
    
    #define ODEs
    
    dxdt = v+w
    dvdt = (-np.sign(v+w)*(v+w)**2)-x
    
    return[dxdt,dvdt]

# initial conditions
z0 = [0,0]

# declare a time vector (time window)
t = np.linspace(0,200,1000)
z = odeint(swarm_model,z0,t)

x = z[:,0]
v = z[:,1]

# plot the results

plt.plot(t,z[:,0])
plt.ylabel('position')
plt.xlabel('time (s)')
plt.savefig('x_vs_t_swarming_wind_m1.png',dpi=300)
#plt.legend(loc='best')
plt.show()

plt.plot(t,z[:,1])
plt.ylabel('velocity (m/s)')
plt.xlabel('time (s)')
plt.savefig('v_vs_t_swarming_wind_m1.png',dpi=300)
#plt.legend(loc='best')
plt.show()


# In[15]:


z0 = [0,0]

SS = fsolve(swarm_model, z0,t)

print(SS)


# In[17]:


def swarm_model (z,t):
    # params
    # q = 3.52*10**(-6)
    # m = 1.12*10**(-5)
    # eta = 1
    w = 4
    
    # assign each ODE to a vector element
    x = z[0]
    v = z[1]
    
    #define ODEs
    
    dxdt = v+w
    dvdt = (-np.sign(v+w)*(v+w)**2)-x
    
    return[dxdt,dvdt]

# initial conditions
z0 = [0,0]

# declare a time vector (time window)
t = np.linspace(0,200,1000)
z = odeint(swarm_model,z0,t)

x = z[:,0]
v = z[:,1]

SS = fsolve(swarm_model, z0,t)

print(SS)


# In[ ]:




