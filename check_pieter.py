# This program simulates bacterial chemotaxis in a constant concentration gradient of attractants.
# The output of this program is a plot of the distribution of tumble propensities given the parameters of the paper by Becker, Ten wolde et. all.
# The response function is taken to be an ADAPTIVE KERNEL with the adaptation time around 10 seconds.
# Counting noise is implemented by adding a Gaussian random number to the concentration.
# All bacteria start at position = 0 and time = 0, which is the time of last tumbling.
# The bacteria run and tumble during a given time T.
# To initiate the experiment we have a relataxion time of 30 seconds,
# in the first 10 seconds the bacteria do a random walk unaware of the concentration gradient.
# in the following 20 seconds the bacteria does a bias random walk where the adaptive response is still influenced by the first 10 seconds. 
# After the relaxation time, the tumble propensities are saved. 
# Apart from a different noise implementation and saving tumble propensities, this code is identical to gradient_adaptive.py
# so for extra comments I hereby refer back to this code. 
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numba import njit,prange
import pandas as pd

# Define the parameters as given by the paper of Becker, Ten wolde et. all.
tau =1              # average running time if c = 0 in seconds
dt = 0.002          # time step of simulations in seconds
c_0 = 1             # average concentration (unitless)
gradient = 0.001    # increase in concentration per meter micro 1/m
sigma = 2           # noise amplitude (unitless)
N_s = 5             # number of bacteria simulated
v = 10              # running speed of bacteria in micro m/s
a=2.05              # constant used for the adaptive kernel in 1/s
A=200               # sensitivity of adaptive kernel in 1/s      

# Define additional parameters
K = 100000          # the number of time steps taken in the experiment
T = int(K* dt)      # the total time duration of the experiment
C = int(10/dt)      # the number of time steps taken in 10 seconds

@njit
def noise(sigma):
   gaussian_n = np.random.normal(0, sigma)
   return gaussian_n

@njit(parallel=True)
def v_drift_1(sigma, c_0, dt):
    t_end  = np.zeros(N_s)
    x = np.zeros(N_s)
    t = np.zeros(N_s)
    i_random = np.zeros(N_s)
    c_t = np.zeros(N_s)
    P_tumble = np.zeros(N_s)
    random_s = np.zeros(N_s)
    list_t = np.zeros(( N_s,K+1))
    list_c_t = np.zeros(( N_s,K+1))
    list_l = np.zeros(( N_s,K+1))
    N_t = np.zeros(N_s)
    integral_c_t = np.zeros(N_s)
    c_0_eff = np.zeros(N_s)
    sign = np.zeros(N_s)
    location = np.zeros(N_s)
    location_start = np.zeros(N_s)
    list_p = np.zeros((N_s, K+1))       # tumble propensities at all time steps
    for s in prange(0, N_s):
        c_0_eff[s]=c_0
        while t[s] < T:
            x[s]=0
            sign[s] = np.random.randint(0, 2)
            if sign[s] == 0:
                sign[s] = -1
            while x[s]<1:
                i_random[s] = np.random.uniform(0,1)

                # the noise is given by a Gaussian number with noise amplitude sigma
                c_t[s] = c_0_eff[s] +sign[s]* v* gradient*dt +  noise(sigma)
                integral_c_t[s] = 0
                if N_t[s]>C:
                    for i in range(0, C):
                        integral_c_t[s] += A*np.exp(-a* list_t[s, i])* (list_t[s,i]-(a/2) * list_t[s,i]**2)* dt * list_c_t[s,int( N_t[s])-1-i]
                P_tumble[s] = 1/tau *(1- integral_c_t[s])*dt
                if i_random[s]<P_tumble[s]:
                    t_end[s] = t[s]
                    x[s] = 1
                location[s] += sign[s]* v * dt
                if N_t[s] == 3*C:
                    location_start[s] =location[s] 
                c_0_eff[s] += sign[s]*v*gradient* dt
                t[s] = t[s]+dt
                list_t[s, int(N_t[s])] = t[s]
                list_c_t[s, int(N_t[s])] =c_t[s]
                list_l[s, int(N_t[s])] = location[s]

                # After the relaxation time, save all tumble propensities
                if N_t[s]> 3*C:
                    list_p[s, int(N_t[s]-3*C)] = P_tumble[s]/dt
                N_t[s]+=1
                if t[s]>=T:
                    break
            x[s]+=0
        t[s]+=0
    v_drift = (location_start-location) /(T -30)
    mean_v = np.mean(v_drift)
    error_v = np.std(v_drift)/np.sqrt(N_s)
    return mean_v, error_v, list_t[0], list_l[0], list_p

@njit(parallel=True)
def v_drift_2(sigma, c_0, dt):
    t_end  = np.zeros(N_s)
    x = np.zeros(N_s)
    t = np.zeros(N_s)
    i_random = np.zeros(N_s)
    c_t = np.zeros(N_s)
    P_tumble = np.zeros(N_s)
    random_s = np.zeros(N_s)
    list_t = np.zeros(( N_s,K+1))
    list_c_t = np.zeros(( N_s,K+1))
    list_l = np.zeros(( N_s,K+1))
    N_t = np.zeros(N_s)
    integral_c_t = np.zeros(N_s)
    integral_random = np.zeros(N_s)
    integral_P_tumble = np.zeros(N_s)
    c_0_eff = np.zeros(N_s)
    sign = np.zeros(N_s)
    location = np.zeros(N_s)
    location_start = np.zeros(N_s)
    list_p = np.zeros((N_s, K+1-3*C))      # tumble propensities at all time steps
    for s in prange(0, N_s):
        c_0_eff[s] = c_0
        while t[s]<T:
            sign[s] = np.random.randint(0, 2)
            if sign[s] == 0:
                sign[s] = -1
            i_random[s] = np.random.uniform(0,1)
            integral_random[s] = np.log(1/i_random[s])
            integral_P_tumble[s] = 0
            x[s] = 0
            while x[s]<1:
                c_t[s] = c_0_eff[s] +sign[s]* v* gradient* dt +  noise(sigma)
                integral_c_t[s] = 0
                if N_t[s]>C:
                    for i in range(0, C):
                        integral_c_t[s] +=A* np.exp(-a* list_t[s,i])* (list_t[s,i]-(a/2) * list_t[s,i]**2)* dt * list_c_t[s,int(N_t[s])-1-i]
                P_tumble[s] = 1/tau *(1- integral_c_t[s])*dt
                integral_P_tumble[s] = integral_P_tumble[s] + P_tumble[s]
                if integral_random[s] < integral_P_tumble[s]:
                    t_end[s] = t[s]
                    x[s] =1
                location[s] += sign[s]* v * dt
                if N_t[s] == 3*C:
                    location_start[s] =location[s]
                c_0_eff[s] += sign[s]*v*gradient* dt
                t[s] = t[s]+dt
                list_t[s, int(N_t[s])] = t[s]
                list_c_t[s, int(N_t[s])] =c_t[s]
                list_l[s, int(N_t[s])] = location[s]

                # After the relaxation time, save all tumble propensities
                if N_t[s]> 3*C:
                    list_p[s, int(N_t[s])-3*C] = P_tumble[s]/dt
                N_t[s]+=1
                if t[s]>= T:
                    break
            x[s]+=0
        t[s]+=0
    v_drift =  (location_start-location) /(T -30)
    mean_v = np.mean(v_drift)
    error_v = np.std(v_drift)/np.sqrt(N_s)
    return mean_v, error_v, list_p

# Combine all tumble propensities in one list
v_1, error_1,t, location, list_p = v_drift_1(sigma, c_0, dt)
list_final = []
for i in range(0, N_s):
    for j in range(0, 85000):
        list_final.append(list_p[i,j])

# Plot the distribution of tumble propensities
plt.hist(list_final)
plt.title('distribution of P_tumble')
plt.xlabel('P_tumble')
plt.show()

# Save data
df = pd.DataFrame({'list_p': list_final})
df.to_csv('distribution_tumble_propensity.csv', index=False)

