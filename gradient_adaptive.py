#Drift speed: constant gradient, adaptive kernel, correlated noise
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numba import njit,prange
import pandas as pd
import sys

#s = int(sys.argv[1]) #an argument


@njit
def noise(sigma):
   gaussian_n = np.random.normal(0, sigma)
   return gaussian_n

tau = 1     #average running time if c = 0
dt = 0.01
c_0 = 0.5   # average concentration
gradient =0.0001
sigma = 0.25
N_s = 1000# number of trajectories simulated
gamma = 10
v = 1000
a=2.05


@njit(parallel=True)
def v_drift_1(sigma, c_0, dt):
    t_end  = np.zeros(N_s)
    x = np.zeros(N_s)
    t = np.zeros(N_s)
    i_random = np.zeros(N_s)
    c_t = np.zeros(N_s)
    P_tumble = np.zeros(N_s)
    random_s = np.zeros(N_s)
    K = int(100000)
    T = K* dt
    C = int(10/dt)
    list_t = np.zeros(( N_s,K))
    list_c_t = np.zeros(( N_s,K))
    list_l = np.zeros(( N_s,K))
    N_t = np.zeros(N_s)
    integral_c_t = np.zeros(N_s)
    c_0_eff = np.zeros(N_s)
    sign = np.zeros(N_s)
    location = np.zeros(N_s)
    location_start = np.zeros(N_s)
    n_old = np.zeros(N_s)
    n_new = np.zeros(N_s)
    for s in prange(0, N_s):
        c_0_eff[s]=c_0
        n_old[s] = 1/np.sqrt(2*gamma) *sigma *noise(1)
        while t[s] < T:
            sign[s] = np.random.randint(0, 2)
            if sign[s] == 0:
                sign[s] = -1
            x[s] = 0
            while x[s]<1:
                i_random[s] = np.random.uniform(0,1)
                n_new[s] = n_old[s] - gamma *(n_old[s])*dt + sigma *noise(1)*(dt)**(0.5)
                c_t[s] = c_0_eff[s] +sign[s]* v* gradient*dt  + n_new[s]
                integral_c_t[s] = 0
                if N_t[s]>=C:
                    for i in range(0, C):
                        integral_c_t[s] += 10*np.exp(-a* list_t[s, i])* (list_t[s,i]-(a/2) * list_t[s,i]**2)* dt * list_c_t[s,int( N_t[s])-1-i]
                P_tumble[s] = 1/tau *(1- integral_c_t[s])*dt
                if i_random[s]<P_tumble[s]:
                    t_end[s] = t[s]
                    x[s] = 1
                if P_tumble[s]<0:
                    print('pas op')
                location[s] += sign[s]* v * dt
                c_0_eff[s] += sign[s]*v*gradient* dt
                t[s] = t[s]+dt
                print(integral_c_t[s])
                n_old[s] = n_new[s]
                list_t[s, int(N_t[s])] = t[s]
                list_c_t[s, int(N_t[s])] =c_t[s]
                list_l[s, int(N_t[s])] = location[s]
                if N_t[s] == 3*C:
                    location_start[s] =location[s]
                N_t[s] += 1
                if t[s]>=T:
                    break
            x[s]+=0
        t[s]+=0
    v_drift = (location- location_start) /(T -30)
    mean_v = np.mean(v_drift)
    error_v =2* np.std(v_drift)/np.sqrt(N_s)
    return mean_v, error_v, list_t[0], list_l[0]

@njit(parallel=True)
def v_drift_2(sigma, c_0, dt):
    t_end  = np.zeros(N_s)
    x = np.zeros(N_s)
    t = np.zeros(N_s)
    i_random = np.zeros(N_s)
    c_t = np.zeros(N_s)
    P_tumble = np.zeros(N_s)
    random_s = np.zeros(N_s)
    K = int(100000)
    T = K* dt
    C = int(10/dt)
    list_t = np.zeros(( N_s,K))
    list_c_t = np.zeros(( N_s,K))
    list_l = np.zeros(( N_s,K))
    N_t = np.zeros(N_s)
    integral_c_t = np.zeros(N_s)
    integral_random = np.zeros(N_s)
    integral_P_tumble = np.zeros(N_s)
    c_0_eff = np.zeros(N_s)
    sign = np.zeros(N_s)
    n_old = np.zeros(N_s)
    n_new = np.zeros(N_s)
    location = np.zeros(N_s)
    location_start = np.zeros(N_s)
    for s in prange(0, N_s):
        c_0_eff[s] = c_0
        n_old[s] = 1/np.sqrt(2*gamma) *sigma *noise(1)
        while t[s]<T:
            sign[s] = np.random.randint(0, 2)
            if sign[s] == 0:
                sign[s] = -1
            i_random[s] = np.random.uniform(0,1)
            integral_random[s] = np.log(1/i_random[s])
            integral_P_tumble[s] = 0
            x[s] = 0
            while x[s]<1:
                n_new[s] = n_old[s] - gamma *(n_old[s])*dt + sigma *noise(1)*(dt)**(0.5)
                c_t[s] = c_0_eff[s] +sign[s]* v* gradient*dt  + n_new[s]
                integral_c_t[s] =0
                if N_t[s]>C:
                    for i in range(0, C):
                        integral_c_t[s] += 10*np.exp(-a* list_t[s,i])* (list_t[s,i]-(a/2) * list_t[s,i]**2)* dt * list_c_t[s,int(N_t[s])-1-i]
                P_tumble[s] = 1/tau *(1- integral_c_t[s])*dt
                integral_P_tumble[s] = integral_P_tumble[s] + P_tumble[s]
                if integral_random[s] < integral_P_tumble[s]:
                    t_end[s] = t[s]
                    x[s] =1
                if P_tumble[s]<0:
                    print('pas op')
                location[s] += sign[s]* v * dt
                c_0_eff[s] += sign[s]*v*gradient* dt
                t[s] = t[s]+dt
                n_old[s] = n_new[s]
                list_t[s, int(N_t[s])] = t[s]
                list_c_t[s, int(N_t[s])] =c_t[s]
                list_l[s, int(N_t[s])] = location[s]
                if N_t[s] == 3*C:
                    location_start[s] =location[s]
                N_t[s] +=1
                if t[s]>=T:
                    break
            x[s]+=0
        t[s]+=0
    v_drift = (location- location_start) /(T -30)
    #print(location_start)
    #print(location)
    #print(v_drift)
    mean_v = np.mean(v_drift)
    error_v = 2*np.std(v_drift)/np.sqrt(N_s)
    return mean_v, error_v


list_v_1 = []
list_v_2 = []
list_sigma = []
list_error_1 = []
list_error_2 = []
theory = []

for i in np.arange(0, 0.25, 0.05):
#for i in np.arange(0, 0.02, 0.01):
    sigma = i
    list_sigma.append(sigma)
    v_1, error_1, t, location = v_drift_1(sigma, c_0, dt)
    v_2, error_2 = v_drift_2(sigma, c_0,  dt)
    list_v_1.append(v_1/v)
    list_error_1.append(error_1/v)
    list_v_2.append(v_2/v)
    list_error_2.append(error_2/v)
    constant = 10* 1/27*v*gradient*tau
    theory.append(constant)
    #plt.plot(t, location)
    #plt.show()

'''
df = pd.DataFrame({'sigma': list_sigma, 'v_1': list_v_1, 'v_1_error': list_error_1, 'v_2': list_v_2, 'v_2_error': list_error_2,'theory': theory})
df.to_csv('Main_output_'+str(s)+'.csv', index=False)  #prints output as an array labeled by 's' in the filename
'''
df = pd.DataFrame({'sigma': list_sigma, 'v_1': list_v_1, 'v_1_error': list_error_1, 'v_2': list_v_2, 'v_2_error': list_error_2})
df.to_csv('data_gradient_adaptive.csv', index=False)

plt.errorbar(list_sigma, list_v_1, yerr=list_error_1, label = 'Walking')
plt.errorbar(list_sigma, list_v_2, yerr=list_error_2, label = 'Gillespie')
#plt.plot(list_sigma, theory, label = 'Theory')
plt.xlabel('$\sigma_c$')
plt.ylabel('$\overline{v}/v_0$')
plt.ylim(bottom = 0)
#plt.title('Relative drift speed: constant gradient, adaptive kernel, correlated noise')
plt.legend()
plt.show()
