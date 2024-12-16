#Drift speed:1 run. constant gradient, delta kernel, correlated noise
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numba import njit,prange
import pandas as pd

@njit
def noise(sigma):
   gaussian_n = np.random.normal(0, sigma)
   return gaussian_n

tau = 1     #average running time if c = 0
dt = 0.001
c_0 = 0.5   # average concentration
gradient =0.0001
sigma = 0.25
N_s = 40000000  # number of trajectories simulated
gamma= 100
v = 10

@njit(parallel=True)
def t_run_1(sigma, c_0, gamma, dt, sign):
    t_end  = np.zeros(N_s)
    x = np.zeros(N_s)
    t = np.zeros(N_s)
    n_old = np.zeros(N_s)
    n_new = np.zeros(N_s)
    i_random = np.zeros(N_s)
    c_t = np.zeros(N_s)
    P_tumble = np.zeros(N_s)
    for s in prange(0, N_s):
        t[s] = dt
        n_old[s] = 1/np.sqrt(2*gamma) *sigma *noise(1)
        while x[s]<1:
            i_random[s] = np.random.uniform(0,1)
            n_new[s] = n_old[s] - gamma *(n_old[s])*dt + sigma *noise(1)*(dt)**(0.5)
            c_t[s] = c_0 + sign * v* gradient* t[s] + n_new[s]
            P_tumble[s] = 1/tau *(1- c_t[s])*dt
            if i_random[s]<P_tumble[s]:
                t_end[s] = t[s]
                x[s] = 1
            if P_tumble[s] <0:
                print('pas op')
            t[s] = t[s]+dt
            n_old[s] = n_new[s]
        x[s]+=0
    return t_end

@njit(parallel=True)
def t_run_2(sigma, c_0, gamma, dt, sign):
    t_end = np.zeros(N_s)
    x = np.zeros(N_s)
    t = np.zeros(N_s)
    n_old = np.zeros(N_s)
    n_new = np.zeros(N_s)
    i_random = np.zeros(N_s)
    c_t = np.zeros(N_s)
    P_tumble = np.zeros(N_s)
    integral_random = np.zeros(N_s)
    integral_P_tumble = np.zeros(N_s)
    for s in prange(0, N_s):
        t[s] = dt
        i_random[s] = np.random.uniform(0,1)
        integral_random[s] = np.log(1/i_random[s])
        n_old[s] = 1/np.sqrt(2*gamma)* sigma *noise(1)
        while x[s]<1:
            n_new[s] = n_old[s] - gamma *(n_old[s])*dt + sigma *noise(1)* (dt)**(0.5)
            c_t[s] = c_0  + sign * v* gradient *t[s] + n_new[s]
            P_tumble[s] = 1/tau *(1- c_t[s])*dt
            integral_P_tumble[s] = integral_P_tumble[s] + P_tumble[s]
            if integral_random[s] < integral_P_tumble[s]:
                t_end[s] = t[s]
                x[s] = 1
            if P_tumble[s]<0:
                print('pas op')
            t[s]= t[s]+dt
            n_old[s] = n_new[s]
        x[s]+=0
    return t_end

@njit
def v_drift_1(sigma, c_0, gamma, dt):
    t_end_up  = t_run_1(sigma, c_0, gamma, dt, 1)
    t_end_down = t_run_1(sigma, c_0, gamma, dt , -1)
    t_average_up = np.mean(t_end_up)
    t_average_down = np.mean(t_end_down)
    frac_up = t_average_up/(t_average_up+t_average_down)
    frac_down = t_average_down/(t_average_up + t_average_down)
    #v_drift = v*(t_end_up * frac_up - t_end_down* frac_down)/ (t_end_up+t_end_down)
    v_drift =v*(t_end_up - t_end_down)/ (t_end_up+t_end_down)
    mean_v = np.mean(v_drift)
    error_v = 2* np.std(v_drift)/np.sqrt(N_s)
    return mean_v, error_v

@njit
def v_drift_2(sigma, c_0, gamma, dt):
    t_end_up  = t_run_2(sigma, c_0, gamma, dt, 1)
    t_end_down = t_run_2(sigma, c_0, gamma, dt , -1)
    t_average_up = np.mean(t_end_up)
    t_average_down = np.mean(t_end_down)
    frac_up = t_average_up/(t_average_up+t_average_down)
    frac_down = t_average_down/(t_average_up + t_average_down)
    #v_drift = v*(t_end_up * frac_up - t_end_down* frac_down)/ (t_end_up+t_end_down)
    v_drift =v*(t_end_up - t_end_down)/ (t_end_up+t_end_down)
    mean_v = np.mean(v_drift)
    error_v = 2* np.std(v_drift)/np.sqrt(N_s)
    return mean_v, error_v

list_v_1 = []
list_v_2 = []
list_sigma = []
list_error_1 = []
list_error_2 = []
theory = []
for i in np.arange(0, 0.25, 0.05):
    sigma = i
    list_sigma.append(sigma)
    v_1, error_1 = v_drift_1(sigma, c_0, gamma, dt)
    v_2, error_2 = v_drift_2(sigma, c_0, gamma, dt)
    list_v_1.append(v_1/v)
    list_error_1.append(error_1/v)
    list_v_2.append(v_2/v)
    list_error_2.append(error_2/v)
    constant = v * gradient* tau
    theory.append(constant)


df = pd.DataFrame({'sigma': list_sigma, 'v_1': list_v_1, 'v_1_error': list_error_1, 'v_2': list_v_2, 'v_2_error': list_error_2,'theory': theory})
df.to_csv('v_drift_delta.csv', index=False)

plt.errorbar(list_sigma, list_v_1, yerr=list_error_1, label = 'Walking')
plt.errorbar(list_sigma, list_v_2, yerr=list_error_2, label = 'Gillespie')
plt.plot(list_sigma, theory, label = 'Theory')
plt.xlabel('counting noise')
plt.ylabel('relative drift speed')
plt.ylim(bottom = 0)
#plt.title('Drift speed: constant gradient, delta kernel, correlated noise')
plt.legend()
plt.show()

