import numpy as np
from numba import njit,prange

@njit
def uniformfunc():
    return np.random.uniform()

@njit(parallel=True)
def printrandom():
    N=10000000000
    psis=np.zeros(N)
    for i in prange(N):
        psis[i]=uniformfunc()
        if i==500 or i==800:
            print(i,psis[i])
    return

printrandom()
