import math

import numpy as np

from fpinns.fraccaputo import fraccaputo_v2


def FracSDOF(m,k,c,dt,F,x0,v0,T,a,tau):
    dt = float(dt)
    Nt = int(np.ceil(T/dt))
    u = np.zeros(Nt)
    t = np.linspace(0, T, Nt)
    u[0] = x0
    u[1] = u[0] + dt*v0 + (F[0] - k*u[0] - c*v0)/(2*m/dt**2)
    tf = int(np.ceil(tau/dt))
    pv = np.arange(tf + 3)
    pv = np.power(pv, 1 - a)
    Weight_vector = (pv[2:] - 2 * pv[1:-1] + pv[:-2])[:tf]
    for n in range(1, Nt - 1):
        if (n < tf):
            wv = Weight_vector[:n].copy()
            wv[n - 1] = wv[n - 1] + pv[n - 1] - pv[n]
            wv = np.flip(wv,0)
            wv = np.concatenate([wv, np.array([1.])])
            wv = wv * (dt ** (-a) / math.gamma(2 - a))
            u[n + 1] = 2 * u[n] - u[n - 1] + (F[n] - k * u[n] - c * np.dot(wv, u[:n + 1])) / (m / dt ** 2)
        else:
            wv = Weight_vector.copy()
            wv[tf - 1] = wv[tf - 1] + pv[tf - 1] - pv[tf]
            wv = np.flip(wv,0)
            wv = np.concatenate([wv, np.array([1.])])
            wv = wv * (dt ** (-a) / math.gamma(2 - a))
            u[n + 1] = 2 * u[n] - u[n - 1] + (F[n] - k * u[n] - c * np.dot(wv, u[n - tf :n+1])) / (m / dt ** 2)
    return u, t

def fdm_no_fractional(m,k,c,dt,F,x0,v0,T):
    dt = float(dt)
    Nt = int(np.ceil(T/dt))
    u = np.zeros(Nt)
    t = np.linspace(0, T, Nt)
    u[0] = x0
    u[1] = u[0] + dt*v0 + (F[0] - k*u[0] - c*v0)/(2*m/dt**2)
    for n in range(1, Nt-1):
        v = (u[n] - u[n-1])/dt
        u[n+1] = 2*u[n] - u[n-1] + (F[n] - k*u[n] - c*v)/(m/dt**2)
    return u, t

def fracSDOF(m,k,c,dt,F,x0,v0,T,a,tau):
    dt = float(dt)
    Nt = int(round(T/dt))
    u = np.zeros(Nt)
    t = np.linspace(0, T, Nt)
    u[0] = x0
    u[1] = u[0] + dt*v0 + (F[0] - k*u[0] - c*v0)/(2*m/dt**2)
    for n in range(1, Nt-1):
        frac=fraccaputo_v2( u[0:n+1], dt, a, tau, n )
        u[n+1]=2*u[n] - u[n-1] + (F[n]-k*u[n]-c*frac)/(m/dt**2)
    return u,t

def ExpSDOF(m, E0, E1, dt, F, x0, v0, T, tauc, tau):
    dt = float(dt)
    Nt = int(np.ceil(T / dt))
    u = np.zeros(Nt)
    t = np.linspace(0, T, Nt)

    u[0] = x0
    u[1] = u[0] + dt*v0 + (F[0] - E0*u[0]) / (2*m/dt**2)        #q0=0, as q=0 for t<=0
    tm = int(np.ceil(tau/dt))

    for n in range(1, Nt - 1):
        start = max(0, n - tm)
        v_hist = (u[start:n+1] - u[start-1:n])/dt if start > 0 else np.concatenate((np.array([v0]), np.diff(u[start:n+1])/dt))

        # Time difference between current time and history points
        s_idx = np.arange(start, n + 1)
        lag = (n - s_idx)*dt
        # Exponential kernel
        kernel = (1/tauc)*np.exp(-lag/tauc)
        # Exponential memory term
        q = np.trapz(kernel*v_hist, dx=dt)
        # FDM update
        u[n+1] = 2*u[n] - u[n-1] + (F[n] - E0*u[n] - E1*q)/(m/dt**2)

    return u, t
