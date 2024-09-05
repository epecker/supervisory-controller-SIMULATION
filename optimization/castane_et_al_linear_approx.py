# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 20:14:06 2024

@author: epecker
"""
import numpy as np
from scipy.signal import lti, lsim, cont2discrete, place_poles
from scipy.linalg import expm, solve_discrete_lyapunov
import timeit

# Parameters
_hfbs = 12   # h supervisory task
_Cfbs = 0.05 # C supervisory task
_C    = 0.05 # Worst Case Execution Time (WCET) for all tasks

# Initial solution (initial guess)
_h0   = 0.17
_h1   = 0.17
_h2   = 0.23
_h3   = 0.27

# Weighting coefficients
_alfa4 = 0.0020 # 0.0035
_alfa3 = 0.2000
_alfa1 = (1 - _alfa3 - _alfa4)/2
_alfa2 = _alfa1
_alfa = np.array([_alfa1,_alfa2,_alfa3,_alfa4])

# noise parameters
_noise_var = 0.1
_sigma2_qs = 10
_sigma2 = np.array([_noise_var,_noise_var,_noise_var,_sigma2_qs])

# Error values from old logs
# input (para 'Queue Server' linealizado alrededor de _xref=10)
#_e_roll  = np.array([[-0.318562],[-0.061728]])# task0: x1 = -0.318562, x2 = -0.061728
#_e_pitch = np.array([[-0.061795],[0.026615]]) # task1: x1 = -0.061795, x2 = 0.026615
#_e_yaw   = np.array([[-0.031719],[0.071972]]) # task2: x1 = -0.031719, x2 = 0.071972
#_e_queue = np.array([[6.014110],[0]])         # task3: x1 =  6.014110, x2 = 0.000000
# input (para 'Queue Server' linealizado alrededor de _xref=60)
#_e_roll  = np.array([[ 0.0472908],[-0.0300783]]) #
#_e_pitch = np.array([[ 0.0253079],[ 0.0696186]]) #
#_e_yaw   = np.array([[ 0.0077565],[-0.0634158]]) #
#_e_queue = np.array([[-0.0420311],[0]])          #
# input (para 'Queue Server' linealizado alrededor de _xref=50)
_e_roll  = np.array([[-0.528910],[-0.284831]]) # 
_e_pitch = np.array([[-0.072620],[ 0.124293]]) # 
_e_yaw   = np.array([[-0.011174],[ 0.077356]]) # 
_e_queue = np.array([[-3.987073],[0]])         #
_e = np.hstack((_e_roll, _e_pitch, _e_yaw, _e_queue))

# Physical Systems
CPS = {}

# UAV Roll
Ixx = 4.856e-3 # moment of inertia [kg/m^2]
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1/Ixx]])
C = np.array([[1, 0]])
D = np.array(0)
pfc = np.array([-10+1j, -10-1j]) # closed-loop poles
CPS['UAV Roll'] = {'A': A, 'B': B, 'C': C, 'D': D, 'pfc': pfc}

# UAV Pitch
Iyy = 4.856e-3 # moment of inertia [kg/m^2]
A = np.array([[0, 1], [0, 0]])
B = np.array([[0],[1/Iyy]])
C = np.array([[1, 0]])
D = np.array(0)
pfc = np.array([-10+1j, -10-1j]) # closed-loop poles
CPS['UAV Pitch'] = {'A': A, 'B': B, 'C': C, 'D': D, 'pfc': pfc}

# UAV Yaw
Izz = 8.801e-3 # moment of inertia [kg/m^2]
A = np.array([[0, 1],[0, 0]])
B = np.array([[0],[1/Izz]])
C = np.array([[1, 0]])
D = np.array(0)
pfc = np.array([-7+1j, -7-1j]) # closed-loop poles
CPS['UAV Yaw'] = {'A': A, 'B': B, 'C': C, 'D': D, 'pfc': pfc}

# Queue Server
# Kihl, M., Robertsson, A., & Wittenmark, B. (2003, July). Analysis of 
# admission control mechanisms using non-linear control theory. In 
# Proceedings of the Eighth IEEE Symposium on Computers and Communica-
# tions. ISCC 2003 (pp. 1306-1311). IEEE.
mu1 = 10
mu2 = 60
alpha = 0.38
media = alpha/mu1 + (1-alpha)/mu2
mu = 1/media
varianza = alpha/(mu1**2) + (1-alpha)/(mu2**2) + alpha*(1/mu1-media)**2 + (1-alpha)*(1/mu2-media)**2
C2 = varianza/(media**2)
_xref = 50 # linearization point

a = -mu/(1-C2)*(1-(_xref+C2)/np.sqrt(_xref**2 + 2*C2*_xref + 1))
b = 1
pfc = np.array([-9.0953614, -0.1048428]) # closed-loop poles
# second-order continuous system with one input
A = np.array([[a, 0],[1, 0]])
B = np.array([[b], [0]])
C = np.array([[1, 0]])
D = np.array(0)
CPS['Queue Server'] = {'A': A, 'B': B, 'C': C, 'D': D, 'pfc': pfc}

# Integrals involving the Exponential matrix
def discretelossfunction(Ac, Bc, Q1c, Q2c, Q12c, h):
    # folowing method is due to Van Loan:
    # C. Van Loan, "Computing integrals involving the matrix exponential," in IEEE Transactions on Automatic Control, vol. 23, no. 3, pp. 395-404, June 1978.
    # It can also be checked in: Franklin & Powell, "Digital Control of Dynamic Systems", p.380.
    Nx = Ac.shape[0] # size(Ac,1)
    Nu = Bc.shape[1] # size(Bc,2)
    n  = Nx + Nu
    # Q12c = np.zeros((Nx,Nu))
    Za = np.zeros((Nx,Nx)) 
    Zb = np.zeros((Nx,Nu)) 
    Zu = np.zeros((Nu,Nu))
    # debug
    # print('Ac: {}, Zb: {}, Q1c: {}, Q12c: {}'.format(Ac, Zb, Q1c, Q12c))
    M = np.vstack((np.hstack((-Ac.T, Zb, Q1c,    Q12c)),
                   np.hstack((-Bc.T, Zu, Q12c.T, Q2c)),
                   np.hstack(( Za,   Zb, Ac,     Bc)),
                   np.hstack(( Zb.T, Zu, Zb.T,   Zu)))
                  )
    phi   = expm(M*h)
    # debug
    # print('phi: {}'.format(phi))
    phi12 = phi[0:n,n:2*n] # n = Nx+Nu
    phi22 = phi[n:2*n,n:2*n]
    QQ    = (phi22.T).dot(phi12)
    QQ    = (QQ+QQ.T)/2 # Make sure QQ is symmetric
    Q1    = QQ[0:Nx,0:Nx]
    Q2    = QQ[Nx:n,Nx:n]
    Q12   = QQ[0:Nx,Nx:n]
    Phi   = phi22[0:Nx,1:Nx] # Ad
    Gamma = phi22[0:Nx,Nx:n] # Bd

    return Q1, Q2, Q12 


# Bounds
_hmin = 0.05 # min val
_hmax = 0.5  # max val
_H = 1e-4
_hvalues = np.arange(_hmin, _hmax+_H, _H)

_Ufbs = _Cfbs/_hfbs
_n = len(CPS.keys()) # number of systems considered in the optimization


# Algorithm from "Resource Management for Control Tasks Based on the Transient 
# Dynamics of Closed-Loop Systems" available at 
# https://lucris.lub.lu.se/ws/portalfiles/portal/6072669/8866196.pdf


# The required calculations to minimize the objective function in (19) can be 
# divided into one off-line part and one on-line part. Given a set of sampling 
# periods and associated controllers, the factors S(h) and 
# J¯(h) = tr S(h)R1(h) + Jv(h) can be calculated off-line for each plant. 
# (These factors will be generically called Si,j , where i identifies the task 
# and j the sampling period).

## Algorithm
# n = number of tasks
# h = [hmin_1, hmin_2, ..., hmin_n]
# C = [C1, C2,..., Cn]
# x = [x1, x2,..., xn]
# while sum(Ck/hk) > Usp - Ufbs do
#     lower_cost_task = 1
#     for i = 1 to n do
#         if h(i) + H <= hmax_i then
#             ΔJ(i) = J(h(i)+H, x(i)) − J(h(i), x(i))
#             if ΔJ(i) < ΔJ(lower_cost_task) then
#                 lower_cost_task = i
#             end if
#         end if
#     end for
#     h(lower_cost_task) = h(lower_cost_task) + H
# end while
# return h



###############################################################################
# APPROX 1: This is a first approach of previous algorithm. Note that the cost
#           function terms are calculated as long as the algorithm is executed.
###############################################################################

# Objective function to be minimised with respect to sampling times h(j)
def J(hh,ee,i):
    # debug
    # print("Received values: hh={}, ee={}, i={}".format(hh,ee,i))
    # continuous-time system
    k = list(CPS.keys())[i]
    sysc = lti(CPS[k]['A'],
               CPS[k]['B'],
               CPS[k]['C'],
               CPS[k]['D'])
        
    # debug
    # print('sysc: {}'.format(sysc))
    # print('h[i]: {}'.format(hh))
        
    # discrete-time system
    sysd = sysc.to_discrete(hh)
    pfd  = np.exp(CPS[k]['pfc'] * hh) # discrete-time poles
    res  = place_poles(sysd.A, sysd.B, pfd) # pole placement
    Kfd  = res.gain_matrix
        
    r, n = CPS[k]['C'].shape
    n, m = CPS[k]['B'].shape
    Q1c  = (CPS[k]['C'].T).dot(CPS[k]['C']) # eye(n,n)
    Q2c  = np.zeros((m,m))
    Q12c = np.zeros((n,m))
    Q0c  = np.zeros((n,n))
    R1c  = _sigma2[i] * np.eye(n)

    Q1, Q2, Q12 = discretelossfunction(sysc.A, sysc.B, Q1c, Q2c, Q12c, hh)
    R1, R2, R12 = discretelossfunction(sysc.A, sysc.B, R1c, np.zeros((m,m)), np.zeros((n,m)), hh)
        
    # Debug
    # print('sysd.A:{} , sysd.B: {}, Kfd): {}'.format(sysd.A, sysd.B, Kfd))
    # print('Q1: {}, Q12: {}, Kfd: {}, Q2: {}'.format(Q1, Q12, Kfd, Q2))
    Sl = -solve_discrete_lyapunov( (sysd.A - sysd.B.dot(Kfd)).T,
                                 -(Q1 - Q12.dot(Kfd)-(Kfd.T).dot(Q12.T) + (Kfd.T).dot(Q2.dot(Kfd)))
                                )

    Ji = (ee.T).dot(Sl.dot(ee)) + _hfbs/hh * np.trace(Sl.dot(R1))
    Ji  = _alfa[i] * Ji

    return Ji


#t_ini = timeit.default_timer()
#_hsol = np.array([_hmin, _hmin, _hmin, _hmin])
#k = 0
##rng = np.random.default_rng()
##tasks = np.arange(0,_n)
#while np.sum(_C/_hsol) > 1 - _Ufbs:
#    lower_cost_task = 0 # start from control task 0
#    DJ = np.zeros(_n)
#    for i in range(0,_n):
#    # for i in reversed(range(0,_n)):
#    # for i in tasks:
#        if _hsol[i] + _H <= _hmax:
#            DJ[i] = J(_hsol[i] + _H, _e[:,i], i) - J(_hsol[i], _e[:,i], i)
#            if DJ[i] < DJ[lower_cost_task]:
#                lower_cost_task = i
#    _hsol[lower_cost_task] = _hsol[lower_cost_task] + _H
#    print('iteration: {}, lower_cost_task: {}'.format(k,lower_cost_task))
#    k = k + 1
#    # rng.shuffle(tasks) # random shuffle of tasks
#t_end = timeit.default_timer()
#elapsed = t_end - t_ini
#
#_U = np.sum(_C/_hsol) + _Ufbs
#print('Solution: {}, Utilization: {}, elapsed time: {} s'.format(_hsol,_U,elapsed))
## Solution: [0.1117 0.1452 0.4821 0.5   ], Utilization: 0.9998597802352749

###############################################################################
# APPROX 2: In this case some terms of the cost function are pre-calculated.
###############################################################################

# Objective function to be minimised with respect to sampling times h(j)
def J2(hh,i):
    # debug
    # print("Received values: hh={}, ee={}, i={}".format(hh,ee,i))
    # continuous-time system
    k = list(CPS.keys())[i]
    sysc = lti(CPS[k]['A'],
               CPS[k]['B'],
               CPS[k]['C'],
               CPS[k]['D'])
        
    # debug
    # print('sysc: {}'.format(sysc))
    # print('h[i]: {}'.format(hh))
        
    # discrete-time system
    sysd = sysc.to_discrete(hh)
    pfd  = np.exp(CPS[k]['pfc'] * hh) # discrete-time poles
    res  = place_poles(sysd.A, sysd.B, pfd) # pole placement
    Kfd  = res.gain_matrix
        
    r, n = CPS[k]['C'].shape
    n, m = CPS[k]['B'].shape
    Q1c  = (CPS[k]['C'].T).dot(CPS[k]['C']) # eye(n,n)
    Q2c  = np.zeros((m,m))
    Q12c = np.zeros((n,m))
    Q0c  = np.zeros((n,n))
    R1c  = _sigma2[i] * np.eye(n)

    Q1, Q2, Q12 = discretelossfunction(sysc.A, sysc.B, Q1c, Q2c, Q12c, hh)
    R1, R2, R12 = discretelossfunction(sysc.A, sysc.B, R1c, np.zeros((m,m)), np.zeros((n,m)), hh)
        
    # Debug
    # print('sysd.A:{} , sysd.B: {}, Kfd): {}'.format(sysd.A, sysd.B, Kfd))
    # print('Q1: {}, Q12: {}, Kfd: {}, Q2: {}'.format(Q1, Q12, Kfd, Q2))
    Sl = -solve_discrete_lyapunov( (sysd.A - sysd.B.dot(Kfd)).T,
                                 -(Q1 - Q12.dot(Kfd)-(Kfd.T).dot(Q12.T) + (Kfd.T).dot(Q2.dot(Kfd)))
                                )

    Ji = _hfbs/hh * np.trace(Sl.dot(R1))

    return Ji, Sl


# the dictionary JJ saves the cost functions terms
JJ = {}
for k,j in zip(CPS.keys(), range(0,_n)):
    JJ[k] = {}
    for i in _hvalues:
        JJ[k][round(i,6)] = J2(i,j)

N=10
elapsed = np.array([])
for r in range(0,N):
    t_ini = timeit.default_timer()
    _hsol = np.array([_hmin, _hmin, _hmin, _hmin])
    k = 0
    while np.sum(_C/_hsol) > 1 - _Ufbs:
        lower_cost_task = 0 # start from control task 0
        DJ = np.zeros(_n)    
        for key, i in zip(CPS.keys(),range(0,_n)):
            if _hsol[i] + _H <= _hmax:
                # DJ[i] = J(_hsol[i] + _H, _e[:,i], i) - J(_hsol[i], _e[:,i], i)
                _hact = round(_hsol[i],6)    # use round() due to representation error
                _hnew = round(_hsol[i]+_H,6) # use round() due to representation error
                Jval1 = JJ[key][_hnew][0]
                Sval1 = JJ[key][_hnew][1]
                Jval2 = JJ[key][_hact][0]
                Sval2 = JJ[key][_hact][1]
                ee = _e[:,i]
                DJ[i] = _alfa[i]*((ee.T).dot(Sval1.dot(ee)) + Jval1) - \
                        _alfa[i]*((ee.T).dot(Sval2.dot(ee)) + Jval2)
                if DJ[i] < DJ[lower_cost_task]:
                    lower_cost_task = i
        _hsol[lower_cost_task] = _hsol[lower_cost_task] + _H
        # print('iteration: {}, lower_cost_task: {}'.format(k,lower_cost_task))
        k = k + 1
    t_end = timeit.default_timer()
    elapsed = np.append(elapsed, t_end - t_ini)
    _U = np.sum(_C/_hsol) + _Ufbs
    print('Solution: {}, Utilization: {}, elapsed time: {} s'.format(_hsol,_U,elapsed[r]))
    # Solution: [0.1117 0.1452 0.4821 0.5   ], Utilization: 0.9998597802352749

print('Mean elapsed time: {}'.format(np.mean(elapsed)))

# Solution with the non-linear optimization algorithm
# [0.1568084  0.1699403  0.23608204 0.29246725]






