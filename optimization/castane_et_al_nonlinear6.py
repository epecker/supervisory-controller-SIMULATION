from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
import numpy as np
from scipy.signal import lti, lsim, cont2discrete, place_poles
from scipy.linalg import expm, solve_discrete_lyapunov
from time import time
import timeit # more accurate than time.time(). See: stackoverflow.com/questions/17579357/time-time-vs-timeit-timeit

# usage
# time python3 castane_et_al_nonlinear6.py 

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
    #print('phi: {}'.format(phi))
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

# Objective function to be minimised with respect to sampling times h(j)
def F(hh):

    # sampling times vector h has sampling times of every system
    # hh = [h1,h2,h3,h4]
    nsys = len(hh) # number of systems considered in the optimization

    J = 0
    for i, j in zip(CPS, range(0,nsys)):
        # continuous-time system
        sysc = lti(CPS[i]['A'],
                   CPS[i]['B'],
                   CPS[i]['C'],
                   CPS[i]['D'])
        
        # debug
        # print('sysc: {}'.format(sysc))
        # print('h[j]: {}'.format(hh[j]))
        
        # discrete-time system
        sysd = sysc.to_discrete(hh[j])
        pfd  = np.exp(CPS[i]['pfc'] * hh[j]) # discrete-time poles
        res  = place_poles(sysd.A, sysd.B, pfd) # pole placement
        Kfd  = res.gain_matrix
        
        r, n = CPS[i]['C'].shape
        n, m = CPS[i]['B'].shape
        Q1c  = (CPS[i]['C'].T).dot(CPS[i]['C']) # eye(n,n)
        Q2c  = np.zeros((m,m))
        Q12c = np.zeros((n,m))
        Q0c  = np.zeros((n,n))
        R1c  = _sigma2[j] * np.eye(n)

        Q1, Q2, Q12 = discretelossfunction(sysc.A, sysc.B, Q1c, Q2c, Q12c, hh[j])
        R1, R2, R12 = discretelossfunction(sysc.A, sysc.B, R1c, np.zeros((m,m)), np.zeros((n,m)), hh[j])
        
        # Debug
        # print('sysd.A:{} , sysd.B: {}, Kfd): {}'.format(sysd.A, sysd.B, Kfd))
        # print('Q1: {}, Q12: {}, Kfd: {}, Q2: {}'.format(Q1, Q12, Kfd, Q2))
        Sl = -solve_discrete_lyapunov( (sysd.A - sysd.B.dot(Kfd)).T,
                                     -(Q1 - Q12.dot(Kfd)-(Kfd.T).dot(Q12.T) + (Kfd.T).dot(Q2.dot(Kfd)))
                                    )

        Ji = (_e[:,j].T).dot(Sl.dot(_e[:,j])) + _hfbs/hh[j] * np.trace(Sl.dot(R1))
        J  = J + _alfa[j] * Ji

    return J

# Parameters
_hfbs = 12   # h supervisory task
_Cfbs = 0.05 # C supervisory task
_C    = 0.05 # Worst Case Execution Time for all tasks

# Initial solution (initial guess)
_h0   = 0.17
_h1   = 0.17
_h2   = 0.23
_h3   = 0.27
init_sol = np.array([_h0, _h1, _h2, _h3])

# Bounds
_hmin = 0.05 # min val
_hmax = 0.5  # max val 
bnds = Bounds([(_hmin, _hmax) for _ in init_sol])

constr = lambda x: _C/x[0] + _C/x[1] + _C/x[2] + _C/x[3]
# nlc = NonlinearConstraint(constr, 1-_Cfbs/_hfbs, 1-_Cfbs/_hfbs) # non-linear equality
nlc = NonlinearConstraint(constr, 0, 1-_Cfbs/_hfbs) # non-linear inequality

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
#_e_roll  = np.array([[-0.318562],[-0.061728]]) # task0: x1 = -0.318562, x2 = -0.061728
#_e_pitch = np.array([[-0.061795],[0.026615]])  # task1: x1 = -0.061795, x2 = 0.026615
#_e_yaw   = np.array([[-0.031719],[0.071972]])  # task2: x1 = -0.031719, x2 = 0.071972
#_e_queue = np.array([[6.014110],[0]])          # task3: x1 =  6.014110, x2 = 0.000000
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

N = 50
elapsed = np.array([])
for i in range(0,N):
    #t_ini = time()
    t_ini = timeit.default_timer()
    # method = SQLP or Trust-Constr
    res = minimize(F,               # function
               init_sol,        # x0
	       bounds=[(_hmin, _hmax) for _ in init_sol], # bounds 
	       constraints=nlc, # constraints
	       # method="SLSQP"
               method="trust-constr"
               # tol = 1e-9
               )
    #t_end = time()
    t_end = timeit.default_timer()
    elapsed = np.append(elapsed, t_end - t_ini)
    # Results
    # print('Message: {}'.format(res.message))
    print('Solution: {}, elapsed time: {}'.format(res.x, elapsed[i]))

print('Mean elapsed time: {}'.format(np.mean(elapsed)))


_h = res.x

# Check
#_U = _Cfbs/_hfbs;
#n = len(_alfa)
#for j in range(0,n):
#    _U = _U + _C/_h[j]
## _U = _C/res.x[0] + _C/res.x[1] + _C/res.x[2] + _C/res.x[3] + _Cfbs/_hfbs
#print('Utilization: {}'.format(_U))

# solution from Scilab
#_h = np.array([0.162336,0.166985,0.232062,0.289111])
# new discrete-time systems
# for i, j in zip(CPS, range(0,len(_h))):
# 
#     # continuous-time system
#     sysc = lti(CPS[i]['A'],
#                CPS[i]['B'],
#                CPS[i]['C'],
#                CPS[i]['D'])
#     
#     # discrete-time system
#     sysd = sysc.to_discrete(_h[j])
#     pfd  = np.exp(CPS[i]['pfc'] * _h[j]) # discrete-time poles
#     res  = place_poles(sysd.A, sysd.B, pfd) # pole placement
#     Kfd  = -res.gain_matrix
#     
#     # print new discrete-time system
#     print('CPS: {}, h: {}, Kfd: {}'.format(i,_h[j],Kfd))
