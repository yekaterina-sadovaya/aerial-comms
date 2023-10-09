import numpy as np
import math
import matplotlib.pyplot as plt
from gekko import GEKKO


def comp_factorial(k):
    fact_k = 1
    for i in range(1, k+1):
        fact_k = fact_k * i
    return fact_k


def comp_md1(m, x, D, l):

    F = 0
    for k in range(0, m):
        fact_k = comp_factorial(k)
        F = F + ((-l*(x - k*D))**k)*np.exp(l*(x - k*D))/fact_k
    F = (1 - l*D)*F
    return F


def Qm(m, c, lambda_val, D):

    Q = []
    e_lambdaD = math.exp(-lambda_val * D)
    q_first = q0(c, lambda_val, D)
    Q.append(q_first)
    for i in range(0, m):
        q_i = 0
        for j in range(0, int(c+i)):
            q_i += j*(((lambda_val * D)**(i+c*j))*e_lambdaD)/math.factorial(i+c-j)
        Q.append(q_i)
    Q = np.sum(Q)
    return Q


def q0(c, lambda_val, D):
    e_lambdaD = math.exp(-lambda_val * D)
    q = 0
    for j in range(0, c):
        q_sec_sum = 0
        for m in range(0, c-j):
            q_sec_sum += (((lambda_val * D)**m)*e_lambdaD)/math.factorial(m)
        q += j*q_sec_sum
    return q


def comp_mdc(k, x, D, l, c):

    if k == 0:
        F = 0
    else:
        F = 0
        a = math.exp(-l * (k * D - x))
        for j in range(0, k*c - 1):
            m = int(k*c-j-1)
            Q = Qm(m, c, l, D)
            numerator = Q*((-l*(x - k*D))**j)
            denominator = math.factorial(j)
            F += numerator / denominator
        F = a*F
    return F


t = np.linspace(0.01, 0.5, 1000)     # maximum allowed time
D_UE = 0.1      # task processing time for UE
lambda_UE = 0.75

# test
waiting_time_md1 = []
waiting_time_mdc = []
for t_i in t:
    n_ues = int(np.floor(t_i / D_UE))
    t_md1 = comp_md1(n_ues, t_i - D_UE, D_UE, lambda_UE)
    waiting_time_md1.append(t_md1 * np.heaviside(t_i - D_UE, 0))
    t_mdc = comp_mdc(n_ues, t_i - D_UE, D_UE, lambda_UE, 2)
    waiting_time_mdc.append(t_mdc * np.heaviside(t_i - D_UE, 0))

print(waiting_time_md1)
print(waiting_time_mdc)

# compute coefficients for baseline strategy
t_star = 0.06
D_UAV = 0.05
D_HAPS = 0.01
lambda_UAV = 0.5
lambda_HAPS = 0.35

model = GEKKO()
eps = model.Var(lb=0)
nu = model.Var(lb=0)

p_ue = comp_md1(int(np.floor(t_star / D_UE)), t_star - D_UE, D_UE, lambda_UE)*np.heaviside(t_star - D_UE, 0)
p_uav = comp_mdc(int(np.floor(t_star / D_UAV)), t_star - D_UAV, D_UAV, lambda_UAV, 2)*np.heaviside(t_star - D_UAV, 0)
p_haps = comp_mdc(int(np.floor(t_star / D_HAPS)),t_star - D_HAPS, D_HAPS, lambda_UAV, 2)*np.heaviside(t_star - D_HAPS, 0)
obj = model.Intermediate(eps*p_ue + nu*p_uav + (1 - eps - nu)*p_haps)
model.Equation(eps + nu <= 1)

model.Maximize(obj)
model.solve(disp=True)
eps = np.array(eps)
nu = np.array(nu)
print('Eps = '+ str(eps)+', Nu = '+str(nu))
res = eps*p_ue + nu*p_uav + (1 - eps - nu)*p_haps
print('Probability of task being computed with the given time P(eps, nu) = '+str(res))

plt.figure()
plt.plot(waiting_time_md1)
plt.figure()
plt.plot(waiting_time_mdc)
plt.show()
