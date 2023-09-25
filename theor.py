import numpy as np
import math
import matplotlib.pyplot as plt


def comp_factorial(k):
    fact_k = 1
    for i in range(1, k+1):
        fact_k = fact_k * i
    return fact_k


def comp_md1(x, D, l):

    F = 0
    for k in range(0, int(np.floor(x/D))):
        fact_k = comp_factorial(k)
        F = F + ((-l*(x - k*D))**k)*np.exp(l*(x - k*D))/fact_k
    F = (1 - l*D)*F
    return F


def Qm(m, c, lambda_val, D):

    Q = np.zeros(m)
    e_lambdaD = math.exp(-lambda_val * D)
    q_first = q0(c, lambda_val, D)
    Q[0] = q_first
    for i in range(1, m):
        q_i = 0
        for j in range(0, int(c+i)):
            q_i += j*(((lambda_val * D)**(i+c*j))*e_lambdaD)/math.factorial(i+c-j)
        Q[i] = q_i
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


def comp_mdc(x, D, l, c):

    k = int(np.floor(x/D))
    m = int(k*c)
    if k == 0:
        F = 0
    else:
        Q = Qm(m, c, l, D)
        F = 0
        a = math.exp(-l * (k * D - x))
        for j in range(0, m):
            numerator = Q[int(k*c-j-1)]*((-l*(x - k*D))**j)
            denominator = math.factorial(j)
            F += numerator / denominator
        F = a*F
    return F


t_max = np.linspace(0.001, 0.01, 100)     # maximum allowed time
D_UE = 0.001      # task processing time for UE
lambda_UE = 0.5

waiting_time_md1 = []
waiting_time_mdc = []
for t_max_i in t_max:
    t_md1 = comp_md1(t_max_i - D_UE, D_UE, lambda_UE)
    waiting_time_md1.append(t_md1)
    t_mdc = comp_mdc(t_max_i - D_UE, D_UE, lambda_UE, 2)
    waiting_time_mdc.append(t_mdc)

print(waiting_time_md1)
print(waiting_time_mdc)

plt.figure()
plt.plot(waiting_time_mdc)
plt.show()
