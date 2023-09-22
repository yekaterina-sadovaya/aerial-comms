import numpy as np
import math

def comp_factorial(k):
    fact_k = 1
    for i in range(1, k+1):
        fact_k = fact_k * i
    return fact_k

def comp_md1(x, D, l):

    F = 0
    for k in range(0, int(np.floor(x/D))):
        fact_k = comp_factorial(k)
        F = F + (-l*((x - k*D)**k)*np.exp(l*(x - k*D)))/fact_k
    F = (1 - l*D)*F
    return F


def Qm(m, lambda_val, D):

    Q = np.zeros(m)
    for i in range(0, m):
        qi = q0(i, lambda_val, D)
        Q[i] = qi
    return Q


def q0(c, lambda_val, D):
    e_lambdaD = math.exp(-lambda_val * D)
    q = 0
    for j in range(c + 1):
        numerator = (lambda_val * D)**(c + j)
        denominator = math.factorial(c - j)
        qj = numerator / (denominator * e_lambdaD)
        q += qj
    return q


def comp_mdc(x, D, l, c):

    k = c / (l * D)
    Q = Qm(c, l, D)
    F = 0
    a = math.exp(-l * (k * D - x))
    for j in range(0, int(k*c - 1)):
        numerator = Q[int(k*c-j-1)]*(-l*(x -k*D)**j)
        denominator = math.factorial(j)
        F += numerator / denominator
    F = a*F
    return F


t_max = 0.05e-3     # maximum allowed time
D_UE = 0.01e-3      # task processing time for UE
lambda_UE = 0.5

print(comp_md1(t_max - D_UE, D_UE, lambda_UE))
print(comp_mdc(t_max - D_UE, D_UE, lambda_UE, 2))
