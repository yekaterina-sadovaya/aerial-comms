
import numpy as np
import math
import networkx as nx
from matplotlib import pyplot as plt

lam = 0.5
D = 0.1
D_1 = 0.1
c = 2
rho = (lam * D)/c
print("rho = ", rho)
rho_1 = lam * D_1
print("rho_1 = ", rho_1)
M = round(0.5 * (1 + rho) * c + 10 * rho * math.sqrt(c))
k_max = int(M/c) - 1

print("M = ", M)
print("k_max = ", k_max)

p = np.zeros(M+1)
q = np.zeros(2*M+1)
p0 = 0.7845
for i in range(M+1):
    p[i] = p0 * 0.5**i
P = p/sum(p)
tau = 0.01

q = np.zeros(2*M)
for l in range(2*M):
    if l == 0:
        q[l] = sum(P[:c+1])
    else:
        if c+l >= M:
            q[l] = P[M] * tau**(c + l - M)
        else:
            q[l] = P[c+l]

def W_mdc(x, c, D, lam, q):
    k = math.floor(x/D)
    temp = 0
    for j in range(k * c - 1):
        Q = sum(q[:k * c - 1 - j])
        temp += Q * np.exp(lam * (x - k * D)) * math.pow( - lam * (x - k * D), j)/ math.factorial(j)
    res = np.exp(lam * (x - k * D)) * temp
    return res

def W_md1(x, c, D, lam):
    temp = 0
    for k in range(math.floor(x/D)):
        temp = temp + (-lam * (x - k * D))**k * math.exp(lam * (x - k * D))/math.factorial(k)
    res = (1 - lam * D) * temp
    return res

x1_value, xc_value, yc, y1 = [], [], [], []

x_all = np.linspace(0, 1, 100)
# x_all = range(0, 20)
for x in x_all:
    yc.append(W_mdc(x, c, D, lam, q))
    xc_value.append(x)

for x in x_all:
    y1.append(W_md1(x, c, D_1, lam))
    x1_value.append(x)

yc_value = yc
y1_value = y1

plt.plot(x1_value, y1_value)
plt.title('M/D/1')
plt.show()

plt.plot(xc_value, yc_value)
plt.title('M/D/c')
plt.show()
