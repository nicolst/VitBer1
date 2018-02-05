import pickle

import numpy as np
from test_example import analytical_solution
import matplotlib.pyplot as plt



def fredholm_rhs(xc, F):
    Nc = xc.shape[0]
    b = np.zeros(Nc)
    b = F(xc, d)
    return b


def fredholm_lhs(xc, xs, xq, w, K):
    Nc = xc.shape[0]
    Ns = xs.shape[0]
    Nq = xq.shape[0]
    A = np.zeros((Nc, Ns))

    for i in range(Nc):
        #print("{0} out of {1}.".format(i, Nc))
        for j in range(Ns):
            term = 0
            for k in range(Nq):
                term += lagrange_poly(xq[k], j, xs)*K(xc[i], xq[k])*w
            A[i][j] = term

    return A


def chebyshev(a, b, N):
    # return Chebyshev's interpolation points on the interval [a,b]
    I = np.arange(1, N+1, 1)
    X = (b + a)/2 + (b - a)/2*np.cos((2*I - 1)*np.pi/(2*N))
    return X


def lagrange_poly(xk, j, Xs):
    li = np.prod(xk - Xs[np.arange(Xs.shape[0])!=j])/np.prod(Xs[j] - Xs[np.arange(Xs.shape[0])!=j])
    return li


def kernel_fred(xc, xq):
    return d/((d**2+(xc-xq)**2)**(3/2))

def analytic_density(x, omega, gamma):
    return np.sin(omega*x)*np.exp(gamma*x)

d = 0.025

Nnc = 40
Nns = 40
Nnq = 100

a= 0
b = 1

print("Chebyshev")
xc = chebyshev(a, b, Nnc)
xs = chebyshev(a, b, Nns)
xq = np.linspace(a, b, Nnq)

w = (b-a)/(Nnq)

omega = 3*np.pi
gamma = -2

Nmax = 75

print("Executing analytical_solution")
try:
    Fa = pickle.load(open("F.pkl", "rb"))

except:
    Fa = analytical_solution(a, b, omega, gamma, Nmax)
    pickle.dump(Fa, open("F.pkl", "wb"))
#Fa = analytical_solution(a, b, omega, gamma, 75)
Fa_eval = fredholm_rhs(xc, Fa)

#print("Executing fredholm")
#A = fredholm_lhs(xc, xs, xq, w, kernel_fred)
p_analytical = analytic_density(xs, omega, gamma)

#print("Matrix product")
#Fd_vals = A.dot(p_analytical)

#diff = np.abs(Fd_vals - Fa_eval)
#print(np.max(diff))

errors = []
top = 40
for i in range(1, top):
    print("Error calculation: {0} of {1}.".format(i, top-1))
    xq = np.linspace(a, b, i)
    w = (b - a) / (i)
    A = fredholm_lhs(xc, xs, xq, w, kernel_fred)
    Fd_vals = A.dot(p_analytical)
    diff = np.abs(Fd_vals - Fa_eval)
    errors.append(np.max(diff))

#plt.plot(xc, Fd_vals, '--')
#plt.plot(xc, Fa_eval, '-')
plt.plot(np.arange(1, top, 1), errors)
plt.show()






























