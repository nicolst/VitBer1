import pickle

import numpy as np
from test_example import analytical_solution
import matplotlib.pyplot as plt
import matplotlib


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
        # print("{0} out of {1}.".format(i, Nc))
        for j in range(Ns):
            term = 0
            for k in range(Nq):
                term += lagrange_poly(xq[k], j, xs) * K(xc[i], xq[k]) * w[k]
            A[i][j] = term

    return A


def chebyshev(a, b, N):
    # return Chebyshev's interpolation points on the interval [a,b]
    I = np.arange(1, N + 1, 1)
    X = (b + a) / 2 + (b - a) / 2 * np.cos((2 * I - 1) * np.pi / (2 * N))
    return X


def lagrange_poly(xk, j, Xs):
    li = np.prod(xk - Xs[np.arange(Xs.shape[0]) != j]) / np.prod(Xs[j] - Xs[np.arange(Xs.shape[0]) != j])
    return li


def kernel_fred(xc, xq):
    return d / ((d ** 2 + (xc - xq) ** 2) ** (3 / 2))


def analytic_density(x, omega, gamma):
    return np.sin(omega * x) * np.exp(gamma * x)


def newton_cotes(top_nq, xc, xs, rho):
    errors = []
    points = []
    for i in range(1, top_nq + 1):
        print("Newton-Cotes: {0} of {1}.".format(i, top_nq))
        w = np.repeat((b - a) / (i), i)
        xq = np.linspace(a, b, i, endpoint=False) + w / 2
        # print(xq)
        A = fredholm_lhs(xc, xs, xq, w, kernel_fred)
        Fd_vals = A.dot(rho)
        diff = np.abs(Fd_vals - Fa_eval)
        errors.append(np.max(diff))
        points.append(Fd_vals)
    return errors, points

def legendre_gauss(top_nq, xc, xs, rho, a, b):
    errors = []
    points = []
    for i in range(1, top_nq + 1):
        print("Legendre-Gauss: {0} of {1}.".format(i, top_nq))
        xq,w = np.polynomial.legendre.leggauss(i)
        w = w * (b-a)/2
        xq = (b+a)/2 + xq*(b-a)/2
        A = fredholm_lhs(xc, xs, xq, w, kernel_fred)
        Fd_vals = A.dot(rho)
        diff = np.abs(Fd_vals - Fa_eval)
        errors.append(np.max(diff))
        points.append(Fd_vals)
    return errors,points


d = 0.025

Nnc = 40
Nns = 40
Nnq = 100

a = 0
b = 1

print("Chebyshev")
xc = chebyshev(a, b, Nnc)
xs = chebyshev(a, b, Nns)

w = (b - a) / (Nnq)

omega = 3 * np.pi
gamma = -2

Nmax = 75

print("Executing analytical_solution")
try:
    Fa = pickle.load(open("F.pkl", "rb"))

except:
    Fa = analytical_solution(a, b, omega, gamma, Nmax)
    pickle.dump(Fa, open("F.pkl", "wb"))
# Fa = analytical_solution(a, b, omega, gamma, 75)
Fa_eval = fredholm_rhs(xc, Fa)

# print("Executing fredholm")
# A = fredholm_lhs(xc, xs, xq, w, kernel_fred)
p_analytical = analytic_density(xs, omega, gamma)

# print("Matrix product")
# Fd_vals = A.dot(p_analytical)

# diff = np.abs(Fd_vals - Fa_eval)
# print(np.max(diff))

# errors = []
# top = 40
# for i in range(1, top):
#     print("Error calculation: {0} of {1}.".format(i, top-1))
#     xq = np.linspace(a, b, i)
#     w = (b - a) / (i)
#     A = fredholm_lhs(xc, xs, xq, w, kernel_fred)
#     Fd_vals = A.dot(p_analytical)
#     diff = np.abs(Fd_vals - Fa_eval)
#     errors.append(np.max(diff))

# plt.plot(xc, Fd_vals, '--')
# plt.plot(xc, Fa_eval, '-')

top = 40
errors_NC, points_NC = newton_cotes(top, xc, xs, p_analytical)
errors_LG, points_LG = legendre_gauss(top, xc, xs, p_analytical, a, b)
#print(errors)

#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

plt.figure(1)
plt.title(r"Error")
plt.xlabel(r"$N_q$")
plt.ylabel(r"$\max |F(x_i) - (\mathbf{A\hat{\rho}})_i|$")
nq_vals = np.arange(1, top + 1, 1)
plt.plot(nq_vals, errors_NC, label="Newton-Cote")
plt.plot(nq_vals, errors_LG, label="Legendre-Gauss", linestyle='--')
plt.legend()

t = 2
mod_amount = 8
count = 0
for pointset in points_NC:
    if count == mod_amount:
        t += 1
        count = 0
        plt.plot(xc, Fa_eval, '--', label="Analytical")
        plt.title("NC")
        plt.legend()
        plt.tight_layout()

    plt.figure(t)
    plt.plot(xc, pointset, label="Nq = {0}".format((t-2)*mod_amount + count + 1))
    count += 1
plt.plot(xc, Fa_eval, '--', label="Analytical")
plt.title("NC")
plt.legend()
plt.tight_layout()

if count > 0:
    t += 1
initial = t
count = 0
for pointset in points_LG:
    if count == mod_amount:
        t += 1
        count = 0
        plt.plot(xc, Fa_eval, '--', label="Analytical")
        plt.title("LG")
        plt.legend()
        plt.tight_layout()

    plt.figure(t)
    plt.plot(xc, pointset, label="Nq = {0}".format((t-initial)*mod_amount + count + 1))
    count += 1
plt.plot(xc, Fa_eval, '--', label="Analytical")
plt.title("LG")
plt.legend()
plt.tight_layout()

plt.show()
