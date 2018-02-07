from utilities import *
import numpy as np
import matplotlib.pyplot as plt

f_analytical = None


class Questions:
    f_analytical = None

    def question_1(self):
        f = lambda x, d: (2 / 3 - x) / (d ** 2 + (2 / 3 - x) ** 2) ** (1 / 2) - (1 / 3 - x) / (d ** 2 + (
        1 / 3 - x) ** 2) ** (1 / 2)
        x = np.linspace(0, 1, 1000)

        plt.figure(11)
        plt.title("Question 1")
        plt.plot(x, f(x, 0.025), 'k-', label=r"$d = 0.025$")
        plt.plot(x, f(x, 0.25), 'k--', label=r"$d = 0.25$")
        plt.plot(x, f(x, 2.5), 'k-.', label=r"$d = 2.5$")
        plt.ylabel(r"$\log F(x)$")
        plt.xlabel(r"$x$")
        plt.xlim([0, 1])
        plt.yscale('log')
        plt.legend()

    def question_3(self):
        a = 0
        b = 1
        d = 0.025
        Nc = Ns = 40
        xs = xc = chebyshev(0, 1, Nc)
        Nq_lim = 50
        omega = 3 * np.pi
        gamma = -2
        rho = analytic_density(xs, 3 * np.pi, -2)
        if self.f_analytical is None:
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)
        f_analytical_vector = fredholm_rhs(xc, self.f_analytical)
        error, points = newton_cotes(Nq_lim, xc, xs, rho, a, b, f_analytical_vector)

        plt.figure(31)
        plt.title(r"Error")
        plt.xlabel(r"$N_q$")
        plt.ylabel(r"$\max |F(x_i) - (\mathbf{A\hat{\rho}})_i|$")
        nq_vals = np.arange(1, Nq_lim + 1, 1)
        plt.plot(nq_vals, error, 'k-', label="Newton-Cotes")
        plt.legend()

    def question_4(self):
        a = 0
        b = 1
        d = 0.025
        Nc = Ns = 40
        xs = xc = chebyshev(0, 1, Nc)
        Nq_lim = 50
        omega = 3 * np.pi
        gamma = -2
        rho = analytic_density(xs, omega, gamma)
        if self.f_analytical is None:
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)
        f_analytical_vector = fredholm_rhs(xc, self.f_analytical, d)
        error, points = legendre_gauss(Nq_lim, xc, xs, rho, 0, 1, f_analytical_vector)

        plt.figure(41)
        plt.title(r"Error")
        plt.xlabel(r"$N_q$")
        plt.ylabel(r"$\max |F(x_i) - (\mathbf{A\hat{\rho}})_i|$")
        nq_vals = np.arange(1, Nq_lim + 1, 1)
        plt.plot(nq_vals, error, 'k--', label="Legendre-Gauss")
        plt.legend()

    def question_5(self):
        ds = [0.025, 0.25, 2.5]
        a = 0
        b = 1
        omega = 3 * np.pi
        gamma = -2
        if self.f_analytical is None:
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)

        count = 50
        for d in ds:
            print("Calculating d = {0}".format(d))
            error = []
            Ncs = Nss = np.arange(5, 31, 1)
            for Ns in Nss:
                print("Calculating (d = {0}) N_c = {1} of {2}".format(d, Ns, 30))
                Nc = Ns
                Nq = Ns ** 2
                xs = xc = chebyshev(a, b, Ns)
                f_analytical_vector = fredholm_rhs(xc, self.f_analytical, d)
                rho_calculated = calculate_rho_from_inverse(xc, xs, Nq, a, b, d, f_analytical_vector)
                rho_analytical = analytic_density(xs, omega, gamma)
                diff = rho_analytical - rho_calculated
                largest_error = np.linalg.norm(diff, np.Inf)
                error.append(largest_error)
            count += 1
            plt.figure(count)
            plt.title("Error for d = {0}".format(d))
            plt.plot(Ncs, error, 'k-')
            plt.ylabel(r"$\log [\max |\mathbf{\hat{\rho}} - \rho(x_j^s)|]$")
            plt.xlabel(r"$N_c$")
            plt.yscale('log')
        plt.show()

questions = Questions()
questions.question_5()
























