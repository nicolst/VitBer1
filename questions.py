import matplotlib
#matplotlib.use('Qt4Agg')

from utilities import *
import numpy as np


import matplotlib.pyplot as plt
import random

f_analytical = None


class Questions:
    f_analytical = None

    def question_1(self):
        print("Executing question 1...")

        print("Defining force function...")
        f = lambda x, d: (2/3 - x)/(d**2 + (2/3 - x)**2)**(1/2) - (1/3 - x)/(d**2 + (1/3 - x)**2)**(1/2)
        x = np.linspace(0, 1, 1000)

        print("Plotting...")
        plt.figure(11)
        plt.title("Question 1")
        plt.plot(x, f(x, 0.025), 'k-', label=r"$d = 0.025$")
        plt.plot(x, f(x, 0.25), 'k--', label=r"$d = 0.25$")
        plt.plot(x, f(x, 2.5), 'k-.', label=r"$d = 2.5$")
        plt.ylabel(r"$\log\ F(x)$", size=20)
        plt.xlabel(r"$x$", size=20)
        plt.xlim([0, 1])
        plt.yscale('log')
        plt.legend()

    def question_3(self):
        print("Executing question 3...")

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
            print("Analytical force function not defined, creating...")
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)

        f_analytical_vector = fredholm_rhs(xc, self.f_analytical, d)

        error, points = newton_cotes(Nq_lim, xc, xs, rho, d, a, b, f_analytical_vector)

        plt.figure(31)
        plt.title(r"Q3: Max error of N-C quadrature")
        #plt.yscale('log')
        plt.xlabel(r"$N_q$", size=20)
        plt.ylabel(r"$\max\ |F(x_i) - (\mathbf{A\hat{\rho}})_i|$", size=20)
        nq_vals = np.arange(1, Nq_lim + 1, 1)
        plt.plot(nq_vals, error, 'k-', label="Newton-Cotes")
        plt.legend()

    def question_4(self):
        print("Executing question 4...")

        a = 0
        b = 1

        d = 0.025

        Nc = Ns = 40
        xs = xc = chebyshev(0, 1, Nc)

        Nq_lim = 150

        omega = 3 * np.pi
        gamma = -2

        rho = analytic_density(xs, omega, gamma)

        if self.f_analytical is None:
            print("Analytical force function not defined, creating...")
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)

        f_analytical_vector = fredholm_rhs(xc, self.f_analytical, d)

        error_NC, points_NC = newton_cotes(Nq_lim, xc, xs, rho, d, a, b, f_analytical_vector)
        error, points = legendre_gauss(Nq_lim, xc, xs, rho, d, a, b, f_analytical_vector)

        print("Plotting...")
        plt.figure(41)
        plt.title(r"Max error of L-G and N-C quadratures")
        plt.yscale('log')
        plt.xlabel(r"$N_q$", size=20)
        plt.ylabel(r"$\log\ \max\ |F(x_i) - (\mathbf{A\hat{\rho}})_i|$", size=20)
        nq_vals = np.arange(1, Nq_lim + 1, 1)
        plt.plot(nq_vals, error, 'k-', label="Legendre-Gauss")
        plt.plot(nq_vals, error_NC, 'k--', label="Newton-Cotes")
        plt.legend()

        plt.figure(42)
        plt.title(r"Max error of L-G and N-C quadratures")
        #plt.yscale('log')
        plt.xlabel(r"$N_q$", size=20)
        plt.ylabel(r"$\max\ |F(x_i) - (\mathbf{A\hat{\rho}})_i|$", size=20)
        nq_vals = np.arange(1, Nq_lim + 1, 1)
        plt.plot(nq_vals, error, 'k-', label="Legendre-Gauss")
        plt.plot(nq_vals, error_NC, 'k--', label="Newton-Cotes")
        plt.legend()

    def question_5(self):
        print("Executing question 5...")

        ds = [0.025, 0.25, 2.5]

        a = 0
        b = 1

        omega = 3 * np.pi
        gamma = -2

        if self.f_analytical is None:
            print("Analytical force function not defined, creating...")
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
                rho_calculated = calculate_rho(xc, xs, Nq, a, b, d, f_analytical_vector)
                rho_analytical = analytic_density(xs, omega, gamma)
                diff = rho_analytical - rho_calculated
                largest_error = np.linalg.norm(diff, np.Inf)
                error.append(largest_error)
            count += 1
            plt.figure(count)
            plt.title("Max error for d = {0}".format(d))
            plt.plot(Ncs, error, 'k-')
            plt.ylabel(r"$\log\ [\max |\mathbf{\hat{\rho}} - \rho(x_j^s)|]$", size=20)
            plt.xlabel(r"$N_c$", size=20)
            plt.yscale('log')


    def question_6(self):
        print("Performing question 6...")
        ds = [0.025, 0.25, 2.5]

        a = 0
        b = 1

        print("Getting Chebyshev interpolation nodes...")
        Ns = Nc = 30
        Nq = Ns**2
        xs = xc = chebyshev(a, b, Ns)

        omega = 3 * np.pi
        gamma = -2

        delta = 10 ** (-3)
        random_noise = np.random.uniform(-delta, delta, Nc)

        if self.f_analytical is None:
            print("Analytical force function not defined, creating...")
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)

        B_analytical_list = []
        B_perturbed_list = []
        fig_count = 60
        for d in ds:
            print("Calculating for d = {0}:".format(d))



            print("Calculating perturbed and non-perturbed force vectors (B)...")
            B_analytical = fredholm_rhs(xc, self.f_analytical, d)
            B_perturbed = B_analytical * (1.0 + random_noise)

            print("Calculating perturbed rho...")
            rho_calc_perturbed = calculate_rho(xc, xs, Nq, a, b, d, B_perturbed)
            print("Calculating non-perturbed rho...")
            rho_calc_nonperturbed = calculate_rho(xc, xs, Nq, a, b, d, B_analytical)
            print("Calculating analytical rho...")
            rho_analytical = analytic_density(xs, omega, gamma)

            B_analytical_list.append(B_analytical)
            B_perturbed_list.append(B_perturbed)

            print("Adding to plot...")
            fig_count += 1
            fig = plt.figure(fig_count)
            ax1 = fig.add_subplot(211)
            ax1.set_ylabel(r"$\rho (x)$", size=20)
            ax1.plot(xs, rho_analytical, 'k-', label="Analytical density")
            ax1.plot(xs, rho_calc_perturbed, 'k-.', label="Solution (perturbed)")
            ax1.set_title("Analytical density and solutions for d = {0}".format(d))
            ax1.legend()

            ax2 = fig.add_subplot(212, sharex=ax1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.set_ylabel(r"$\rho (x)$", size=20)
            ax2.set_xlabel(r"$x$", size=20)
            ax2.plot(xs, rho_analytical, 'k-', label="Analytical density")
            ax2.plot(xs, rho_calc_nonperturbed, 'k--', label="Solution (non-perturbed)")
            ax2.legend()


        fig_count += 1
        for i in range(3):
            fig = plt.figure(fig_count)
            fig_count += 1
            ax1 = fig.add_subplot(211)
            ax1.set_ylabel(r"$F(x)$", size=20)
            ax1.set_title("Perturbed and non-perturbed forces for d = {0}".format(ds[i]))
            ax1.plot(xs, B_analytical_list[i], 'k-', label=r"Non-perturbed ($\mathbf{b}$)")
            ax1.plot(xs, B_perturbed_list[i], 'k--', label=r"Perturbed ($\mathbf{\~b}$)")
            ax1.legend()

            diff = B_perturbed_list[i] - B_analytical_list[i]
            ax2 = fig.add_subplot(212, sharex=ax1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.set_xlabel(r"$x$", size=20)
            ax2.set_ylabel(r"$F(x)$", size=20)
            ax2.set_title("Difference between perturbed and non-perturbed force")
            ax2.plot(xs, diff, 'k-')



    def question_7(self):
        print("Performing question 6...")
        ds = [0.25]

        a = 0
        b = 1

        print("Getting Chebyshev interpolation nodes...")
        Ns = Nc = 30
        Nq = Ns ** 2
        xs = xc = chebyshev(a, b, Ns)

        omega = 3 * np.pi
        gamma = -2

        if self.f_analytical is None:
            print("Analytical force function not defined, creating...")
            self.f_analytical = analytical_force(a, b, omega, gamma, 75)

        B_analytical_list = []
        B_perturbed_list = []
        fig_count = 70

        delta = 10 ** (-3)
        random_noise = np.random.uniform(-delta, delta, Nc)

        print("Calculating analytical rho...")
        rho_analytical = analytic_density(xs, omega, gamma)

        xq, w = np.polynomial.legendre.leggauss(Nq)
        w = w * (b - a) / 2
        xq = (b + a) / 2 + xq * (b - a) / 2

        for d in ds:

            print("Calculating perturbed and non-perturbed force vectors (B)...")
            B_analytical = fredholm_rhs(xc, self.f_analytical, d)
            B_perturbed = B_analytical * (1.0 + random_noise)

            A = fredholm_lhs(xc, xs, xq, d, w, kernel_fred)
            A_T = np.matrix.transpose(A)
            lhs_A = A_T.dot(A)
            rhs = A_T.dot(B_perturbed)

            lmbda_space = np.geomspace(10**(-14), 10, 140)
            errors = []
            points = []
            for lmbda in lmbda_space:
                print("Calculating for d = {0}, lambda={1}:".format(d, lmbda))


                print("Calculating perturbed rho with tikhonov regularization...")
                #rho_calc_perturbed_thikonoff = calculate_rho_thikonoff(xc, xs, Nq, a, b, d, B_perturbed, lmbda)
                rho_calc_perturbed_thikonoff = calculate_rho_thikonoff_givenA(xc, lmbda, rhs, lhs_A)


                error = rho_calc_perturbed_thikonoff - rho_analytical
                biggest_error = np.linalg.norm(error, np.Inf)
                errors.append(biggest_error)
                points.append([rho_analytical, rho_calc_perturbed_thikonoff])

            fig_count += 1
            plt.figure(fig_count)
            plt.loglog(lmbda_space, errors)
            plt.xlabel(r"$\lambda$")
            plt.ylabel(r"$\max_j\  |\rho (x_j^s) - \mathbf{\hat{\rho}}|$")
            plt.title(r"Largest errors for d = {0}".format(d))

            fig_count += 1
            plt.figure(fig_count)
            ind = np.argmin(errors)
            plt.title(r"Solution with smallest error (lambda={0})".format(lmbda_space[ind]))
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\rho$")
            plt.plot(xs, points[ind][0], 'k-', label="Analytical")
            plt.plot(xs, points[ind][1], 'k--', label="Tikhonov")
            plt.legend()


    def question_8(self):
        #f_test = open('q8_test.npz', 'rb')
        #np_test = np.load(f_test)

        f_1 = open('q8_1.npz', 'rb')
        np_1 = np.load(f_1)

        f_2 = open('q8_2.npz', 'rb')
        np_2 = np.load(f_2)

        f_3 = open('q8_3.npz', 'rb')
        np_3 = np.load(f_3)

        nps = [np_1, np_2, np_3]

        i = 1
        counter = 1
        for nep in nps:
            print("Nep: {0}".format(i))
            i += 1

            a = nep['a']
            b = nep['b']
            d = nep['d']
            xc = nep['xc']
            F = nep['F']
            xs = chebyshev(a, b, len(xc))
            Nq = len(xs)**2

            Ns = len(nep['xc'])
            print("Getting leg-gaus nodes,weights")
            xq, w = np.polynomial.legendre.leggauss(Nq)
            w = w * (b - a) / 2
            xq = (b + a) / 2 + xq * (b - a) / 2
            print("Calc A")
            A = fredholm_lhs(xc, xs, xq, d, w, kernel_fred)
            print("A_T")
            A_T = np.matrix.transpose(A)
            print("A_T dot A")
            lhs_A = A_T.dot(A)
            rhs = A_T.dot(F)

            lmbda_space = np.geomspace(10 ** (-14), 10, 10)
            for lbd in lmbda_space:
                print("Lambda: {0}".format(lbd))
                #rho = calculate_rho_thikonoff(xc, xs, Nq, a, b, d, F, lbd)
                rho = calculate_rho_thikonoff_givenA(xc, lbd, rhs, lhs_A)
                plt.figure(counter)
                plt.title("Calculated rho for lbd={0}, nep {1}".format(lbd, i-1))
                plt.plot(xs, rho)
                counter += 1




print(np.__version__)
questions = Questions()
questions.question_1()
print("DONE")
plt.show()
print("test")























