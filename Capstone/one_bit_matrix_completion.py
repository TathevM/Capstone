
from Capstone import functions as fun
from Capstone import constants as cnts

import numpy as np
from scipy.optimize import minimize_scalar

class OneBitMatrixCompletion:

    r = 2
    alpha = 0.8
    gamma = 0.5
    NUM_STEPS = 1000

    def __init__(self):
        pass

    def complete(self, Y):
        """

        :param Y:
        :type Y: np.matrix
        """
        omega, Mk = self.compute_omega_m0(Y)
        for k in range(1, 1000):
            _, _, _, lam = fun.svd_and_lamdba_x(Mk, self.r, self.alpha)
            proj_input = Mk - self.gamma * fun.gradient_log_likelihood(Mk, Y, omega)
            step = fun.projection_on_set(proj_input, self.r, self.alpha) - Mk
            rho = minimize_scalar(fun.bisection_objective, bounds=(0, 1), args=(Mk, Y, omega, step))
            Mk = Mk + np.dot(rho.x, step)


    def compute_omega_m0(self, Y):
        omega = []
        M = Y
        d1, d2 = Y.shape
        for i, j in zip(range(d1), range(d2)):
            if Y.item((i, j)) == cnts.NO_VALUE:
                omega.append((i, j))
                M[i, j] = 0
        return omega, M

