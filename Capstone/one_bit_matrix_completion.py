
from Capstone import functions as fun
from Capstone import constants as cnts
import numpy as np
from scipy.optimize import minimize_scalar


class OneBitMatrixCompletion:

    r = int
    alpha = float
    gamma = float
    num_steps = int

    def __init__(self, r_, step_, gamma_, alpha_):
        self.r = r_
        self.num_steps = step_
        self.gamma = gamma_
        self.alpha = alpha_

    def complete(self, Y):
        """
        Parameters
        --------------
        :param Y: 1 bit matrix
        :type Y: np.matrix
        """
        omega, mk = self.compute_omega_m0(Y)
        for k in range(1, self.num_steps):
            if k == int(self.num_steps / 2):
                print('50% progress')
            elif k == int(0.75 * self.num_steps):
                print('75% progress')
            _, _, _, lam = fun.svd_and_lamdba_x(mk, self.r, self.alpha)
            proj_input = mk - self.gamma * fun.gradient_log_likelihood(mk, Y, omega)
            step = fun.projection_on_set(proj_input, self.r, self.alpha) - mk
            rho = minimize_scalar(fun.bisection_objective, bounds=(0, 1), method='bounded',
                                  args=(mk, Y, omega, step))
            mk = mk + np.dot(rho.x, step)

        mk = self.normalize_matrix(mk)
        return mk

    @staticmethod
    def compute_omega_m0(Y):
        """
        Parameters
        --------------
        :param Y: 1 bit matrix
        :type Y: np.matrix
        """
        omega = []
        M = Y
        d1, d2 = Y.shape
        for i, j in zip(range(d1), range(d2)):
            if Y.item((i, j)) != cnts.NO_VALUE:
                omega.append((i, j))
        return omega, M

    @staticmethod
    def normalize_matrix(Mhat):
        """

        :param Mhat: 
        :return: 
        """
        d1, d2 = Mhat.shape
        for i in range(d1):
            for j in range(d2):
                val = fun.logistic(Mhat[i, j])
                if val >= 1 / 2:
                    Mhat[i, j] = 1
                else:
                    Mhat[i, j] = -1
        return Mhat
