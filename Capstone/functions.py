"""The main functions that we are going to use in our implementation are in this file."""

import numpy as np
import math
import scipy

"""Logistic function:"""

def logistic(x):
    """ This is the logistic function for a given numeric value x
    Parameters
    ----------
    :param x: input number to compute the logistic function for
    :type x: float
    
    Returns
    -------
    :return: the logistic value for x
    :rtype: float
    """

    try:
        logist = math.exp(x) / (1 + math.exp(x))
        #logist = x / np.sqrt(1 + x * x)
    except OverflowError:
        if x > 0:
            logist = 1
        else:
            logist = 0
   # logist = scipy.special.expit(x)
    return logist


"""Log-Likelihood: Function that calculates log-likelihood of given Matrix."""


def log_likelihood(X, Y, omega):
    """
    Parameters
    ----------
    :param X: Input matrix to compute the loglikelihood 
    :type X: np.matrix
    :param Y: matrix with 1bit values
    :param omega: set of observed entries
    
      Returns
    -------
    :return: the lloglikelihood 
    :rtype: float
    """
    log_lik = 0
    for i, j in omega:
        elem = 0
        x_ij = X.item((i, j))
        y_ij = Y.item((i, j))
        logistic_x_ij = logistic(x_ij)

        if y_ij == 1:
            if logistic_x_ij == 0:
                elem = 0
            else:
                elem = math.log(logistic_x_ij, math.e)
        elif y_ij == -1:
            temp = 1 - logistic_x_ij
            if temp == 0:
                elem = 0
            else:
                try:
                    elem = math.log(temp, math.e)
                except ValueError:
                    print((x_ij, temp))
        else:
            print( "Y should be either 1 or -1")
        log_lik += elem

    return log_lik

"""Gradient: Computes the gradient of the log likelihood function at a given point(Matrix)"""
def gradient_log_likelihood(X, Y, omega):
    """
     Parameters
    ----------
    :param X: Input matrix
    :type X: np.matrix
    :param Y: matrix with 1bit values
    :param omega: set of observed entries
    """
    gradient_log_lik = 0
    for i, j in omega:
        grad_elem = 0
        x_ij = X.item((i, j))
        y_ij = Y.item((i, j))

        try:
            x_ij_exp = math.exp(x_ij)
        except OverflowError:
            x_ij_exp = float('inf')

        if y_ij == 1:
            try:
                grad_elem = 1/ (x_ij_exp +1)

            except OverflowError:
                grad_elem = 0

        elif y_ij == -1:
            try:
                grad_elem = - x_ij_exp / (x_ij_exp + 1)
            except OverflowError:
                grad_elem = 0
        else:
             print("Y should be either 1 or -1!")
    gradient_log_lik += grad_elem

    return gradient_log_lik

"""Lambda function, computes lambda for Projection function"""

def svd_and_lamdba_x(X, r, alpha):
    """
     Parameters
    ----------
    :param X: input matrix
    :param r: rank of the matrix
    :param alpha: constant value to be chosen
    :return: matrices U,D,V, lam
    """
    U, D, V = np.linalg.svd(X, full_matrices=False)
    d1, d2 = X.shape

    lam = 0
    for k in range(1, min(d1,d2)):
        if nuclear_norm_condition(alpha, r, d1, d2, D, k):
            lam = (np.sum(D[0:k]) - alpha * np.sqrt(r * d1 * d2)) / k
            break
        #print(lam)
        #if lam > 0:
    return U, D, V, lam
            # if D[k] < -1:
            #     return U, D, V, 1


"""Check the nuclear norm condition"""
def nuclear_norm_condition(alpha, r, d1, d2, D, k):
    """
     Parameters
    ----------
    :param r: rank of the matrix
    :param alpha: constant value to be chosen
    :param d1: dimension of the matrix
    :param d2: dimension of the matrix
    :param D: Diagonal matrix
    :param k: 
    :return: boolean answer, true or false
    """
    sum_dk1 = np.sum(D[0:k + 1]) - k * D.item(k - 1)
    sum_dk2 = np.sum(D[0:k + 1]) - k * D.item(k)
    radi = alpha * np.sqrt(r * d1 * d2)
    if sum_dk1 <= radi <= sum_dk2:
        return True
    else:
        return False

"""Function to implement project on a set"""
def projection_on_set(X, r, alpha):
    """
     Parameters
    ----------
    :param X: input matrix
    :param r: rank of the matrix
    :param alpha: constant value to be chosen
    :return: 
    """

    U, D, V, lam = svd_and_lamdba_x(X, r, alpha)
    lam_vector = np.ones(D.shape) * lam
    diff_vector = D - lam_vector
    diff_vector = diff_vector.clip(min=0)
    diff_matrix = np.diag(diff_vector)
    udv = np.dot(U, np.dot(diff_matrix, V))
    return udv

"""Function to compute the objective for the bisection method"""
def bisection_objective(rho, X, Y, omega, s):
    """
     Parameters
    ----------
      :param X: input matrix
      :param rho: result of bisection
      :param s: step size
      :param Y: 1 bit matrix
      :param: omega: set of observed entries
      :return: objective function
      """

    return -1 * log_likelihood(np.dot(rho, X) + np.dot(1 - rho, X + s), Y, omega)


if __name__ == "__main__":
    X = np.matrix('1 2; 3 4; 6 7')
    Y = np.matrix('1 -1; 1 1; -1 1')
    omega = [(0, 1)]
    print(log_likelihood(X, Y, omega))
