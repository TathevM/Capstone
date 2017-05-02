"""
All important functions that we are going to use are in this file.
"""
import numpy as np
import math
import scipy

"""
Logistic function:
"""

def logistic(x):
    """ This is the logistic function for a given numeric value x
  
    Parameters
    ----------
    :param x: input number to comute the logistic function for
    :type x: float
    
    Returns
    -------
    :return: the logistic value for x
    :rtype: float
    """
    print(x)

    if x > 0:
        logist = 1 / (1 + math.exp(x))
    else:
        logist = 1 / (1 + math.exp(-1 * x))
   # logist = scipy.special.expit(x)
    return logist


"""
Log-Likelihood: Function that calculates log-likelihood of given Matrix.
"""


def log_likelihood(X, Y, omega):
    """
  
    :param X:
    :type X: np.matrix
    :param Y: 
    :param omega: 
    """
    log_lik = 0
    for i, j in omega:
        elem = 0
        x_ij = X.item((i, j))
        y_ij = Y.item((i, j))
        logistic_x_ij = logistic(x_ij)
        if y_ij == 1:
            elem = math.log(logistic_x_ij, math.e)
        elif y_ij == -1:
            temp = 1 - logistic_x_ij
            #print((i, j, temp, x_ij))
            elem = math.log(temp, math.e)
        else:
            print( "Y should be either 1 or -1")
        log_lik += elem

    return log_lik

"""
Gradient: Computes the gradient of the log likelihood function at a given point(Matrix)
"""
def gradient_log_likelihood(X, Y, omega):
    """

    :param X:
    :type X: np.matrix
    :param Y: 
    :param omega: 
    """
    gradient_log_lik = 0
    for i, j in omega:
        grad_elem = 0
        x_ij = X.item((i, j))
        y_ij = Y.item((i, j))
        x_ij_exp = math.exp(x_ij)
        if y_ij == 1:
            grad_elem = -x_ij_exp / (x_ij_exp + 1)
        elif y_ij == -1:
            grad_elem = 1 / (x_ij_exp + 1)
        else:
            print("What is wrong with you, man! Y should be either 1 or -1, goddamn it")
        gradient_log_lik += grad_elem

    return gradient_log_lik


"""
Lambda function, computes lambda for Projection function
"""

def svd_and_lamdba_x(X, r, alpha):
    """

    :param X: 
    :param r: 
    :param alpha: 
    :return: 
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


"""
Check the nuclear norm condition
"""
def nuclear_norm_condition(alpha, r, d1, d2, D, k):
    """

    :param alpha: 
    :param r: 
    :param d1: 
    :param d2: 
    :param D: 
    :param k: 
    :return: 
    """
    sum_dk1 = np.sum(D[0:k + 1]) - k * D.item(k - 1)
    sum_dk2 = np.sum(D[0:k + 1]) - k * D.item(k)
    radi = alpha * np.sqrt(r * d1 * d2)
    if sum_dk1 <= radi <= sum_dk2:
        print('TASHI')
        return True
    else:
        return False

"""
Function to implement project on a set
"""
def projection_on_set(X, r, alpha):
    """

    :param X: 
    :param r: 
    :param alpha: 
    :return: 
    """

    U, D, V, lam = svd_and_lamdba_x(X, r, alpha)
    lam_vector = np.ones(D.shape) * lam
    diff_vector = D - lam_vector
    diff_vector = diff_vector.clip(min=0)
    diff_matrix = np.diag(diff_vector)
    udv = np.dot(U, np.dot(diff_matrix, V))
    return udv

def bisection_objective(rho, X, Y, omega, s):
    return -1 * log_likelihood(np.dot(rho, X) + np.dot(1 - rho, X + s), Y, omega)


if __name__ == "__main__":
    X = np.matrix('1 2; 3 4; 6 7')
    Y = np.matrix('1 -1; 1 1; -1 1')
    omega = [(0, 1)]
    print(log_likelihood(X, Y, omega))
