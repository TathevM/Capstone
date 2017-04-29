"""
All important functions that we are going to use are in this file.
"""
import numpy as np
import math

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
    return math.exp(x) / (1 + np.exp(x))


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
    gradient_log_lik = 0
    for i, j in omega:
        elem = 0
        grad_elem = 0
        x_ij = X.item((i, j))
        y_ij = Y.item((i, j))
        logistic_x_ij = logistic(x_ij)
        if y_ij == 1:
            elem = math.log(logistic_x_ij, math.e)
            if x_ij is not 0:
                grad_elem = 1 / (math.exp(x_ij) + 1)
        elif y_ij == -1:
            temp = 1 - logistic_x_ij
            elem = math.log(temp, math.e)
            if temp is not 0:
                grad_elem = -(math.exp(x_ij) / (math.exp(x_ij) + 1))
        else:
            print("What is wrong with you, man! Y should be either 1 or -1, goddamn it")
        log_lik += elem
        gradient_log_lik += grad_elem

    return log_lik, gradient_log_lik

"""
Lambda function, computes lambda for Projection function
"""
def lamdba_x(X, r, alpha):
    """

    :param X: 
    :param r: 
    :param alpha: 
    :return: 
    """
    U, D, V = np.linalg.svd(X)
    d1, d2 = X.shape
    for k in range(0, r + 1):
        if nuclear_norm_condition(alpha, r, d1, d2, D, k):
            lam = (np.sum(D[0:k]) - alpha * np.sqrt(r * d1 * d2)) / k
            return U, D, V, lam


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
    sum_dk1 = np.sum(D[0:k]) - k * D.item(k + 1)
    sum_dk2 = np.sum(D[0:k]) - k * D.item(k)
    radi = alpha * np.sqrt(r * d1 * d2)
    if sum_dk1 >= radi and sum_dk2 <= radi:
        return True
    else:
        return False


def projection_on_set(X, r, alpha):
    U, D, V, lam = lamdba_x(X, r, alpha)
    lam_vector = np.ones(D.shape) * lam
    diff_vector = D - lam_vector
    diff_vector = diff_vector.clip(min=0)
    return np.dot(np.dot(U, diff_vector), np.transpose(V))



if __name__ == "__main__":
    X = np.matrix('1 2; 3 4; 6 7')
    Y = np.matrix('1 -1; 1 1; -1 1')
    omega = [(0, 1)]
    print(log_likelihood(X, Y, omega))
