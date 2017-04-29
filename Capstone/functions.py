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



if __name__ == "__main__":
    X = np.matrix('1 2; 3 4; 6 7')
    Y = np.matrix('1 -1; 1 1; -1 1')
    omega = [(0, 1)]
    print(log_likelihood(X, Y, omega))
