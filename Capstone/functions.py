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
  sum = 0
  for i, j in omega:
    elem = 0
    x_ij = X.item((i, j))
    y_ij = Y.item((i, j))
    logistic_x_ij = logistic(x_ij)
    if y_ij == 1:
      elem = math.log(logistic_x_ij, math.e)
    elif y_ij == -1:
      elem = math.log(1 - logistic_x_ij, math.e)
    else:
      print("What is wrong with you, man! Y should be either 1 or -1, goddamn it")
    sum += elem

  return sum

"""
Gradient: Function that calculates gradient of log-likelihood at given point(Matrix) M.
"""

if __name__ == "__main__":
  X = np.matrix('1 2; 3 4; 6 7')
  Y = np.matrix('1 -1; 1 1; -1 1')
  omega = [(0, 1)]
  print(log_likelihood(X, Y, omega))