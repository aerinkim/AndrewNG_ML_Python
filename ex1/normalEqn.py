import numpy as np


def normalEqn(X,y):
	""" Computes the closed-form solution to linear regression
	   normalEqn(X,y) computes the closed-form solution to linear
	   regression using the normal equations.
	"""
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#	
	a = np.linalg.inv( (np.dot(X.T,X)))
	return np.dot(np.dot(a, X.T), y)

# ============================================================

