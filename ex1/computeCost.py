import numpy as np

def computeCost(X, y,theta):
	"""
	   computes the cost of using theta as the parameter for linear 
	   regression to fit the data points in X and y
	"""

	m = y.size

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
# =========================================================================
	y_hat = np.dot(X,theta)
	J = np.sum (1.0/(2*m) * (np.square(y_hat - y)))
	return J


