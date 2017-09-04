from sigmoid import sigmoid
from numpy import squeeze, asarray
import numpy as np
from costFunction import *

def h(X,theta):
	return sigmoid(X.dot(theta))


def gradientFunction(theta, X, y):
	"""
	Compute cost and gradient for logistic regression.
	"""

	m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
# =============================================================
	h0 = h(X, theta) 

	return 1.0/m* X.T.dot(h0-y)
