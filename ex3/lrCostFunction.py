import numpy as np
from costFunctionReg import costFunctionReg
from sigmoid import sigmoid 

def lrCostFunction(theta, X, y, Lambda):
	"""computes the cost of using
	theta as the parameter for regularized logistic regression and the
	gradient of the cost w.r.t. to the parameters.
	"""
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations. 
#
# =============================================================
	m = X.shape[0]
	return (1.0/m) *np.sum(-y * np.log( sigmoid(X.dot(theta))) -(1-y)* np.log(1-sigmoid(X.dot(theta)))) + (Lambda/(2.0*m))*np.sum( theta[1:]**2 )
