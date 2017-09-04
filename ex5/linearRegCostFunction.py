import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
	"""computes the
	cost of using theta as the parameter for linear regression to fit the
	data points in X and y. Returns the cost in J and the gradient in grad
	"""
	
	m, _ = X.shape
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
	
	J = 1/(2.0*m) * np.sum(( np.dot(X,theta)-y)**2) + (Lambda/(2.0*m))*np.sum(Lambda*Lambda)
	H= np.dot(X,theta)-y 
	grad = 1/float(m) * np.sum( (H.reshape(m,1) )*X,axis=0 ) + Lambda/float(m) * Lambda

	return J, grad