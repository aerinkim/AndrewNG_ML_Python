from computeCost import computeCost
import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
	"""
	 Performs gradient descent to learn theta
	   theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
	   taking num_iters gradient steps with learning rate alpha
	"""
	# Initialize some useful values
	theta_history = []
	J_history = np.zeros(num_iters)
	m = y.size  # number of training examples
	for i in range(num_iters):
		# ====================== YOUR CODE HERE ======================
		# Perform a single gradient step on the parameter vector theta.
		# Hint: While debugging, it can be useful to print out the values
		#       of the cost function (computeCost) and gradient here.
		# ============================================================
		# J = np.sum (1.0/(2*m) * (np.square(y_hat - y)))
		y_hat = np.dot(X,theta)
		theta = theta - alpha*(1.0/m)*np.dot(X.T, y_hat-y)  
		# Save the cost J in every iteration
		theta_history.append(theta) 
		J_history[i] = computeCost(X, y, theta)
		if i%100==0:
			print J_history[i]
	return theta, theta_history, J_history