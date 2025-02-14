import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # prob = sigmoid(np.array([1, 45, 85]).dot(theta))

    m = y.size # number of training examples
    J = (1.0/m) *np.sum(-y.values.flatten() * np.log( sigmoid(X.dot(theta))) -(1-y) * np.log(1-sigmoid(X.dot(theta))))
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
    return J
