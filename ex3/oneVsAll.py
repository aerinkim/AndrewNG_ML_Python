import numpy as np
from scipy.optimize import minimize
from scipy import optimize
from lrCostFunction import lrCostFunction
from gradientFunctionReg import gradientFunctionReg
from sigmoid import *

def h(X,theta):
    return sigmoid(X.dot(theta))

def gradientFunction(theta, X, y):
    m = len(y)   # number of training examples
    return 1.0/m* X.T.dot(h0-y)

def computeCost(theta, X, y, Lambda =0.):
    m = len(y) 
    return (1.0/m) *np.sum(-y * np.log( sigmoid(X.dot(theta))) -(1-y) * np.log(1-sigmoid(X.dot(theta))))

def gradientFunction(theta, X, y, Lambda =0.):
    m = len(y)   # number of training examples
    h0 = h(X, theta) 
    return 1.0/m* X.T.dot(h0-y)

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin_cg(computeCost, fprime=gradientFunction, x0=mytheta, \
                              args=(myX, myy, mylambda), maxiter=50, disp=False,\
                              full_output=True)
    return result[0], result[1]

def oneVsAll(X, y, num_labels, Lambda=0.):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """
    #m, n = X.shape
    #all_theta = np.zeros((num_labels, n + 1))
    #X = np.column_stack((np.ones((m, 1)), X))

# Hint: theta(:) will return a column vector.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.
    """
    Function that determines an optimized theta for each class
    and returns a Theta function where each row corresponds
    to the learned logistic regression params for one class
    """
    print 'Training One-vs-All Logistic Regression...'

    initial_theta = np.zeros((X.shape[1],1)).reshape(-1)
    Theta = np.zeros((10,X.shape[1]))
    for i in xrange(10):
        iclass = i if i else 10 #class "10" corresponds to handwritten zero
        print "Optimizing for handwritten number %d..."%i
        logic_Y = np.array([1 if x == iclass else 0 for x in y])#.reshape((X.shape[0],1))
        itheta, imincost = optimizeTheta(initial_theta,X,logic_Y,Lambda)
        Theta[i,:] = itheta
    print "Done!"
    return Theta



