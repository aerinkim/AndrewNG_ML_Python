from matplotlib import use
use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from featureNormalize import featureNormalize
from show import show
from plotData import *
# ================ Part 1: Feature Normalization ================

print 'Loading data ...'

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

X = np.insert(X,0,1,axis=1)


# Print out some data points
print 'First 10 examples from the dataset:'
print np.column_stack( (X[:10], y[:10]) )


plt.grid(True)
plt.xlim([-100,5000])
dummy = plt.hist(X[:,0],label = 'col1')
dummy = plt.hist(X[:,1],label = 'col2')
dummy = plt.hist(X[:,2],label = 'col3')
plt.title('Clearly we need feature normalization.')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show(block=False)

# Scale features and set them to zero mean
print 'Normalizing Features ...'

Xnorm, mu, sigma = featureNormalize(X)
print '[mu] [sigma]'
print mu, sigma


#Quick visualize the feature-normalized data
plt.grid(True)
plt.xlim([-5,5])
dummy = plt.hist(Xnorm[:,0],label = 'col1')
dummy = plt.hist(Xnorm[:,1],label = 'col2')
dummy = plt.hist(Xnorm[:,2],label = 'col3')
plt.title('Feature Normalization Accomplished')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show(block=False)


# ================ Part 2: Gradient Descent ================
#
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print 'Running gradient descent ...'

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)
theta, theta_history, J_history = gradientDescentMulti(Xnorm, y, theta, alpha, num_iters)

# Plot the convergence graph
plotConvergence(J_history,num_iters)
#dummy = plt.ylim([4,7])
plt.show(block=False)

# Display gradient descent's result
print 'Theta computed from gradient descent: '
print theta

# Estimate the price of a 1650 sq-ft, 3 br house
#We did: Xnorm, mu, sigma = featureNormalize(X)


test = np.array([ 1650, 3])
scaled = (test -mu[1:] )/ sigma[1:]
scaled = np.insert(scaled, 0,1)
price = scaled.dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ',price

raw_input("Program paused. Press Enter to continue...")

# ================ Part 3: Normal Equations ================

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

print 'Solving with normal equations...'

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size

# Add intercept term to X
X = np.concatenate((np.ones((m,1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print 'Theta computed from the normal equations:'
print ' %s \n' % theta

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650,3]).dot(theta)

# ============================================================

print "Predicted price of a 1650 sq-ft, 3 br house "
print '(using normal equations):\n $%0.2f\n' % price

raw_input("Program paused. Press Enter to continue...")
