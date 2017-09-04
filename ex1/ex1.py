from matplotlib import use, cm
use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#import importlib
#importlib.import_module('mpl_toolkits.mplot3d').Axes3D
from mpl_toolkits.mplot3d import axes3d
from sklearn import linear_model

from gradientDescent import gradientDescent
from computeCost import computeCost
from warmUpExercise import warmUpExercise
from plotData import *
from show import show

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print 'Running warmUpExercise ...'
print '5x5 Identity Matrix:'
warmup = warmUpExercise()
print warmup
raw_input("Program paused. Press Enter to continue...")

# ======================= Part 2: Plotting =======================
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = np.vstack(zip(np.ones(m),data[:,0]))
y = data[:, 1]

# Plot Data
# Note: You have to complete the code in plotData.py
print 'Plotting Data ...'
plotData(X,y)
plt.show()

raw_input("Program paused. Press Enter to continue...")

# =================== Part 3: Gradient descent ===================
print 'Running Gradient Descent ...'

#Later We can update 2 to 3, 4, 5 for multivariate. 
theta = np.zeros(2)

# compute and display initial cost
J = computeCost(X, y, theta)
print 'cost: %0.4f ' % J

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent
theta, theta_history, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print 'Theta found by gradient descent: '
print '%s %s \n' % (theta[0], theta[1])

# Plot the J converge by every iteration.
plotConvergence(J_history, iterations) 
dummy = plt.ylim([4,7])
plt.show(block=False)


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print 'Visualizing J(theta_0, theta_1) ...'

#Import necessary matplotlib tools for 3d plots
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

# Grid over which we will calculate J
theta0_vals = np.arange(-10, 10, 0.5) # xvals 
theta1_vals = np.arange(-1, 4, 0.1) # yvals

myxs, myys, myzs = [], [], []
for david in theta0_vals:
    for kaleko in theta1_vals:
        myxs.append(david)
        myys.append(kaleko)
        myzs.append(computeCost(X,y , np.array([[david], [kaleko]])))

scat = ax.scatter(myxs,myys,myzs,c=np.abs(myzs),cmap=plt.get_cmap('YlOrRd'))

plt.xlabel(r'$\theta_0$',fontsize=20)
plt.ylabel(r'$\theta_1$',fontsize=20)
plt.title('Cost (Minimization Path Shown in Blue)',fontsize=20)
plt.plot([x[0] for x in theta_history],[x[1] for x in theta_history],J_history,'bo-')
plt.show(block=False)



"""
# Create grid coordinates for plotting
B0 = np.linspace(-10, 10, 50)
B1 = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))
# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = computeCost(X,y, theta=[[xx[i,j]], [yy[i,j]]])
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
# Contour plot
CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')
plt.show(block=False)


# Contour plot
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.contour(xx, yy, Z, np.logspace(-2, 3, 20))
plt.clabel(ax, inline=1, fontsize=10)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(0.0, 0.0, 'rx', linewidth=2, markersize=10)
plt.show(block=False)
"""

"""
# =============Use Scikit-learn =============

regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
regr.fit(X, y)

print 'Theta found by scikit: '
print '%s %s \n' % (regr.coef_[0], regr.coef_[1])

predict1 = np.array([1, 3.5]).dot(regr.coef_)
predict2 = np.array([1, 7]).dot(regr.coef_)
print 'For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000)
print 'For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000)

plt.figure()
plotData(data)
plt.plot(X[:, 1],  X.dot(regr.coef_), '-', color='black', label='Linear regression wit scikit')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()

raw_input("Program paused. Press Enter to continue...")
"""

