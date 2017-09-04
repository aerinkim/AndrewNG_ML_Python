import matplotlib.pyplot as plt
import numpy as np

def plotData(X,y):
	"""
	plots the data points and gives the figure axes labels of
	population and profit.
	"""
# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);


	plt.figure(figsize=(10,6))
	plt.plot(X[:,1],y,'rx',markersize=10)
	plt.grid(True) #Always plot.grid true!
	plt.ylabel('Profit in $10,000s')
	plt.xlabel('Population of City in 10,000s')

	plt.figure()  # open a new figure window
	#plt.show()
# ============================================================



#Plot the convergence of the cost function
def plotConvergence(J_history, iterations):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(J_history)),J_history,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*iterations,1.05*iterations])
    #dummy = plt.ylim([4,8])









