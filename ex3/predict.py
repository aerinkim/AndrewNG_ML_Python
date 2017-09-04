import numpy as np

from sigmoid import sigmoid

def propagateForward(Arow,Theta_list):
	features = Arow
	for i in xrange(len(Theta_list)):
		Theta = Theta_list[i]
		z = Theta.dot(features)
		a = sigmoid(z)
		if i == len(Theta_list)-1:
			return a
		a = np.insert(a,0,1) #Add the bias unit
		features = a
		
def predictNN(row,Theta_list):
	"""
	Function that takes a row of features, propagates them through the
	NN, and returns the predicted integer that was hand written
	"""
	classes = range(1,10) + [10]
	output = propagateForward(row,Theta_list)
	return classes[np.argmax(np.array(output))]



def predict(Theta1, Theta2, X,y):
	""" outputs the predicted label of X given the
	trained weights of a neural network (Theta1, Theta2)
	"""
	# "You should see that the accuracy is about 97.5%"
	pred =np.zeros(y.shape)
	Theta_list = [ Theta1, Theta2 ]

	# Add ones to the X data matrix
	m=X.shape[0]
	X = np.column_stack((np.ones((m, 1)), X))
	print "X shape:",X.shape

	n_correct, n_total = 0., 0.
	incorrect_indices = []
	
	for irow in xrange(X.shape[0]):
		n_total += 1
		pred[irow]=(predictNN(X[irow],Theta_list))
		if predictNN(X[irow],Theta_list) == int(y[irow]): 
			n_correct += 1
		else: incorrect_indices.append(irow)
	print "Training set accuracy: %0.1f%%"%(100*(n_correct/n_total))
	return pred       # add 1 to offset index of maximum in A row

