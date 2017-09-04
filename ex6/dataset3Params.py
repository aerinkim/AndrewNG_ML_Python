import numpy as np
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
	"""returns your choice of C and sigma. You should complete
	this function to return the optimal C and sigma based on a
	cross-validation set.
	"""
# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
# =========================================================================
	# The score() function for a trained SVM takes in
	# X and y to test the score on, and the (float)
	# value returned is "Mean accuracy of self.predict(X) wrt. y"

	Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
	sigmavalues = Cvalues
	best_pair, best_score = (0, 0), 0

	for Cvalue in Cvalues:
		for sigmavalue in sigmavalues:
			gamma = np.power(sigmavalue,-2.)
			gaus_svm = svm.SVC(C=Cvalue, kernel='rbf', gamma=gamma)
			gaus_svm.fit( X, y.flatten() )
			this_score = gaus_svm.score(Xval,yval)
			#print this_score
			print Cvalue,sigmavalue,this_score
			if this_score > best_score:
				best_score = this_score
				best_pair = (Cvalue, sigmavalue)

					
	print "Best C, sigma pair is (%f, %f) with a score of %f."%(best_pair[0],best_pair[1],best_score)
	return best_pair[0], best_pair[1]
