import numpy as np


def findClosestCentroids(X, centroids):
	"""returns the closest centroids
	in idx for a dataset X where each row is a single example. idx = m x 1
	vector of centroid assignments (i.e. each entry in range [1..K])
	"""

# You need to return the following variables correctly.
	idx = np.zeros(X.shape[0])

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every example, find its closest centroid, and store
#               the index inside idx at the appropriate location.
#               Concretely, idx(i) should contain the index of the centroid
#               closest to example i. Hence, it should be a value in the 
#               range 1..K

	for i in xrange(len(X)):
		min_di=99999999
		for j in xrange(len(centroids)):
			if np.sum((X[i]-centroids[j])**2) < min_di:
				min_di=np.sum((X[i]-centroids[j])**2)
				idx[i]=int(j)				
	return idx

