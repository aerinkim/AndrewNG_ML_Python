import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def polyFeatures(X, p):
	"""takes a data matrix X (size m x 1) and maps each example into its polynomial features where
	X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
	"""
	poly = PolynomialFeatures(degree=p)
	X_train_poly = poly.fit_transform(X.reshape(-1,1))
	return X_train_poly
	"""
	newX = X.copy()
	for i in xrange(p):
		dim = i+2
		newX = np.insert(newX,newX.shape[1],np.power(newX[:,1],dim),axis=1)
	return newX
	"""