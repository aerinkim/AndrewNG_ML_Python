## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m`
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

from matplotlib import use
use('TkAgg')
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from plotDataPoints import plotDataPoints
from displayData import displayData
from show import show

## ================== Part 1: Load Example Dataset  ===================

print 'Visualizing example dataset for PCA.'
#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data = scipy.io.loadmat('ex7data1.mat')
X = data['X']

#  Visualize the example dataset
plt.scatter(X[:, 0], X[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
plt.axis([0.5, 6.5, 2, 8])
plt.axis('equal')
plt.show(block=False)



raw_input('Program paused. Press Enter to continue...')  

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print 'Running PCA on example dataset.'




#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

plt.figure(figsize=(7,5))
plot = plt.scatter(X[:,0], X[:,1], s=30, facecolors='none', edgecolors='b')
plt.title("PCA - Eigenvectors Shown",fontsize=20)
plt.xlabel('x1',fontsize=16)
plt.ylabel('x2',fontsize=16)
plt.grid(True)
# To draw the principal component, 
# draw them starting at the mean of the data
plt.plot([mu[0], mu[0] + 1.5*S[0]*U[0,0]], #x
		 [mu[1], mu[1] + 1.5*S[0]*U[0,1]], #y
		color='red',linewidth=3,
		label='First Principal Component')
plt.plot([mu[0], mu[0] + 1.5*S[1]*U[1,0]], 
		 [mu[1], mu[1] + 1.5*S[1]*U[1,1]],
		color='green',linewidth=3,
		label='Second Principal Component')
leg = plt.legend(loc=4)
plt.show(block=False)





print 'Top eigenvector: '
print ' U(:,1) = %f %f ', U[0,0], U[1,0]
print '(you should expect to see -0.707107 -0.707107)'

raw_input('Program paused. Press Enter to continue...')  




## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m

print 'Dimension reduction on example dataset.'

#  Plot the normalized dataset (returned from pca)
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
plt.axis([-4, 3, -4, 3]) #axis square
plt.axis('equal')
plt.show(block=False)




K = 1
Z = projectData(X_norm, U, K)
print 'Projection of the first example: %f', Z[0]  
print '(this value should be about 1.481274)'

X_rec  = recoverData(Z, U, K)
print 'Approximation of the first example: %f %f'% (X_rec[0, 0], X_rec[0, 1])
print '(this value should be about  -1.047419 -1.047419)'

#  Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], marker='o', color='r', facecolor='none', lw=1.0)
for i in range(len(X_norm)):
	plt.plot([X_norm[i, 0], X_rec[i, 0]], [X_norm[i, 1], X_rec[i, 1]], '--k')

plt.show(block=False)
raw_input('Program paused. Press Enter to continue...')  




## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print 'Loading face dataset.'

#  Load Face dataset
data = scipy.io.loadmat('ex7faces.mat')
X = data['X']

import matplotlib.cm as cm 
	
def ReshapeIntoImage(row, width):
	height = width
	square_image = row.reshape(width,height)
	return square_image.T	

def displayFaceData(X, rows, cols):
	"""
	Select the first K(100) features from X, creates an image based on reduced pixels. 
	"""
	width, height = 32, 32 # This is the shape of original photo (32*32)
	pictures_combined = np.zeros((height*rows, width*cols))
	
	row, col = 0, 0
	for a_picture_index in xrange(rows*cols):
		if col == cols:
			row += 1
			col  = 0
		a_picture = ReshapeIntoImage(X[a_picture_index],width)
		pictures_combined[row*height:(row*height+a_picture.shape[0]), col*width:(col*width+a_picture.shape[1])] = a_picture
		col += 1

	fig = plt.figure(figsize=(10,10))
	img = scipy.misc.toimage( pictures_combined )
	plt.imshow(img,cmap = cm.Greys_r)
	plt.show(block=False)


def displayProjectedFace100(X, rows, cols): #rows=10, cols=10
	"""
	display 10*10 image
	"""
	width, height = 10, 10 
	pictures_combined = np.zeros((height*rows, width*cols)) #(100,100)
	row, col = 0, 0
	for a_picture_index in xrange(rows*cols):
		if col == cols:
			row += 1
			col  = 0
		a_picture = X[a_picture_index].reshape(10,10).T
		pictures_combined[row*height:(row*height+a_picture.shape[0]), col*width:(col*width+a_picture.shape[1])] = a_picture
		col += 1
	fig = plt.figure(figsize=(10,10))
	img = scipy.misc.toimage( pictures_combined )
	plt.imshow(img,cmap = cm.Greys_r)
	plt.show(block=False)



#  Normalize before running PCA!
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

########################################
# Displaying Eigenvectors
########################################

#  Display the first 100 faces in the dataset
displayFaceData(X[0:100], rows=10, cols=10)

# How does the eigen vector look like?
# Let's visualize eigenvectors 
displayFaceData(U.T, rows=32, cols=32)

# Let's visualize top 49
displayFaceData(U[:,:49].T, rows=7, cols=7)

# Ok, what about top 9
displayFaceData(U[:,:9].T, rows=3, cols=3)



########################################
# Displaying Projected Faces
########################################
# Project images to the eigen space using the top k eigenvectors 
# Test with different K

K = 100
projectedFace = projectData(X_norm, U, K)
print 'The projected data Z has a size of: '
print '%d %d'% projectedFace.shape







# Display normalized data
plt.subplot(1, 2, 1)
displayFaceData(X_norm[:9],3,3)
plt.title('Original faces')
plt.axis('equal')
plt.show(block=False)

plt.subplot(1, 2, 2)
displayProjectedFace(projectedFace,3,3)



# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayFaceData(X_rec[:9],3,3)
plt.title('Recovered faces')
plt.axis('equal')
plt.show(block=False)
raw_input('Program paused. Press Enter to continue...')  



from matplotlib import pyplot as plt
plt.imshow(data, interpolation='nearest')
plt.show()





## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first


means, stds, A_norm = featureNormalize(A)
# Run SVD
U, S, V = getUSV(A_norm)




A = scipy.misc.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

A = A / 255.0
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.random(1000) * len(X)) + 1

#  Setup Color Palette

#  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Xs = np.array([X[int(s)] for s in sel])
xs = Xs[:, 0]
ys = Xs[:, 1]
zs = Xs[:, 2]
cmap = plt.get_cmap("jet")
idxn = sel.astype('float')/max(sel.astype('float'))
colors = cmap(idxn)
# ax = Axes3D(fig)
ax.scatter3D(xs, ys, zs=zs, edgecolors=colors, marker='o', facecolors='none', lw=0.4, s=10)

plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show(block=False)

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S, V = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.figure()
zs = np.array([Z[int(s)] for s in sel])
idxs = np.array([idx[int(s)] for s in sel])

# plt.scatter(zs[:,0], zs[:,1])
plotDataPoints(zs, idxs)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show(block=False)





