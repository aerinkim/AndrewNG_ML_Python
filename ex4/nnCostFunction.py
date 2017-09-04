import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
import itertools
import scipy

data = scipy.io.loadmat('ex4data1.mat')
X = data['X']
y = data['y']
m, _ = X.shape
n_training_samples = X.shape[0]

input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10 
n_training_samples = X.shape[0]

def flattenParams(thetas_list, input_layer_size, hidden_layer_size, output_layer_size):
	"""
	Hand this function a list of theta matrices, and it will flatten it
	into one long (n,1) shaped numpy array
	"""
	theta1=thetas_list[0]
	theta2=thetas_list[1]
	return np.r_[theta1.ravel(), theta2.ravel()].reshape(len(theta1.ravel())+len(theta2.ravel()),1)

def reshapeParams(flattened_array):
    theta1 = flattened_array[:(input_layer_size+1)*hidden_layer_size] \
            .reshape((hidden_layer_size,input_layer_size+1))
    theta2 = flattened_array[(input_layer_size+1)*hidden_layer_size:] \
            .reshape((output_layer_size,hidden_layer_size+1))
    
    return [ theta1, theta2 ]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples,input_layer_size))


def computeCost(mythetas_flattened,myX_flattened,myy,mylambda=0.):
    """
    This function takes in:
        1) a flattened vector of theta parameters (each theta would go from one
           NN layer to the next), the thetas include the bias unit.
        2) the flattened training set matrix X, which contains the bias unit first column
        3) the label vector y, which has one column
    It loops over training points (recommended by the professor, as the linear
    algebra version is "quite complicated") and:
        1) constructs a new "y" vector, with 10 rows and 1 column, 
            with one non-zero entry corresponding to that iteration
        2) computes the cost given that y- vector and that training point
        3) accumulates all of the costs
        4) computes a regularization term (after the loop over training points)
    """
    
    # First unroll the parameters
    mythetas = reshapeParams(mythetas_flattened)
    
    # Now unroll X
    myX = reshapeX(myX_flattened)
    
    #This is what will accumulate the total cost
    total_cost = 0.
    
    m = n_training_samples

    # Loop over the training points (rows in myX, already contain bias unit)
    for irow in xrange(m):
        myrow = myX[irow]
                
        # First compute the hypothesis (this is a (10,1) vector
        # of the hypothesis for each possible y-value)
        # propagateForward returns (zs, activations) for each layer
        # so propagateforward[-1][1] means "activation for -1st (last) layer"
        myhs = propagateForward(myrow,mythetas)[-1][1]

        # Construct a 10x1 "y" vector with all zeros and only one "1" entry
        # note here if the hand-written digit is "0", then that corresponds
        # to a y- vector with 1 in the 10th spot (different from what the
        # homework suggests)
        tmpy  = np.zeros((10,1))
        tmpy[myy[irow]-1] = 1
        
        # Compute the cost for this point and y-vector
        mycost = -tmpy.T.dot(np.log(myhs))-(1-tmpy.T).dot(np.log(1-myhs))
     
        # Accumulate the total cost
        total_cost += mycost
  
    # Normalize the total_cost, cast as float
    total_cost = float(total_cost) / m
    
    # Compute the regularization term
    total_reg = 0.
    for mytheta in mythetas:
        total_reg += np.sum(mytheta*mytheta) #element-wise multiplication
    total_reg *= float(mylambda)/(2*m)
        
    return total_cost + total_reg

def flattenParams(thetas_list, input_layer_size, hidden_layer_size, output_layer_size):
	"""
	Hand this function a list of theta matrices, and it will flatten it
	into one long (n,1) shaped numpy array
	"""
	flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
	combined = list(itertools.chain.from_iterable(flattened_list))
	assert len(combined) == (input_layer_size+1)*hidden_layer_size + \
							(hidden_layer_size+1)*output_layer_size
	return np.array(combined).reshape((len(combined),1))


def propagateForward(row,Thetas):
	"""
	Function that given a list of Thetas (NOT flattened), propagates the
	row of features forwards, assuming the features ALREADY
	include the bias unit in the input layer, and the 
	Thetas also include the bias unit

	The output is a vector with element [0] for the hidden layer,
	and element [1] for the output layer
		-- Each element is a tuple of (zs, as)
		-- where "zs" and "as" have shape (# of units in that layer, 1)
	
	***The 'activations' are the same as "h", but this works for many layers
	(hence a vector of thetas, not just one theta)
	Also, "h" is vectorized to do all rows at once...
	this function takes in one row at a time***
	"""

	features = row #1*400
	zs_as_per_layer = []
	for i in xrange(len(Thetas)):  
		Theta = Thetas[i]
		#Theta is (25,401), features are (401, 1)
		#so "z" comes out to be (25, 1)
		#this is one "z" value for each unit in the hidden layer
		#not counting the bias unit
		z = Theta.dot(features).reshape((Theta.shape[0],1))
		a = sigmoid(z)
		zs_as_per_layer.append( (z, a) )
		if i == len(Thetas)-1:
			return np.array(zs_as_per_layer)
		a = np.insert(a,0,1) #Add the bias unit
		features = a


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
		"""computes the cost and gradient of the neural network. The
	parameters for the neural network are "unrolled" into the vector
	nn_params and need to be converted back into the weight matrices.
	The returned parameter grad should be a "unrolled" vector of the
	partial derivatives of the neural network.
		"""
	# Obtain Theta1 and Theta2 back from nn_params
		# hidden_layer_size =25, input_layer_size = 400
		#										25, 401
		Theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
		Theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,(hidden_layer_size+1))
		m, _ = X.shape    

	# Part 1: Feedforward the neural network and return the cost in the
	#         variable J. After implementing Part 1, you can verify that your
	#         cost function computation is correct by verifying the cost computed in ex4.m
		a1 = X # 5000 * 400
		a1_= np.column_stack((np.ones((m, 1)), X)) # 5000 * 401
		z2 = Theta1.dot(a1_.T) # Theta1 shape: (25, 401) -->  25 * 5000 
		a2 = np.column_stack((np.ones((z2.T.shape[0], 1)), sigmoid(z2.T) ))  # (5000, 26) # after sigmoid you add 1's
		z3 = Theta2.dot(a2.T) # Theta2 shape: (10, 26) --> 
		a3 = sigmoid(z3) #  (10, 5000)
		a3_ = a3.T # (5000, 10)

		y_ = np.zeros((X.shape[0],10)) # (5000, 10)
		for i in xrange(m):
			y_[i][y[i]-1]=1  # index 9 is zero. 

		J = 1.0/m * np.sum(  np.sum( np.multiply( -y_, np.log(a3_) ) - np.multiply(1-y_, np.log(1-a3_)) , 0) )\
			+Lambda/(2.0*m)* ( np.sum( np.square(Theta1[:,1:])) + np.sum( np.square(Theta2[:,1:]))     )
		#								 5000*10,  5000*10

# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.

		#Note: the Delta matrices should include the bias unit
		#The Delta matrices have the same shape as the theta matrices
		Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
		Delta2 = np.zeros((num_labels,hidden_layer_size+1))

		# Loop over the training points (rows in myX, already contain bias unit)
		for irow in xrange(m):
			myrow = a1_[irow]
			a1 = myrow.reshape((input_layer_size+1,1))
			# propagateForward returns (zs, activations) for each layer excluding the input layer
			temp = propagateForward(myrow,[Theta1,Theta2])
			z2 = temp[0][0]
			a2 = temp[0][1]
			z3 = temp[1][0]
			a3 = temp[1][1]
			delta3 = a3 - y_[irow].reshape(a3.shape[0],1) 
			delta2 = Theta2.T[1:,:].dot(delta3)*sigmoidGradient(z2) #remove 0th element
			a2 = np.insert(a2,0,1,axis=0)
			Delta1 += delta2.dot(a1.T) #(25,1)x(1,401) = (25,401) (correct)
			Delta2 += delta3.dot(a2.T) #(10,1)x(1,25) = (10,25) (should be 10,26)
			
		D1 = Delta1/float(m)
		D2 = Delta2/float(m)
		
		#Regularization:
		D1[:,1:] = D1[:,1:] + (float(Lambda)/m)*Theta1[:,1:]
		D2[:,1:] = D2[:,1:] + (float(Lambda)/m)*Theta2[:,1:]		

		"""Vectorized version
		d3 = a3_ - y_ # 5000x10
		d2 = np.dot(Theta2[:,1:].T, d3.T ) * sigmoidGradient(z2) 
		  # 25x10 *10x5000 * 25x5000 = 25x5000
		
		#why isn't this theta1 dot delta2?
		delta1 = d2.dot(a1) # 25x5000 * 5000x401 = 25x401 
		delta2 = d3.T.dot(a2) # 10x5000 *5000x26 = 10x26
		
		theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
		theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
		
		theta1_grad = delta1/m + (theta1_*reg)/m
		theta2_grad = delta2/m + (theta2_*reg)/m
		"""		
		# Unroll gradient
		# grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))

# Part 3: Implement regularization with the cost function and gradients.
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
		return J, flattenParams([D1, D2],input_layer_size, hidden_layer_size, num_labels).flatten()
