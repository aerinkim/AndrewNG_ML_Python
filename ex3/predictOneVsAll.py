import numpy as np
from sigmoid import sigmoid
from displayData import displayData

def h(X,theta):
    return sigmoid(X.dot(theta))

def predictOneVsAllHelper(Theta,myrow) :
    Theta = np.insert(Theta, 0, values=0, axis=1) 
    classes = [10] + range(1,10)
    hypots  = [0]*len(classes)
    #Compute a hypothesis for each possible outcome
    #Choose the maximum hypothesis to find result
    for i in xrange(len(classes)):
        hypots[i] = h(myrow,Theta[i])
    return classes[np.argmax(np.array(hypots))]

def predictOneVsAll(all_theta, X,y):
    """will return a vector of predictions
  for each example in the matrix X. Note that X contains the examples in
  rows. all_theta is a matrix where the i-th row is a trained logistic
  regression theta vector for the i-th class. You should set p to a vector
  of values from 1..K (e.g., p = [1 3 1 2] predicts classes 1, 3, 1, 2
  for 4 examples) """

    # Add ones to the X data matrix
    m=X.shape[0]
    X = np.column_stack((np.ones((m, 1)), X))
    print "X shape:",X.shape

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters (one-vs-all).
#               You should set p to a vector of predictions (from 1 to
#               num_labels).
#
# Hint: This code can be done all vectorized using the max function.
#       In particular, the max function can also return the index of the 
#       max element, for more information see 'help max'. If your examples 
#       are in rows, then, you can use max(A, [], 2) to obtain the max 
#       for each row.
# =========================================================================
    pred =np.zeros(y.shape)
    n_correct, n_total = 0., 0.
    incorrect_indices = []
    for irow in xrange(X.shape[0]):
        n_total += 1
        pred[irow]=(predictOneVsAllHelper(all_theta,X[irow]))
        if predictOneVsAllHelper(all_theta,X[irow]) == y[irow]: 
            n_correct += 1
        else: incorrect_indices.append(irow)

    print "Training set accuracy: %0.1f%%"%(100*(n_correct/n_total))
    return pred

