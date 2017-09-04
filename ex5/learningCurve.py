import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def learningCurve(X, y, Xval, yval, Lambda):
    """returns the train and
    cross validation set errors for a learning curve. In particular,
    it returns two vectors of the same length - error_train and
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
    """
    m, _ = X.shape

    error_train = np.zeros(m)
    error_val   = np.zeros(m)
    
    for i in range(1,m):
        theta = trainLinearReg(X[:i], y[:i], 0)
        error_train[i],_ = linearRegCostFunction(X[:i], y[:i], theta, 0)
        error_val[i],_ = linearRegCostFunction(Xval, yval, theta, 0)

# Note: You should evaluate the training error on the first i training
#       examples (i.e., X(1:i, :) and y(1:i)).
#       For the cross-validation error, you should instead evaluate on
#       the _entire_ cross validation set (Xval and yval).
# Note: If you are using your cost function (linearRegCostFunction)
#       to compute the training and cross validation error, you should 
#       call the function with the lambda argument set to 0. 
#       Do note that you will still need to use lambda when running
#       the training to obtain the theta parameters.
    return error_train, error_val








