import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    """returns the cost and gradient """

    # Unfold the U and W matrices from params
    X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()

    # You need to return the following values correctly
    J = (1.0/2)*(np.sum(((X.dot(Theta.T))*R - Y)**2)) + (Lambda/2.0)*np.sum(Theta**2) + (Lambda/2.0)*np.sum(X**2)

    dJdX = np.zeros(X.shape)
    dJd0 = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        dJdX - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        dJd0 - num_users x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of Theta
    # =============================================================

    dJdX = (X.dot(Theta.T)*R - Y).dot(Theta) + Lambda*X
    dJd0 = (X.dot(Theta.T)*R - Y).T.dot(X) + Lambda*Theta

    grad = np.hstack((dJdX.T.flatten(),dJd0.T.flatten()))

    return J, grad
