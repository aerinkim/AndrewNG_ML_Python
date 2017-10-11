import numpy as np

def recoverData(Z, U, K):
    # Compute the approximation of the data by projecting back onto 
    # the original space using the top K eigenvectors in U.
    U= U[:,:K]
    return Z.dot(U.T.dot((U.dot(U.T)).T))
