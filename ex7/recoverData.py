import numpy as np
def recoverData(Z, U, K):
    # Compute the approximation of the data by projecting back onto the original space using the top K eigenvectors in U.
    new_U=U[:, :K]
    return Z.dot(new_U.T) # We can use transpose instead of inverse because U is orthogonal.
