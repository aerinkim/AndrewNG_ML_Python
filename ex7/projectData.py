
def projectData(X, U, K):
    # Compute the projection of the data using only the top K eigenvectors in U (first K columns).
    # X: data
    # U: Eigenvectors
    # K: your choice of dimension
    new_U = U[:,:K]
    return X.dot(new_U)