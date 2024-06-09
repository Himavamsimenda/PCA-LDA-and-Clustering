import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    N, a, b = X.shape
    X_flat = X.reshape(N, -1)
    calculated_mean = np.mean(X_flat, axis=0)
    centered_data = X_flat - calculated_mean
    covariance_m = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_m)
    index = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, index]
    basis_vector = eigenvectors[:, :k]
    return basis_vector
    pass
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    N, a, b = X.shape
    X_flattened = X.reshape(N, -1)
    projections = np.dot(X_flattened, basis)
    return projections
    pass
    # END TODO
    

    
    