import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None
 
    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        
        self.linear_discriminants=np.zeros((len(X[0]*len(X[0])),self.n_components)) # Modify as required 
        d_2 = X.shape[1]*X.shape[2]
        n_samples = X.shape[0]
        X_reshaped = X.reshape(n_samples, -1)
        mean_total = np.mean(X_reshaped, axis=0)
        Sb = np.zeros((d_2, d_2))
        Sw = np.zeros((d_2, d_2))
        unique_classes = np.unique(y)

        for class_label in unique_classes:
            X_class = X_reshaped[y == class_label]
            mean_class = np.mean(X_class, axis=0)
            Sw += (X_class - mean_class).T.dot(X_class - mean_class)
            mean_diff = (mean_class - mean_total).reshape(-1, 1)
            Sb += X_class.shape[0] * mean_diff.dot(mean_diff.T)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

        eig_pairs = []
        for i in range(len(eigenvalues)):
            eig_pairs.append((np.abs(eigenvalues[i]), eigenvectors[:,i]))
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        linear_discriminants = []
        for i in range(0, self.n_components):
            linear_discriminants.append(eig_pairs[i][1].reshape(d_2, 1))
        self.linear_discriminants = np.hstack(linear_discriminants)
        return self.linear_discriminants  
    
    
    
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        return np.dot(X.reshape(len(X),-1),w)
        # END TODO

