import numpy as np
'''
Homework5: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.eig(): compute the eigen decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

    def __init__(self, X, n_components):
        '''
        Args:
            X: The data matrix of shape [n_samples, n_features].
            n_components: The number of principal components. A scaler number.
        '''

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()

    
    def _do_pca(self):
        '''
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_samples, n_components].
        '''
        ### YOUR CODE HERE

        X_hat = self.X - np.mean(self.X, axis=0)
        S = np.cov(np.transpose(X_hat))
        # print('S shape:', S.shape)

        eigenvals, eigenvecs = np.linalg.eig(S)
        # print('eigenvals shape:', eigenvals.shape)
        # print(eigenvals)

        idx = np.argsort(eigenvals)[::-1]
        Up = eigenvecs[:, idx[:self.n_components]]
        # print('Up shape:', Up.shape)
        
        Xp = np.dot(X_hat, Up)
        # print('Xp shape:', Xp.shape)

        return Up, Xp

        ### END YOUR CODE

    def get_reduced(self):
        '''
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_any, n_features] or None. 
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_any, n_components].
        '''
        return self.Xp

    def reconstruction(self, Xp):
        '''
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        Return:
        X_re: The reconstructed matrix of shape [n_samples, n_features].
        '''
        ### YOUR CODE HERE

        X_re = np.dot(Xp, np.transpose(self.Up)) + np.mean(self.X, axis=0)

        return X_re

        ### END YOUR CODE


def reconstruct_error(A, B):
    '''
    To compute the reconstruction error.

    Args: 
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return: 
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    '''
    ### YOUR CODE HERE

    error = np.linalg.norm(A - B) ** 2

    return error

    ### END YOUR CODE

