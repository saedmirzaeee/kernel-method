import numpy as np
import cvxopt
import cvxopt.solvers
from utils import *


class Kernel_svm_model():
    """
        Implementof SVM Kernel
    """
    def __init__(self,  kernel_matrix, regularisation_constant=10, threshold=1e-5):
        """
        kernel_matrix : is the kernel(.e.i gussian, spectrum)
        regularisation_constant : the svm regularisation constant
        threshold: value to specify if alpha is a support vector or not
        """
        self.kernel_matrix = kernel_matrix
        self.regularisation_constant = regularisation_constant
        self.threshold = threshold


    def fit(self, X, y):
        """
        X: training input
        y: training lables
        """

        self.fit_index = np.array(list(range(len(X))))
        
        self.kernel_fit = self.kernel_matrix
        
        self.n = self.kernel_fit.shape[0]
        
        self.y_fit = y
        
        r, o, z = np.arange(self.n), np.ones(self.n), np.zeros(self.n)
        
        
        
        # Define the quadratic optimization problem

        P = cvxopt.matrix(self.kernel_fit.astype(float), tc='d')
        
        q = cvxopt.matrix(-self.y_fit, tc='d')

        G = cvxopt.spmatrix(np.r_[self.y_fit, -self.y_fit], np.r_[r, r + self.n], np.r_[r, r], tc='d')
        
        h = cvxopt.matrix(np.r_[o * self.regularisation_constant, z], tc='d')
        
        
        # Solve the quadratic optimization problem using cvxopt

        minimization = cvxopt.solvers.qp(P, q, G, h)
        
        self.lagr_mult = np.ravel(minimization['x'])
    
        self.support_vector_index = np.where(np.abs(self.lagr_mult) > self.threshold)
        
        self.y_fit = self.y_fit[self.support_vector_index]
        
        self.lagr_mult = self.lagr_mult[self.support_vector_index]
        
        self.support_vector_index = self.fit_index[self.support_vector_index]
        
        self.y_hat = []
        for i in self.support_vector_index:
            self.y_hat.append(np.dot(self.lagr_mult, self.kernel_matrix[self.support_vector_index, i]).squeeze())
        self.y_hat = np.array(self.y_hat)

        self.b = np.mean(self.y_fit - self.y_hat)


    def predict(self, K_test):
        """
        K_test : features
        """
        predicted_value = []
        for i in range(len(K_test[0])):
            predicted_value.append(np.sign(np.dot(self.lagr_mult, K_test[self.support_vector_index, i].squeeze()) + self.b))
        
        return np.array(predicted_value)