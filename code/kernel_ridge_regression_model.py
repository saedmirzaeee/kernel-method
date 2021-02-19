from functools import partial
import numpy as np
from utils import *
import kernels

"""
    This class implements Kernel Ridge Regression
"""

class Kernel_ridge_regression_model(object):
    def __init__(self):
        pass


    def solve_linear_system(self, K, n, lam, y):
        """
            K = nxn size matrix
            y = n size vector
        """
        I = np.identity(n)
        mat_coef = K + n * lam * I
        alpha = np.linalg.solve(mat_coef, y)

        return alpha


    def predict_labels(self, alpha, K):
        return np.dot(K, alpha)


    def compute_alpha_KRR(self, train_x, train_y, lam, sigma, distrib):
        kernel = partial(self.gaussian_kernel, sigma)

        # Compute training Gram matrix
        K_train = self.kernel_matrix_training(train_x, kernel)

        # Solve the linear system in order to find the vector weights
        alpha = self.solve_linear_system(K_train, len(train_x), lam, train_y)
        alpha = alpha.reshape(len(train_x), 1)

        save_object(alpha, 'alpha_velues_sigma={}_lambda={}_distrib={}'.format(sigma, lam, distrib))

        return alpha


    def run_KRR(self, x_train, y_train, x_test):
        """
            KRR with gaussian kernel
        """
        test_labels = []
        for i in range(3):
            # Compute alpha coefficients using the training set
            alpha = self.compute_alpha_KRR(x_train[i], y_train[i], 0.001, 0.1, i)
            
            # Define the gaussian kernel
            kernel = partial(self.gaussian_kernel, 0.1)
            
            # Predict the labels over the test set
            labels = self.do_predictions(x_train[i], y_train[i], x_test[i], alpha, kernel)
            test_labels = test_labels + labels

            write_labels_csv(test_labels)


    def run_KRR_spectrum(self, x_train, y_train, x_test, distrib):
        kernel_func = partial(np.dot)
        histograms_X_train = kernels.spectrum_histogram(x_train, x_train, 7, distrib)
        gram_matrix = kernels.kernel_matrix_training(histograms_X_train, kernel_func)

        save_object(gram_matrix, 'spectr_kernel_aug_k=7_train_distrib={}'.format(distrib))

        # Solve the linear system in order to find the vector weights
        alpha = self.solve_linear_system(gram_matrix, len(x_train), 0.1, y_train)
        alpha = alpha.reshape(len(x_train),1)

        # Build the Gram matrix for the test data
        histograms_X_test = kernels.spectrum_histogram(x_train, x_test, 7, distrib)
        gram_mat_test = kernels.kernel_matrix_test(histograms_X_train, histograms_X_test, kernel_func)

        save_object(gram_matrix, 'spectr_kernel_aug_k=7_test_distrib={}'.format(distrib))

        # Compute predictions over the test data
        pred = self.predict_labels(alpha, np.matrix.transpose(gram_mat_test))

        # Convert predictions to labels
        pred = array_to_labels(pred, 0)

        return pred

    def do_predictions(self, train_x, train_y, test_x, alpha, kernel):

        # Compute test Gram matrix
        K_test = kernels.kernel_matrix_test(train_x, test_x, kernel)
        labels = self.predict_labels(alpha, np.matrix.transpose(K_test))

        labels = array_to_labels(labels, 0)
        return labels


    def train_folds(self, data, labels, folds):
        """
        docstring
        """

        len_data = len(data)
        data = np.array(list(zip(data, labels)))
        len_fold = int(len(data) / folds)

        lambda_values = [0.1, 0.3, 0.6, 0.9]
        sigma_values = [0.0001, 0.001, 0.01, 0.1, 0.5]

        for lam in lambda_values:
            accuracy_values = []
            for sigma in sigma_values:
                # Build a partial gaussian function with the current 'sigma' value
                kernel_func = partial(self.gaussian_kernel, sigma)
                print('Processing sigma value={}'.format(sigma))

                # TODO Compute the whole gram matrix here only once
                # each fold will extract

                fold_accuracy = 0
                for i in range(folds):
                    # print('Fold: {}'.format(i))
                    # Training data is obtained by concatenating the 2 subsets: at the right + at the left
                    # of the current fold
                    train_data = [*data[0:i*len_fold], *data[(i+1)*len_fold:len_data]]

                    # The current fold is used to test the model
                    test_data = [*data[i*len_fold:(i+1)*len_fold]]
                    
                    x_train = np.array([x[0] for x in train_data])
                    y_train = np.array([x[1] for x in train_data])

                    x_test = np.array([x[0] for x in test_data])
                    y_test = np.array([x[1] for x in test_data])

                    # Build the Gram matrix
                    gram_matrix = kernels.kernel_matrix_training(x_train, kernel_func)

                    # Solve the linear system in order to find the vector weights
                    alpha = self.solve_linear_system(gram_matrix, len(x_train), lam, y_train)
                    alpha = alpha.reshape(len(x_train),1)

                    # Build the Gram matrix for the test data
                    gram_mat_test = kernels.kernel_matrix_test(x_train, x_test, kernel_func)

                    # Compute predictions over the test data
                    pred = self.predict_labels(alpha, np.matrix.transpose(gram_mat_test))

                    # Convert predictions to labels
                    pred = array_to_labels(pred, -1)

                    fold_accuracy += accuracy_score(pred, y_test)
                
                # Compute average accuracy for all folds
                average_accuracy = fold_accuracy / folds
                accuracy_values.append(average_accuracy)

            print('lambda={}'.format(lam))
            print('For the sigma values: {}'.format(sigma_values))
            print('Accuracies: {}\n'.format(accuracy_values))


    def train_folds_spectrum(self, data, labels, folds, distribution):
        """
        docstring
        """

        len_data = len(data)
        data = np.array(list(zip(data, labels)))
        len_fold = int(len(data) / folds)

        lambda_values = [0.001, 0.01, 0.1, 0.9]
        k_values = [8,9,10,11,12,13, 14]

        for lam in lambda_values:
            accuracy_values = []
            
            # Build a partial gaussian function with the current 'sigma' value
            kernel_func = partial(np.dot)

            for k in k_values:
            
                # TODO Compute the whole gram matrix here only once
                # each fold will extract

                fold_accuracy = 0
                for i in range(folds):
                    # print('Fold: {}'.format(i))
                    # Training data is obtained by concatenating the 2 subsets: at the right + at the left
                    # of the current fold
                    train_data = [*data[0:i*len_fold], *data[(i+1)*len_fold:len_data]]

                    # The current fold is used to test the model
                    test_data = [*data[i*len_fold:(i+1)*len_fold]]
                    
                    x_train = np.array([x[0] for x in train_data])
                    y_train = np.array([x[1] for x in train_data])

                    x_test = np.array([x[0] for x in test_data])
                    y_test = np.array([x[1] for x in test_data])

                    y_test = y_test.astype(np.int64)
                    y_train = y_train.astype(np.int64)

                    # Build the Gram matrix for the spectrum kernel
                    histograms_X_train = kernels.spectrum_histogram(x_train, x_train, k, distribution)
                    gram_matrix = kernels.kernel_matrix_training(histograms_X_train, kernel_func)

                    # Solve the linear system in order to find the vector weights
                    alpha = self.solve_linear_system(gram_matrix, len(x_train), lam, y_train)
                    alpha = alpha.reshape(len(x_train),1)

                    # Build the Gram matrix for the test data
                    histograms_X_test = kernels.spectrum_histogram(x_train, x_test, k, distribution)
                    gram_mat_test = kernels.kernel_matrix_test(histograms_X_train, histograms_X_test, kernel_func)

                    # Compute predictions over the test data
                    pred = self.predict_labels(alpha, np.matrix.transpose(gram_mat_test))

                    # Convert predictions to labels
                    pred = array_to_labels(pred, -1)

                    fold_accuracy += accuracy_score(pred, y_test)
                
                # Compute average accuracy for all folds
                average_accuracy = fold_accuracy / folds
                accuracy_values.append(average_accuracy)

            print('lambda={}'.format(lam))
            print('For the k values: {}'.format(k_values))
            print('Accuracies: {}\n'.format(accuracy_values))