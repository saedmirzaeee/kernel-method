from utils import *
import numpy as np
import sys
from kernel_ridge_regression_model import *
import kernels
from kernel_svm_model import *


def run_svm_kernel():
    x_train = np.array(read_x_data(train=True, raw=True))
    y_train = np.array(read_y_data())
    x_test = np.array(read_x_data(train=False, raw=True))

    # Build a partial spectrum function
    kernel_func = partial(np.dot)

    all_labels = []

    for distribution in range(3):
        # Build the Gram matrix for the spectrum kernel
        histograms_X_train = kernels.spectrum_histogram(x_train[distribution], x_train[distribution], 8, 0)
        gram_matrix_train = kernels.kernel_matrix_training(histograms_X_train, kernel_func)


        # Build the Gram matrix for the test data
        histograms_X_test = kernels.spectrum_histogram(x_train[distribution], x_test[distribution], 8, 0)
        gram_mat_test = kernels.kernel_matrix_test(histograms_X_train, histograms_X_test, kernel_func)


        model = Kernel_svm_model(gram_matrix_train)

        model.fit(x_train[distribution], y_train[distribution])

        predicted_values = list(model.predict(gram_mat_test))
    
        all_labels += list(predicted_values)

    all_labels = [0 if x == -1 else 1 for x in all_labels]

    write_predicted_labels_csv(all_labels, 'Yte.csv')


def run_kernel_ridge_regression():
    model = Kernel_ridge_regression_model()
    # Variable to indicate which dataset distribution to use
    distribution = -1

    # Each of the following lists contains 3 elements:
    # An element is a list which incorporate data for a single distribution
    # E.g. x_train = [train0, train1, train2]
    x_train = np.array(read_x_data(train=True, raw=True))
    y_train = np.array(read_y_data())
    x_test = np.array(read_x_data(train=False, raw=True))

    # Run the KRR using the spectrum kernel
    krr_predicted_values = []
    for distribution in range(3):
        predictedValue = model.run_KRR_spectrum(x_train[distribution], y_train[distribution], x_test[distribution], distribution)
        krr_predicted_values =  krr_predicted_values + predictedValue
        
    write_predicted_labels_csv(krr_predicted_values, 'Yte.csv')
    


if __name__ == '__main__':

    #create a ridge regression Model, load the data and write the predicted data to the Yte.csv file
    run_kernel_ridge_regression()
    
    #create a SVM model and train the model using QP Solver and Spectrum Kernel
#    run_svm_kernel()
    
    
    
    



