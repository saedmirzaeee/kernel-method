import numpy as np
from utils import *
import itertools as it


"""
    This file provides different functions for creating kernel matrixes through out the project
"""


"""
    Gaussian Kernel value for x1,x2 point and specific sigma value
"""
def gaussian_kernel(sigma, x1, x2):
    return np.exp(-0.5 * (np.linalg.norm(x1 - x2) ** 2) / sigma**2) #/ (sigma * np.sqrt(2*np.pi))


"""
    X1 - second list of sequences(train)
    X2 - first list of sequences(train/test)
    k - length of the substring
"""
def spectrum_histogram(X1, X2, k, distribution):

    # Build the kmers dictionary for the training sequences
    kmer_dict = build_kmers_dict(X1, k, distribution)

    conv = Converter(k)

    # List which stores the kmers frequencies for each sequence
    histogram_X = []

    for seq in X2:
        # Set all values in the dictionary to 0
        kmer_dict = dict.fromkeys(kmer_dict, 0)

        # For each kmer in the current seq, increment its occurence nb in the frequency dictionary
        for kmer in conv.all_kmers_as_ints(seq):

            # Check especially 
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1

        # Get a snapshot of the kmer_dic and insert as a list of kmer frequencies in the histogram
        histogram_X.append(list(kmer_dict.values()))

    save_object(histogram_X, 'spectrum_histogram_distrib={}_k={}'.format(distribution, k))
    return np.array(histogram_X)


def kernel_matrix_training(X, kernel):
    """ 
        Compute the kernel matrix for the training data.
    """
    X_count = len(X)

    K = np.zeros((X_count, X_count))
    for i in range(X_count):
        K[i,i] = kernel(X[i], X[i])

    for i, j in it.combinations(range(X_count), 2):
        K[i,j] = K[j,i] = kernel(X[i], X[j])
    return K


def kernel_matrix_test(X1, X2, kernel):
    X1_count = X1.shape[0]
    X2_count = X2.shape[0]

    K = np.zeros((X1_count, X2_count))
    for i in range(X1_count):
        for j in range(X2_count):
            K[i,j] = kernel(X1[i], X2[j])
    return K


