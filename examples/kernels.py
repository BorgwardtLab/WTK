'''
Contains various kernels for time series analysis.
'''


import numpy as np

import ot
# import tqdm

from sklearn.metrics import pairwise
from sklearn.preprocessing import scale

def subsequences(time_series, k):
    time_series = np.asarray(time_series)
    n = time_series.size
    shape = (n - k + 1, k)
    strides = time_series.strides * 2

    return np.lib.stride_tricks.as_strided(
        time_series,
        shape=shape,
        strides=strides
    )

def pairwise_subsequence_kernel(
    time_series_train,
    time_series_test,
    k,
    functor, 
    par_grid = [1],
    normalized = False):
    '''
    Applies a calculation functor to all pairs of a data set. As
    a result, two matrices will be calculated:

    1. The square matrix between all pairs of training samples
    2. The rectangular matrix between test samples (rows), and
       training samples (columns).

    These matrices can be fed directly to a classifier.

    Notice that this function will apply the kernel to *each* of
    the subsequences in both time series.
    '''

    n = len(time_series_train)
    m = len(time_series_test)



    K_train = np.zeros((n, n))  # Need to initialize with zeros for symmetry
    K_test = np.empty((m, n))   # Since this is rectangular, no need for zeros

    if functor in (custom_rbf_kernel, brownian_bridge_kernel):

        K_par_train = []
        K_par_test = []

        for i in range(len(par_grid)):
            K_par_train.append(np.zeros((n, n)))
            K_par_test.append(np.empty((m, n)))
    # Create subsequences of the time series. These cannot be easily
    # shared with other calls of the method.

    subsequences_train = dict()
    subsequences_test = dict()

    for i, ts_i in enumerate(time_series_train):
        subsequences_train[i] = subsequences(ts_i, k)

    for i, ts_i in enumerate(time_series_test):
        subsequences_test[i] = subsequences(ts_i, k)


    # Evaluate the functor for *all* relevant pairs, while filling up
    # the initially empty kernel matrices.

    desc = 'Pairwise kernel computations'
    # for i, ts_i in enumerate(tqdm.tqdm(time_series_train, desc=desc)):
    for i, ts_i in enumerate(time_series_train):
        for j, ts_j in enumerate(time_series_train[i:]):
            s_i = subsequences_train[i]     # first shapelet
            s_j = subsequences_train[i + j] # second shapelet
            if normalized:
                s_i = scale(s_i, axis=1)
                s_j = scale(s_j, axis=1)

            if functor in (custom_rbf_kernel, brownian_bridge_kernel):
                # euclidean distance
                if functor == custom_rbf_kernel:
                    e_dist = pairwise.euclidean_distances(s_i, s_j)

                for idx, par in enumerate(par_grid):
                    if functor == custom_rbf_kernel:
                        K_par_train[idx][i, i + j] = functor(s_i, s_j, e_dist, par)
                    else:
                        K_par_train[idx][i, i + j] = functor(s_i, s_j, par)
            else:
                K_train[i, i + j] = functor(s_i, s_j)

        for j, ts_j in enumerate(time_series_test):
            s_i = subsequences_train[i] # first shapelet

            # Second shapelet; notice that there is no index shift in
            # comparison to the code above.
            s_j = subsequences_test[j]
            if normalized:
                s_i = scale(s_i, axis=1)
                s_j = scale(s_j, axis=1)

            if functor in (custom_rbf_kernel, brownian_bridge_kernel):
                if functor == custom_rbf_kernel:
                    # euclidean distance
                    e_dist = pairwise.euclidean_distances(s_i, s_j)

                for idx, par in enumerate(par_grid):

                    if functor == custom_rbf_kernel:
                        K_par_test[idx][j, i] = functor(s_i, s_j, e_dist, par)
                    else:
                        K_par_test[idx][j, i] = functor(s_i, s_j, par)



            else:
                # Fill the test matrix; since it has different dimensions
                # than the training matrix, the indices are swapped here.
                K_test[j, i] = functor(s_i, s_j)


    # Makes the matrix symmetric since we only fill the upper diagonal
    # in the code above.


    # Makes the matrix symmetric since we only fill the upper diagonal
    # in the code above.
    if functor in (custom_rbf_kernel, brownian_bridge_kernel):
        for k_idx in range(len(par_grid)):
            K_train_cur = K_par_train[k_idx]
            K_par_train[k_idx] = K_train_cur + K_train_cur.T

        K_train = K_par_train
        K_test = K_par_test
    else:
        K_train = K_train + K_train.T


    return K_train, K_test


def wasserstein_kernel(subsequences_1, subsequences_2, metric='euclidean'):
    '''
    Calculates the distance between two time series using their
    corresponding set of subsequences. The metric used to align
    them may be optionally changed.
    '''

    C = ot.dist(subsequences_1, subsequences_2, metric=metric)
    return ot.emd2([], [], C)

def binarized_wasserstein_kernel(subsequences_1, subsequences_2, metric='euclidean'):
    '''
    Calculates the distance between two time series using their
    corresponding set of subsequences. The metric used to align relies
    on a thresholded version of the euclidean (or other) distance
    '''

    C = ot.dist(subsequences_1, subsequences_2, metric=metric)
    C = (C>np.percentile(C,5)).astype(int)
    return ot.emd2([], [], C)

def linear_kernel(subsequences_1, subsequences_2):
    '''
    Calculates the linear kernel between two time series using their
    corresponding set of subsequences.
    '''

    K_lin = pairwise.linear_kernel(subsequences_1, subsequences_2)
    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    return np.sum(K_lin)/(n*m)

def polynomial_kernel(subsequences_1, subsequences_2, p=2, c=1.0):
    '''
    Calculates the linear kernel between two time series using their
    corresponding set of subsequences.
    '''

    K_poly = pairwise.polynomial_kernel(subsequences_1, subsequences_2,
        degree=p, coef0=c, gamma=1.0)
    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    return np.sum(K_poly)/(n*m)

def rbf_kernel(subsequences_1, subsequences_2):
    '''
    Calculates the rbf kernel between two time series using their
    corresponding set of subsequences.
    '''

    K_rbf = pairwise.rbf_kernel(subsequences_1, subsequences_2)
    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    return np.sum(K_rbf)/(n*m)


def custom_rbf_kernel(subsequences_1, subsequences_2, e_dist, gamma):

    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    K_rbf = np.exp(-gamma*(e_dist**2))

    return np.sum(K_rbf)/(n*m)

def brownian_bridge_kernel(subsequences_1, subsequences_2, c):

    n = subsequences_1.shape[0]
    m = subsequences_2.shape[0]

    K_brown = c - np.abs(subsequences_1-subsequences_2)
    K_brown[K_brown<0] = 0


    return np.sum(K_brown)/(n*m)

