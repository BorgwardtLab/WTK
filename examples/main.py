#!/usr/bin/env python3

import argparse
import logging

import numpy as np

from kernels import pairwise_subsequence_kernel
from kernels import wasserstein_kernel
from utilities import read_ucr_data, custom_grid_search_cv, ensure_psd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

ENSURE_PSD = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('TRAIN', type=str)
    parser.add_argument('TEST', type=str)
    parser.add_argument('--k', '-k',
        type=int,
        default=10,
        help='Subsequence (shapelet) length'
    )

    args = parser.parse_args()

    X_train, y_train, name = read_ucr_data(args.TRAIN)
    X_test, y_test, _ = read_ucr_data(args.TEST)

    logging.basicConfig()

    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    logger.info('Starting analysis')

    K_train, K_test = pairwise_subsequence_kernel(
        X_train,
        X_test,
        args.k,
        wasserstein_kernel
    )
    logger.info('Wasserstein distance computed, starting SVM grid search...')

    param_grid = {
        'C': np.logspace(-3, 5, num=9),
    }
    gammas = np.logspace(-4,1,num=6)

    kernel_matrices_train = []
    kernel_matrices_test = []
    kernel_params = []
    for g in gammas:
        M_train = np.exp(-g*K_train) 
        M_test = np.exp(-g*K_test)
        # Add psd-ensuring conditions
        if ENSURE_PSD:
            M_train = ensure_psd(M_train)

        kernel_matrices_train.append(M_train)
        kernel_matrices_test.append(M_test)
        kernel_params.append(g)

    svm = SVC(kernel='precomputed')

    # Gridsearch
    gs, best_params = custom_grid_search_cv(svm, param_grid, kernel_matrices_train, y_train, cv=5)
    # Store best params
    gamma = kernel_params[best_params['K_idx']]
    C = best_params['params']['C']
    print(f"Best C: {C}")
    print(f"Best gamma: {gamma}")

    y_pred = gs.predict(kernel_matrices_test[best_params['K_idx']])
    accuracy = accuracy_score(y_test, y_pred)

    logger.info('Accuracy = {:2.2f}'.format(accuracy * 100))
