'''
Contains various utility functions for data handling, pre-processing,
and so on.
'''


import os
import numpy as np
import logging

from glob import glob
from pathlib import Path
import shutil

import numpy as np
from os.path import basename, join

from numpy.linalg import eigh

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.metrics import make_scorer, accuracy_score


def strip_suffix(s, suffix):
    '''
    Removes a suffix from a string if the string contains it. Else, the
    string will not be modified and no error will be raised.
    '''

    if not s.endswith(suffix):
        return s
    return s[:len(s)-len(suffix)]

def get_ucr_dataset(data_dir: str, dataset_name: str):
    '''
    Loads train and test data from a folder in which
    the UCR data sets are stored.
    '''

    X_train, y_train, _ = read_ucr_data(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN'))
    X_test, y_test, _ = read_ucr_data(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST'))

    return X_train, y_train, X_test, y_test

def read_ucr_data(filename):
    '''
    Loads an UCR data set from a file, returning the samples and the
    respective labels. Also extracts the data set name such that one
    may easily display results.
    '''

    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]

    # Remove all potential suffixes to obtain the data set name. This is
    # somewhat inefficient, but we only have to do it once.
    name = os.path.basename(filename)
    name = strip_suffix(name, '_TRAIN')
    name = strip_suffix(name, '_TEST')

    return X, Y, name

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    # Custom model for an array of precomputed kernels
    # 1. Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc)
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]

from numpy.linalg import eigh

def ensure_psd(K, tol=1e-8):
    # Helper function to remove negative eigenvalues
    w,v = eigh(K)
    if (w<-tol).sum() >= 1:
        neg = np.argwhere(w<-tol)
        w[neg] = 0
        Xp = v.dot(np.diag(w)).dot(v.T)
        return Xp
    else:
        return K

def krein_svm_grid_search(K_train: np.ndarray, K_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray, 
        param_grid: dict={'C': np.logspace(-3, 5, num=9)},
        gammas: np.ndarray=np.logspace(-4,1,num=6)):

    logger = logging.getLogger()
    logger.info('Starting analysis')

    kernel_matrices_train = []
    kernel_matrices_test = []
    kernel_params = []
    for g in gammas:
        M_train = np.exp(-g*K_train) 
        M_test = np.exp(-g*K_test)
        # Add psd-ensuring conditions
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

    return gs

class KreinSVC(BaseEstimator, ClassifierMixin):
    '''
    An SVC which ensures positive definiteness before training.
    It also computes the kernel matrix from the given distance matrix:
        K = np.exp(-self.psd_gamma*D_matrix) 
    '''
    def __init__(self, C=1.0, kernel='precomputed', degree=3, gamma='auto_deprecated',
             coef0=0.0, shrinking=True, probability=False,
             tol=1e-3, cache_size=200, class_weight=None,
             verbose=False, max_iter=-1, decision_function_shape='ovr',
             random_state=None, psd_tol=1e-8, psd_gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.shrinking = shrinking
        self.probability = probability
        self.verbose = verbose
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.psd_tol = psd_tol
        self.psd_gamma = psd_gamma

        self.svc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
             coef0=coef0, shrinking=shrinking, probability=probability,
             tol=tol, cache_size=cache_size, class_weight=class_weight,
             verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
             random_state=random_state)
    
    def get_params():
        return {
            'C': self.C,
            'kernel': self.kernel,
            'shrinking': self.shrinking,
            'probability': self.probability,
            'verbose': self.verbose,
            'cache_size': self.cache_size,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter,
            'decision_function_shape': self.decision_function_shape,
            'random_state': self.random_state,
            'psd_tol': self.psd_tol,
            'psd_gamma': self.psd_gamma
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, sample_weight=None):
        '''
        Args:
            X (np.ndarray): Distance_matrix
        '''
        M_train = np.exp(-self.psd_gamma*X) 
        K = ensure_psd(M_train, tol=self.psd_tol)
        self.svc.fit(K, y, sample_weight)

    def predict(self, X):
        M_test = np.exp(-self.psd_gamma*X) 
        return self.svc.predict(M_test)

    def predict_proba(self, X):
        return self.svc.predict_proba(X)

