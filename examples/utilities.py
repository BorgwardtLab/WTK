'''
Contains various utility functions for data handling, pre-processing,
and so on.
'''


import os

from glob import glob
from pathlib import Path
import shutil

import numpy as np
from os.path import basename, join

from numpy.linalg import eigh

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score

def strip_suffix(s, suffix):
    '''
    Removes a suffix from a string if the string contains it. Else, the
    string will not be modified and no error will be raised.
    '''

    if not s.endswith(suffix):
        return s
    return s[:len(s)-len(suffix)]


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
