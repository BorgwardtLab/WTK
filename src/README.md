# Wasserstein Time Series Kernel

### Installation
The easiest way is to install WTK from the Python Package Index (PyPI) via

```
$ pip install wtk
```

### Usage
The package provides functions to transform a set of `n` training time series and `o` test time series into an `n x n` distance matrix for training and an `o x n` distance matrix for testing.
Additionally, we provide a way to run a grid search for a krein space SVM. `krein_svm_grid_search` runs a `5`-fold
cross-validation on the training set to determine the best hyperparameters. Then, its classification accuracy is
computed on the test set.

```
from wtk import transform_to_dist_matrix
from wtk.utilities import get_ucr_dataset, krein_svm_grid_search

# Read UCR data
X_train, y_train, X_test, y_test = get_ucr_dataset('../data/UCR/raw_data/', 'DistalPhalanxTW')

# Compute wasserstein distance matrices with subsequent length k=10
D_train, D_test = transform_to_dist_matrix(X_train, X_test, 10)

# Run the grid search
svm_clf = krein_svm_grid_search(D_train, D_test, y_train, y_test)
```

Alternatively, you can get the kernel matrices computed from the distance matrices and train your own classifier.
```
from sklearn.svm import SVC
from wtk import get_kernel_matrix
from sklearn.metrics import accuracy_score

# Get the kernel matrices
K_train = get_kernel_matrix(D_train, psd=True, gamma=0.2)
K_test = get_kernel_matrix(D_test, psd=False, gamma=0.2)

# Train your classifier
clf = SVC(C=5, kernel='precomputed')
clf.fit(K_train, y_train)

y_pred = clf.predict(K_test)
print(accuracy_score(y_test, y_pred))
```

# Help

If you have questions concerning WTK or you encounter problems when
trying to build the tool under your own system, please open an issue in
[the issue tracker](https://github.com/BorgwardtLab/WTK/issues). Try to
describe the issue in sufficient detail in order to make it possible for
us to help you.

# Contributors

WTK is developed and maintained by members of the [Machine Learning and
Computational Biology Lab](https://www.bsse.ethz.ch/mlcb) of [Prof. Dr.
Karsten Borgwardt](https://www.bsse.ethz.ch/mlcb/karsten.html):

- Christian Bock ([GitHub](https://github.com/chrisby))
- Matteo Togninalli ([GitHub](https://github.com/mtog))
- Bastian Rieck ([GitHub](https://github.com/Submanifold))
