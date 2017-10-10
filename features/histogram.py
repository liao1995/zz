# Authors: LI AO <aoli@hit.edu.cn>
#
# License: 

import numpy as np
from sklearn.utils.validation import check_array

def fit_transform(X):
    """Fit the model with X and apply the histogram calculator on X.
       Note : This function return histogram of X, which NOT tolerate NaN.


    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data, where n_samples is the number of samples
        and n_features is the number of features. 

    Returns
    -------
    X_new : array-like, shape (n_samples, n_bins)


    """
    X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
    if np.isnan(np.min(X)): raise ValueError("It contains NaN.")
    X_new = list()
    for sample in X:
        counts, bins = np.histogram(sample)
        X_new.append(counts)
    return np.array(X_new)
