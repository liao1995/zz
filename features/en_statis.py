# Authors: LI AO <aoli@hit.edu.cn>
#
# License: 

import numpy as np
from scipy import stats
from sklearn.utils.validation import check_array

def fit_transform(X):
    """Fit the model with X and apply the statistics calculator on X.
       Note : This function return enhanced statistics, which NOT tolerate NaN.


    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data, where n_samples is the number of samples
        and n_features is the number of features. 

    Returns
    -------
    X_new : array-like, shape (n_samples, n_statistics)
    names : 1-D array, shape (n_statistics,)


    """
    X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
    if np.isnan(np.min(X)): raise ValueError("It contains NaN.")
    X_gmean      = stats.gmean(X, axis=1)
    X_kurt       = stats.kurtosis(X, axis=1)
    mode_res     = stats.mode(X, axis=1)
    X_mode_mode  = mode_res.mode
    X_mode_count = mode_res.count
    X_skew       = stats.skew(X, axis=1)
    X_sem        = stats.sem(X, axis=1) 
    X_iqr        = stats.iqr(X, axis=1)
    names = np.array(['gmean', 'kurt', 'mode_mode', 'mode_count', 'skew', \
                      'sem', 'iqr'])
    X_new = np.column_stack((X_gmean,  X_kurt, X_mode_mode, X_mode_count,  \
                             X_skew, X_sem, X_iqr))
    return X_new, names
