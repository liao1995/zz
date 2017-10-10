# Authors: LI AO <aoli@hit.edu.cn>
#
# License: 

import numpy as np
from sklearn.utils.validation import check_array

def fit_transform(X):
    """Fit the model with X and apply the statistics calculator on X.
       Note : This function return basic statistics, which tolerate NaN.


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
    X_mean   = np.nanmean(X, axis=1)
    X_std    = np.nanstd(X, axis=1)
    X_var    = np.nanvar(X, axis=1)
    X_median = np.nanmedian(X, axis=1) 
    X_min    = np.nanmin(X, axis=1)
    X_max    = np.nanmax(X, axis=1)
    X_range  = X_max - X_min
    X_nnz    = np.sum(X!=0, axis=1)
    X_quad   = np.nanpercentile(X, 25, axis=1)
    X_half   = np.nanpercentile(X, 50, axis=1)
    names = np.array(['mean', 'std', 'var', 'median', 'min', 'max', 'range', 'nnz',\
                      'quad', 'half'])
    X_new = np.column_stack((X_mean, X_std, X_var, X_median, X_min, X_max, X_range, \
                             X_nnz, X_quad, X_half))
    return X_new, names
