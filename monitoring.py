#!/usr/bin/env python
"""
example performance monitoring script
"""

import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance
from model import get_preprocessor

def get_latest_train_data():
    """
    load the data used in the latest training
    """

    data_file = os.path.join("models",'latest-train.pickle')

    if not os.path.exists(data_file):
        raise Exception("cannot find {}-- did you train the model?".format(data_file))

    with open(data_file,'rb') as tmp:
        data = pickle.load(tmp)

    return(data)

    
def get_monitoring_tools(X,y):
    """
    determine outlier and distance thresholds
    return thresholds, outlier model(s) and source distributions for distances
    NOTE: for classification the outlier detection on y is not needed

    """

    preprocessor = get_preprocessor()
    preprocessor = preprocessor.fit(X)
    X_pp = preprocessor.transform(X)
    
    xpipe = Pipeline(steps=[('pca', PCA(2)),
                            ('clf', EllipticEnvelope(random_state=0,contamination=0.01))])
    xpipe.fit(X_pp)
    
    bs_samples = 1000
    outliers_X = np.zeros(bs_samples)
    wasserstein_X = np.zeros(bs_samples)
    wasserstein_y = np.zeros(bs_samples)
    
    for b in range(bs_samples):
        n_samples = int(np.round(0.80 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,replace=True).astype(int)
        y_bs=y[subset_indices]
        X_bs=X_pp[subset_indices,:]
    
        test1 = xpipe.predict(X_bs)
        wasserstein_X[b] = wasserstein_distance(X_pp.flatten(),X_bs.flatten())
        wasserstein_y[b] = wasserstein_distance(y,y_bs.flatten())
        outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))

    ## determine thresholds as a function of the confidence intervals
    outliers_X.sort()
    outlier_X_threshold = outliers_X[int(0.975*bs_samples)] + outliers_X[int(0.025*bs_samples)]

    wasserstein_X.sort()
    wasserstein_X_threshold = wasserstein_X[int(0.975*bs_samples)] + wasserstein_X[int(0.025*bs_samples)]

    wasserstein_y.sort()
    wasserstein_y_threshold = wasserstein_y[int(0.975*bs_samples)] + wasserstein_y[int(0.025*bs_samples)]
    
    to_return = {"outlier_X": np.round(outlier_X_threshold,1),
                 "wasserstein_X":np.round(wasserstein_X_threshold,2),
                 "wasserstein_y":np.round(wasserstein_y_threshold,2),
                 "preprocessor":preprocessor,
                 "clf_X":xpipe,
                 "X_source":X_pp,
                 "y_source":y,
                 "latest_X":X,
                 "latest_y":y}
    return(to_return)


if __name__ == "__main__":

    ## get latest training data
    data = get_latest_train_data()
    y = data['y']
    X = data['X']

    ## get performance monitoring tools
    pm_tools = get_monitoring_tools(X,y)
    print("outlier_X",pm_tools['outlier_X'])
    print("wasserstein_X",pm_tools['wasserstein_X'])
    print("wasserstein_y",pm_tools['wasserstein_y'])
    
    print("done")
    
