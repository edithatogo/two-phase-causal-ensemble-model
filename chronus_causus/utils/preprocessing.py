# chronus_causus/utils/preprocessing.py

"""Data Preprocessing Utilities."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(X, signif=0.05):
    """
    Check the stationarity of time series. If not, difference them.

    Parameters
    ----------
    X: dataframe
        Multivariable time series dataset

    Returns
    -------
    X_stationary: dataframe
        Differenced multivariable time series dataset
    """

    # Check the stationarity
    non_stationary = False
    for _, column in X.items():
        r = adfuller(column, autolag='AIC')
        p_value = round(r[1], 4)
        if p_value > signif:
            non_stationary = True

    # if the time series are not stationary, difference them repeatedly
    # until they are stationary
    while (non_stationary):
        X = X.diff().dropna()
        for _, column in X.items():
            r = adfuller(column, autolag='AIC')
            p_value = round(r[1], 4)
            if p_value <= signif:
                non_stationary = False
            else:
                non_stationary = True
                break

    X_stationary = X
    return X_stationary
