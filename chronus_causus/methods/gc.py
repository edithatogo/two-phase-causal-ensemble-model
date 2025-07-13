# chronus_causus/methods/gc.py

"""Granger Causality (GC) Causal Discovery Method."""

import numpy as np
from sklearn.base import BaseEstimator

class GCDiscoverer(BaseEstimator):
    """
    Causal discovery using Granger Causality (GC).

    Granger causality is a statistical concept of causality that is based on
    prediction. According to Granger causality, if a signal X1 "Granger-causes"
    (or "G-causes") a signal X2, then past values of X1 should contain
    information that helps predict X2 above and beyond the information
    contained in past values of X2 alone.

    Parameters
    ----------
    max_lag : int, default=1
        The maximum lag to consider.

    Attributes
    ----------
    causal_matrix_ : np.ndarray
        The computed causal matrix after fitting.
        The shape is (n_features, n_features).
        Entry (i, j) > 0 might indicate a causal link from feature j to feature i.
    feature_names_in_ : list[str]
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    """
    def __init__(self, max_lag: int = 1, test: str = 'ssr_chi2test', signif: float = 0.05, threshold: float = 0.3, **kwargs):
        self.max_lag = max_lag
        self.test = test
        self.signif = signif
        self.threshold = threshold
        self.kwargs = kwargs
        self.causal_matrix_ = None

    def fit(self, X, y=None):
        """
        Compute the causal matrix for a single data partition using GC.

        Args:
            X (np.ndarray): A 2D array representing the time series data
                              for a single partition. Shape (n_samples, n_features).
            y : None, ignored.

        Returns
        -------
        self : GCDiscoverer
            The fitted estimator.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(n_features)]

        from statsmodels.tsa.stattools import grangercausalitytests
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.api import VAR
        import scipy.stats
        import pandas as pd

        df = pd.DataFrame(X, columns=self.feature_names_in_)

        # Stationarity check
        for name, column in df.items():
            r = adfuller(column, autolag='AIC')
            if r[1] > self.signif:
                df = df.diff().dropna()
                break

        df_gc = pd.DataFrame(np.zeros((n_features, n_features)),
                             columns=self.feature_names_in_, index=self.feature_names_in_)
        df_corr = pd.DataFrame(np.zeros((n_features, n_features)),
                               columns=self.feature_names_in_, index=self.feature_names_in_)

        for c in df_gc.columns:
            for r in df_gc.index:
                if c == r:
                    continue
                gc_result = grangercausalitytests(df[[r, c]], maxlag=self.max_lag, verbose=False)
                p_values = [round(gc_result[i + 1][0][self.test][1], 4) for i in range(self.max_lag)]
                min_p_value = np.min(p_values)
                df_gc.loc[r, c] = min_p_value

        for c in df_corr.columns:
            for r in df_corr.index:
                if c == r:
                    continue
                if df_gc.loc[r, c] < self.signif:
                    if len(df[r]) > self.max_lag:
                        # VAR model
                        model = VAR(df[[r,c]])
                        res = model.fit(maxlags=self.max_lag)
                        pred = res.forecast(y=df[[r,c]].values[-self.max_lag:], steps=self.max_lag)
                        corr, _ = scipy.stats.pearsonr(df[r][-self.max_lag:], pred[:,0])
                        df_corr.loc[r, c] = round(abs(corr), 4) if abs(corr) > self.threshold else 0.

        for c in df_corr.columns:
            for r in df_corr.index:
                if c > r:
                    if df_corr.loc[r, c] > df_corr.loc[c, r] > 0:
                        df_corr.loc[c, r] = 0.
                    if df_corr.loc[c, r] > df_corr.loc[r, c] > 0:
                        df_corr.loc[r, c] = 0.

        self.causal_matrix_ = df_corr.values
        return self

    def _validate_input(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array (n_samples, n_features).")
        if X.shape[0] < 10:
             raise ValueError(f"Input X must have at least 10 samples, got {X.shape[0]}.")
        if X.shape[1] < 2:
            raise ValueError(f"Input X must have at least 2 features, got {X.shape[1]}.")
        return X
