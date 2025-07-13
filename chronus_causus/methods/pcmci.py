# chronus_causus/methods/pcmci.py

"""PCMCI Causal Discovery Method."""

import numpy as np
from sklearn.base import BaseEstimator

class PCMCIDiscoverer(BaseEstimator):
    """
    Causal discovery using the PCMCI algorithm.

    PCMCI (Peter and Clark Momentary Conditional Independence) is a causal
    discovery algorithm for time series data. It is based on a two-step
    procedure: first, a PC-based algorithm is used to identify the parents
    of each variable, and then a momentary conditional independence test
    is used to identify the causal links.

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
    def __init__(self, max_lag: int = 1, pc_alpha: float = 0.05, cond_ind_test: str = 'ParCorr', **kwargs):
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        self.cond_ind_test = cond_ind_test
        self.kwargs = kwargs
        self.causal_matrix_ = None

    def fit(self, X, y=None):
        """
        Compute the causal matrix for a single data partition using PCMCI.

        Args:
            X (np.ndarray): A 2D array representing the time series data
                              for a single partition. Shape (n_samples, n_features).
            y : None, ignored.

        Returns
        -------
        self : PCMCIDiscoverer
            The fitted estimator.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(n_features)]

        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.independence_tests.gpdc import GPDC
        from tigramite.independence_tests.cmiknn import CMIknn
        from tigramite.independence_tests.cmisymb import CMIsymb

        if self.cond_ind_test == 'ParCorr':
            cit = ParCorr()
        elif self.cond_ind_test == 'GPDC':
            cit = GPDC()
        elif self.cond_ind_test == 'CMIknn':
            cit = CMIknn()
        elif self.cond_ind_test == 'CMIsymb':
            cit = CMIsymb()
        else:
            raise ValueError("cond_ind_test should be ParCorr, GPDC, CMIknn or CMIsymb")

        dataframe = pp.DataFrame(X, var_names=self.feature_names_in_)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=self.max_lag, pc_alpha=self.pc_alpha)

        link_matrix = results['graph']
        val_matrix = results['val_matrix']
        n, m, tau = val_matrix.shape

        val = np.zeros([n, m])
        link = np.array([''] * (n * m), dtype='<U3').reshape((n, m))
        for u in range(n):
            for v in range(m):
                argmax = np.abs(val_matrix[u, v][1:]).argmax() + 1
                val[u, v] = np.abs(np.around(val_matrix[u, v][argmax], 4))
                link[u, v] = link_matrix[u, v, argmax]

        self.causal_matrix_ = np.zeros((n, m))
        link_c = link_matrix[:, :, 0]

        for u in range(n):
            for v in range(m):
                if u != v and link[u, v] == '-->':
                    self.causal_matrix_[u, v] = np.abs(val[u, v])
                if link_c[u, v] == '-->' and link_c[v, u] == '<--':
                    self.causal_matrix_[u, v] = np.abs(round(val_matrix[u, v, 0], 4))

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
