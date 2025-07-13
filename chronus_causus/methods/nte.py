# chronus_causus/methods/nte.py

"""Non-linear Transfer Entropy (NTE) Causal Discovery Method."""

import numpy as np
from sklearn.base import BaseEstimator

class NTEDiscoverer(BaseEstimator):
    """
    Causal discovery using Non-linear Transfer Entropy (NTE).

    NTE measures the transfer of information from one time series to another.
    If the information transfer from X to Y is greater than from Y to X,
    then X is considered to cause Y.

    Parameters
    ----------
    lag : int, default=1
        The lag to use for the transfer entropy calculation.
    embed_dim : int, default=3
        The embedding dimension for state space reconstruction.

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
    def __init__(self, lag: int = 1, embed_dim: int = 3, k: int = 1, **kwargs):
        self.lag = lag
        self.embed_dim = embed_dim
        self.k = k
        self.kwargs = kwargs
        self.causal_matrix_ = None

    def _nte_param_compute(self, X, Y, k=1, embedding=1, safetyCheck=False, GPU=False):
        """
        Calculates the transfer entropy.
        This is a re-implementation of the nte_param_compute function from
        demostration/utilities_nte.py, using the CPU_TE.py file as a reference.
        """
        from sklearn.neighbors import KDTree
        from . import CPU_TE

        # This is a simplified version of the make_spaces function from PyIF
        def make_spaces(X, Y, embedding):
            x_outer = X[:-embedding]
            y_outer = Y[:-embedding]
            x_inner = np.array([X[i:i+embedding] for i in range(len(X)-embedding)])
            y_inner = np.array([Y[i:i+embedding] for i in range(len(Y)-embedding)])
            xkyPts = np.hstack([x_outer[:,np.newaxis], y_outer[:,np.newaxis], x_inner])
            kyPts = y_outer[:,np.newaxis]
            xkPts = np.hstack([x_outer[:,np.newaxis], x_inner])
            kPts = y_outer[:,np.newaxis]
            nPts = len(xkyPts)
            return xkyPts, kyPts, xkPts, kPts, nPts

        xkyPts, kyPts, xkPts, kPts, nPts = make_spaces(X, Y, embedding=embedding)

        xkykdTree = KDTree(xkyPts, metric="chebyshev")
        kykdTree = KDTree(kyPts, metric="chebyshev")
        xkkdTree = KDTree(xkPts, metric="chebyshev")
        kkdTree = KDTree(kPts, metric="chebyshev")

        HY, HYX, TE = CPU_TE.compute(xkykdTree, kykdTree, xkkdTree, kkdTree,
                                     xkyPts, kyPts, xkPts, kPts, nPts, X, embedding=embedding, k=k)
        return HY, HYX, TE

    def fit(self, X, y=None):
        """
        Compute the causal matrix for a single data partition using NTE.

        Args:
            X (np.ndarray): A 2D array representing the time series data
                              for a single partition. Shape (n_samples, n_features).
            y : None, ignored.

        Returns
        -------
        self : NTEDiscoverer
            The fitted estimator.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(n_features)]

        self.causal_matrix_ = np.zeros((n_features, n_features))

        from itertools import combinations
        from random import shuffle
        from statsmodels.stats.weightstats import ztest as ztest

        feature_combinations = list(combinations(np.arange(0, n_features, 1), 2))

        for i in feature_combinations:
            x_1 = X[:, i[0]]
            xs_1 = x_1.copy()
            shuffle(xs_1)
            x_2 = X[:, i[1]]
            xs_2 = x_2.copy()
            shuffle(xs_2)

            HY_1, _, TE_1 = self._nte_param_compute(x_1, x_2, k=self.k, embedding=self.lag)
            HY_2, _, TE_2 = self._nte_param_compute(x_2, x_1, k=self.k, embedding=self.lag)
            _, _, TES_1 = self._nte_param_compute(xs_1, x_2, k=self.k, embedding=self.lag)
            _, _, TES_2 = self._nte_param_compute(xs_2, x_1, k=self.k, embedding=self.lag)

            NTE_1 = max(((TE_1 - TES_1) / HY_1), 0)
            NTE_2 = max(((TE_2 - TES_2) / HY_2), 0)

            # z-test
            test_1 = []
            test_2 = []
            s = [0]
            for j in range(10):
                s.append(int(n_samples/10*(j+1)))

            for j in range(9):
                xtest_1 = x_1[s[j]:s[j+2]]
                xtest_2 = x_2[s[j]:s[j+2]]
                xs_test_1 = xtest_1.copy()
                shuffle(xs_test_1)
                xs_test_2 = xtest_2.copy()
                shuffle(xs_test_2)
                HY_test_1, _, TE_test_1 = self._nte_param_compute(xtest_1, xtest_2, k=self.k, embedding=self.lag)
                HY_test_2, _, TE_test_2 = self._nte_param_compute(xtest_2, xtest_1, k=self.k, embedding=self.lag)
                _, _, TES_test_1 = self._nte_param_compute(xs_test_1, xtest_2, k=self.k, embedding=self.lag)
                _, _, TES_test_2 = self._nte_param_compute(xs_test_2, xtest_1, k=self.k, embedding=self.lag)
                NTE_test_1 = max(((TE_test_1 - TES_test_1) / HY_test_1), 0)
                NTE_test_2 = max(((TE_test_2 - TES_test_2) / HY_test_2), 0)
                test_1.append(NTE_test_1)
                test_2.append(NTE_test_2)

            p_z = ztest(test_1, test_2, value=0)
            ToF_z = True if p_z[1] < 0.05 else False

            if ToF_z and NTE_1 > NTE_2 and NTE_1 > 0:
                self.causal_matrix_[i[1], i[0]] = round(NTE_1, 4)
            if ToF_z and NTE_2 > NTE_1 and NTE_2 > 0:
                self.causal_matrix_[i[0], i[1]] = round(NTE_2, 4)

        self.causal_matrix_ = te_mapping(self.causal_matrix_)

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

def te_mapping(X):
    """
    Curve fitting of NTE values so that they can be compared with
    the coefficents of other submodels.

    Parameters
    ----------
    X: 2d array
        The original causality strength matrix of TE

    Returns
    -------
    output: 2d array
        The mapped causality strength matrix of TE
    """

    a = 0.19604753
    b = 6.01585935
    c = 0.00793923
    d = -0.29553602
    e = 0.94236698
    output = a * np.log(b*X + c) + d*X + e
    # make sure the range is [0,1]
    output[output<=0.] = 0.
    output[output>=1.] = 1.

    return output
