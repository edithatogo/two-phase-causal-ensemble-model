# chronus_causus/ensemble/causal_ensemble.py

"""Causal Ensemble."""

import numpy as np
from .l1_ensemble import L1Ensembler

class CausalEnsemble:
    """
    Causal Ensemble.

    The Causal Ensemble combines the results of multiple causal discovery
    algorithms using a two-layer ensembling approach.

    Parameters
    ----------
    estimators : list of (str, object)
        A list of tuples, where each tuple contains the name of the
        causal discovery algorithm and the estimator object.
    num_partitions : int, default=10
        The number of data partitions to create.
    l2_threshold : float, default=1.9
        The trustness boundry of the evaluation of l1_ensemble.
    boundry : float, default=0.3
        The trustness boundry of the causality strength of l2_ensemble.

    Attributes
    ----------
    causal_matrix_ : np.ndarray
        The computed causal matrix after fitting.
    """
    def __init__(self, estimators, num_partitions: int = 10, l2_threshold: float = 1.9, boundry: float = 0.3):
        self.estimators = estimators
        self.num_partitions = num_partitions
        self.l2_threshold = l2_threshold
        self.boundry = boundry
        self.causal_matrix_ = None

    def fit(self, X, y=None):
        """
        Fit the causal ensemble.

        Args:
            X (np.ndarray): A 2D array representing the time series data.
                              Shape (n_samples, n_features).
            y : None, ignored.

        Returns
        -------
        self : CausalEnsemble
            The fitted estimator.
        """
        l1_results = []
        l1_accuracy_indices = []

        for name, estimator in self.estimators:
            l1_ensembler = L1Ensembler(estimator, self.num_partitions)
            l1_ensembler.fit(X)
            l1_results.append(l1_ensembler.causal_matrix_)
            l1_accuracy_indices.append(l1_ensembler.accuracy_index_)

        self.causal_matrix_ = self.l2_ensemble(l1_results, l1_accuracy_indices)

        return self

    def l2_ensemble(self, l1_results, l1_accuracy_indices):

        X_1, X_2, X_3, X_4 = l1_results
        acc_1, acc_2, acc_3, acc_4 = l1_accuracy_indices

        X_1[acc_1 < 0.9] = 0.
        X_2[acc_2 < 0.9] = 0.
        X_3[acc_3 < 0.9] = 0.
        X_4[acc_4 < 0.9] = 0.

        n, m = X_1.shape
        strength_ensemble = np.zeros((n, m))

        for u in range(n):
            for v in range(m):
                ensemble_temp = np.hstack((X_1[u, v], X_2[u, v], X_3[u, v], X_4[u, v]))
                weight = np.hstack((acc_1[u, v], acc_2[u, v], acc_3[u, v], acc_4[u, v]))

                weight_temp = weight.copy()
                weight_temp[ensemble_temp <= 0.3] = 0.
                if weight_temp.sum() > 0:
                    normalized_weight = weight_temp / weight_temp.sum()
                else:
                    normalized_weight = np.zeros_like(weight_temp)

                target = (ensemble_temp * normalized_weight).sum()

                if np.count_nonzero(ensemble_temp) > 2:
                    strength_ensemble[u, v] = target
                elif np.count_nonzero(ensemble_temp) == 2:
                    temp = weight.copy()
                    temp.sort()
                    if temp[-1] > 9.9:
                        strength_ensemble[u, v] = target
                    elif temp[-2] > self.l2_threshold:
                        strength_ensemble[u, v] = target
                    else:
                        strength_ensemble[u, v] = 0.
                else:
                    strength_ensemble[u, v] = 0.

        strength_ensemble = self.direction_choosing(strength_ensemble)
        strength_ensemble[strength_ensemble < self.boundry] = 0.

        return self.optimization(strength_ensemble)

    def direction_choosing(self, X):
        n, m = X.shape
        for i in range(n):
            for j in range(m):
                if i > j:
                    if X[i, j] > X[j, i] > 0:
                        X[i, j] = 0.
                    if X[j, i] > X[i, j] > 0:
                        X[j, i] = 0.
        return X

    def optimization(self, X):
        n, _ = X.shape
        list_1 = np.linspace(0, n - 1, n).astype(int)

        for i in list_1:
            list_rest = list_1[list_1 != i]
            from itertools import combinations
            combi = list(combinations(list_rest, 2))
            for j in combi:
                if (X[i, j[0]] > 0.) & (X[i, j[1]] > 0.):
                    if (X[j[0]][j[1]]) > 0:
                        list_3 = np.array([X[i][j[0]], X[i][j[1]], X[j[0]][j[1]]])
                        list_3[np.argmin(list_3)] = 0
                        X[i][j[0]], X[i][j[1]], X[j[0]][j[1]] = list_3
                    if (X[j[1]][j[0]]) > 0:
                        list_3 = np.array([X[i][j[0]], X[i][j[1]], X[j[1]][j[0]]])
                        list_3[np.argmin(list_3)] = 0
                        X[i][j[0]], X[i][j[1]], X[j[1]][j[0]] = list_3
        return X
