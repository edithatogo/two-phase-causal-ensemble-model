# chronus_causus/ensemble/l1_ensemble.py

"""L1 Ensembler."""

import numpy as np
from sklearn.mixture import GaussianMixture

class L1Ensembler:
    """
    L1 Ensembler.

    The L1 ensembler applies a single base causal discovery algorithm to
    multiple data partitions and combines the results using a Gaussian
    Mixture Model (GMM).

    Parameters
    ----------
    estimator : object
        A causal discovery algorithm that has a `fit` method.
    num_partitions : int, default=10
        The number of data partitions to create.

    Attributes
    ----------
    causal_matrix_ : np.ndarray
        The computed causal matrix after fitting.
    accuracy_index_ : np.ndarray
        The accuracy index for each causal link.
    """
    def __init__(self, estimator, num_partitions: int = 10):
        self.estimator = estimator
        self.num_partitions = num_partitions
        self.causal_matrix_ = None
        self.accuracy_index_ = None

    def fit(self, X, y=None):
        """
        Fit the L1 ensembler.

        Args:
            X (np.ndarray): A 2D array representing the time series data.
                              Shape (n_samples, n_features).
            y : None, ignored.

        Returns
        -------
        self : L1Ensembler
            The fitted estimator.
        """
        n_samples, n_features = X.shape
        partition_size = n_samples // self.num_partitions

        results = []
        for i in range(self.num_partitions):
            partition = X[i * partition_size:(i + 1) * partition_size]
            self.estimator.fit(partition)
            results.append(self.estimator.causal_matrix_)

        results = np.array(results)
        self.causal_matrix_ = np.zeros((n_features, n_features))
        self.accuracy_index_ = np.zeros((n_features, n_features))

        self.causal_matrix_ = self.l1_ensemble(results)
        self.accuracy_index_ = self.accuracy_index(results)

        return self

    def l1_ensemble(self, X):
        n, m, l = X.shape
        ensemble_1 = np.zeros((m, l))
        data_reshape = []

        for u in range(m):
            for v in range(l):
                data_reshape.append(X[:, u, v].tolist())
        data_reshape = np.array(data_reshape)

        gmm = GaussianMixture(n_components=2, random_state=0, n_init=1)
        r = gmm.fit_predict(data_reshape).reshape((m, l))
        gmm_means = gmm.means_
        gmm_non = np.argmin(gmm_means.mean(axis=1))

        for u in range(m):
            for v in range(l):
                a = X[:, u, v]
                if r[u, v] != gmm_non:
                    a = a[a != 0]
                    if len(a) > 0:
                        ensemble_1[u, v] = np.median(a)

        return self.direction_choosing(ensemble_1)

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

    def accuracy_index(self, X):
        n, m, l = X.shape
        acc = np.zeros((m, l))

        def score(x):
            x = x[x > 0.]
            num = x.shape[0]
            if num < 2:
                return 0.
            else:
                std = np.around(np.std(x), 4)
                mean = np.around(np.mean(x), 4)
                return np.round((mean / (std + 1e-8)) * num / n, 4)

        for u in range(m):
            for v in range(l):
                acc[u, v] = score(X[:, u, v])
        return acc
