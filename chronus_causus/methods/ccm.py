# chronus_causus/methods/ccm.py

"""Convergent Cross Mapping (CCM) Causal Discovery Method."""

import numpy as np
# Potentially: from sklearn.base import BaseEstimator

class CCMDiscoverer: # Could inherit from BaseEstimator later
    """
    Causal discovery using Convergent Cross Mapping (CCM).

    CCM tests for causality between time series by measuring the extent
    to which the historical record of one time series can reliably estimate
    the state of another. If states of variable Y can be predicted from
    the library of states of variable X, then X is considered to cause Y.

    Parameters
    ----------
    lag : int, default=1
        The time lag to use for constructing the shadow manifold.
        (This is a common CCM parameter, subject to review based on demostration code)
    embed_dim : int, default=3
        The embedding dimension for state space reconstruction.
        (This is a common CCM parameter, subject to review based on demostration code)
    # Add other relevant hyperparameters from the demonstration code/paper.

    Attributes
    ----------
    causal_matrix_ : np.ndarray
        The computed causal matrix after fitting.
        The shape is (n_features, n_features).
        Entry (i, j) > 0 might indicate a causal link from feature j to feature i.
        More precisely, `causal_matrix_[effect_idx, cause_idx]` stores the strength.
    feature_names_in_ : list[str]
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    """
    def __init__(self,
                 lag: int = 1,
                 embed_dim: int = 3,
                 split_percent: float = 0.75,
                 max_lib_size_iter: int = 40,
                 convergence_error_num: int = 8,
                 convergence_threshold: float = 0.03,
                 **kwargs):
        """
        Initialize the CCMDiscoverer.

        Args:
            lag (int): The time lag to use for constructing the shadow manifold.
            embed_dim (int): The embedding dimension for state space reconstruction.
            split_percent (float): Percent of the data to use for the training set
                                   when performing cross-mapping.
            max_lib_size_iter (int): Number of iterations for varying library sizes
                                     to test for convergence.
            convergence_error_num (int): Number of recent score values to use for
                                         calculating relative error to check convergence.
            convergence_threshold (float): The threshold for relative error to determine
                                           if CCM scores have converged.
            **kwargs: Additional keyword arguments (currently unused).
        """
        self.lag = lag
        self.embed_dim = embed_dim
        self.split_percent = split_percent
        self.max_lib_size_iter = max_lib_size_iter
        self.convergence_error_num = convergence_error_num
        self.convergence_threshold = convergence_threshold

        self.causal_matrix_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = 0
        # Initialize other attributes

    def _validate_input(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X_partition must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input X_partition must be a 2D array (n_samples, n_features).")
        if X.shape[0] < 10: # Arbitrary minimum, can be adjusted
             raise ValueError(f"Input X_partition must have at least 10 samples, got {X.shape[0]}.")
        if X.shape[1] < 2:
            raise ValueError(f"Input X_partition must have at least 2 features, got {X.shape[1]}.")
        return X

    def fit(self, X_partition: np.ndarray, feature_names: list[str] = None):
        """
        Compute the causal matrix for a single data partition using CCM.

        Args:
            X_partition (np.ndarray): A 2D array representing the time series data
                                      for a single partition. Shape (n_samples, n_features).
            feature_names (list[str], optional): List of feature names.
                                                 If provided, used for `self.feature_names_in_`.

        Returns
        -------
        self : CCMDiscoverer
            The fitted estimator.
        """
        X_partition = self._validate_input(X_partition)
        n_samples, n_features = X_partition.shape

        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError("Length of feature_names must match the number of features in X_partition.")
            self.feature_names_in_ = feature_names
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(n_features)]
        self.n_features_in_ = n_features

        # Embed all time series
        X_embed = []
        for i in range(n_features):
            # Ensure skccm is imported if not already at the top level
            # For now, assuming it will be (e.g. import skccm as ccm_lib)
            # This might need to be `import skccm as ccm_sk` to avoid conflict if we have a var `ccm`
            try:
                import skccm as ccm_sk # Renaming to avoid conflict with potential ccm variable
            except ImportError:
                raise ImportError("skccm library is required for CCMDiscoverer. Please install it.")

            embedder = ccm_sk.Embed(X_partition[:, i])
            # Note: The original demo code X_embed.append(e.embed_vectors_1d(lag,embed))
            # uses self.lag and self.embed_dim
            X_embed.append(embedder.embed_vectors_1d(self.lag, self.embed_dim))

        self.causal_matrix_ = np.zeros((n_features, n_features))

        # Generate combinations of feature indices
        # (Using itertools.combinations as in the demonstration code)
        from itertools import combinations
        feature_indices = np.arange(n_features)
        feature_pairs = list(combinations(feature_indices, 2))

        for idx_pair in feature_pairs:
            feat_idx1, feat_idx2 = idx_pair

            # Get the embedded time series for the pair
            embedded_ts1 = X_embed[feat_idx1]
            embedded_ts2 = X_embed[feat_idx2]

            # Split embedded time series into training and testing sets
            # This also needs skccm.utilities
            try:
                from skccm.utilities import train_test_split
            except ImportError:
                raise ImportError("skccm.utilities.train_test_split is required. Ensure skccm is correctly installed.")

            # Ensure there are enough samples in embedded_ts for train_test_split
            # The minimum length for splitting depends on skccm's internal logic,
            # but typically, we need more than a few samples.
            # A common check is if len(embedded_ts1) > some_minimum (e.g., 20 or what lib_lens starts with)
            # For now, we assume embedded_ts are long enough.
            # Error handling for too short series might be needed here or in skccm.
            if len(embedded_ts1) < self.convergence_error_num * 2 or len(embedded_ts2) < self.convergence_error_num * 2 : # Rough check
                 print(f"Skipping pair ({feat_idx1}, {feat_idx2}) due to insufficient embedded length after lagging/embedding.")
                 continue


            x1tr, x1te, x2tr, x2te = train_test_split(
                embedded_ts1, embedded_ts2, percent=self.split_percent
            )

            # TODO:
            # 1. Initialize skccm.CCM()
            # 2. Define library lengths (lib_lens)
            # 3. Fit CCM: CCM.fit(x1tr, x2tr)
            # 4. Predict: x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
            # 5. Get scores: sc1, sc2 = CCM.score()
            # 6. Implement convergence check (using self.convergence_error_num, self.convergence_threshold)
            # 7. Populate self.causal_matrix_ based on converged scores.
            #    Remember: causal_matrix_[effect, cause] = strength
            #    If sc1 (X2->X1) is significant, causal_matrix_[feat_idx1, feat_idx2] = sc1_converged
            #    If sc2 (X1->X2) is significant, causal_matrix_[feat_idx2, feat_idx1] = sc2_converged

            # Placeholder for further implementation
            # print(f"Processing pair: ({self.feature_names_in_[feat_idx1]}, {self.feature_names_in_[feat_idx2]})")
            # print(f"  x1tr shape: {x1tr.shape}, x2tr shape: {x2tr.shape}")
            # print(f"  x1te shape: {x1te.shape}, x2te shape: {x2te.shape}")


        return self

    # Potentially add other methods like:
    # - predict (if applicable, though less common for pure discovery)
    # - get_params
    # - set_params
    # - _validate_input

if __name__ == '__main__':
    # Example Usage (for testing during development)
    print("Running basic CCMDiscoverer example...")
    dummy_data = np.random.rand(100, 3) # 100 samples, 3 features

    ccm_discoverer = CCMDiscoverer(lag=2, embed_dim=3)
    ccm_discoverer.fit(dummy_data)

    if ccm_discoverer.causal_matrix_ is not None:
        print("Fitted causal_matrix_ (example):")
        print(ccm_discoverer.causal_matrix_)
    else:
        print("Causal matrix not yet implemented in fit method.")

    print("CCMDiscoverer example finished.")
