# chronus_causus/methods/ccm.py

"""Convergent Cross Mapping (CCM) Causal Discovery Method."""

import numpy as np
import math # Added for math.isnan
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

    @staticmethod
    def _error(a, b):
        """
        Relative error to determine convergence.
        Adapted from demostration/utilities_ccm.py.
        """
        # Handles cases where 'a' can be zero, causing ZeroDivisionError.
        # If 'a' is zero, relative error is problematic.
        # If both a and b are zero, error is 0. If a is zero and b is not, error is effectively infinite (or very large).
        # The original code implies if a is zero, error is large (or NaN handled later).
        # Let's ensure 'a' isn't zero before division.
        if a == 0:
            if b == 0:
                return 0.0  # No difference
            else:
                return np.inf # Or a very large number, signifies large difference
        err = abs((a - b) / a)
        return err

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

            # Initialize skccm.CCM
            # Ensure skccm was imported as ccm_sk earlier or handle import here
            if 'ccm_sk' not in locals() and 'ccm_sk' not in globals():
                try:
                    import skccm as ccm_sk
                except ImportError:
                    raise ImportError("skccm library is required for CCMDiscoverer. Please install it.")

            ccm_instance = ccm_sk.CCM()

            # Define library lengths (lib_lens)
            len_tr = len(x1tr) # Length of the training set for the first series

            # Ensure len_tr is sufficient for lib_lens generation
            # The original demo starts lib_lens from 20.
            # max_lib_size_iter should not lead to a step of 0 or negative.
            if len_tr < 20 or self.max_lib_size_iter <= 0:
                print(f"Skipping pair ({feat_idx1}, {feat_idx2}) due to insufficient training length ({len_tr}) for lib_lens generation.")
                continue

            # Calculate step ensuring it's at least 1, or handle small len_tr
            step = len_tr / self.max_lib_size_iter
            if step < 1: # If len_tr is smaller than max_lib_size_iter, iterate one by one
                step = 1
                # Adjust max_lib_size_iter to avoid very long computations if len_tr is small
                # This part might need refinement based on skccm behavior with small lib_lens steps
                # For now, let's cap iterations if step becomes 1.
                # effective_iters = len_tr - 20 + 1 # Iterate up to full library

            # Ensure the start of lib_lens (e.g., 20) is less than len_tr
            lib_lens_start = 20
            if lib_lens_start >= len_tr:
                 print(f"Skipping pair ({feat_idx1}, {feat_idx2}) as lib_lens_start ({lib_lens_start}) >= len_tr ({len_tr}).")
                 continue

            lib_lens = np.arange(lib_lens_start, len_tr, step, dtype='int')
            if len(lib_lens) == 0: # If arange results in empty, add at least one point if possible
                if len_tr > lib_lens_start:
                    lib_lens = np.array([min(len_tr-1, lib_lens_start + int(step))]) # ensure one step if possible
                else: # Cannot form a valid lib_lens
                    print(f"Skipping pair ({feat_idx1}, {feat_idx2}) due to inability to form lib_lens.")
                    continue
            if lib_lens[-1] < len_tr - step : # Ensure the last point is close to len_tr
                 lib_lens = np.append(lib_lens, len_tr-1)


            # Fit CCM
            ccm_instance.fit(x1tr, x2tr)

            # Predict
            # Note: predict can sometimes fail if lib_lengths are too large relative to test set,
            # or if test sets are too small. skccm might have internal checks.
            try:
                x1p, x2p = ccm_instance.predict(x1te, x2te, lib_lengths=lib_lens)
            except Exception as e:
                print(f"CCM predict failed for pair ({feat_idx1}, {feat_idx2}): {e}. Skipping.")
                continue

            # Get scores
            # sc1: X2 -> X1 (how well X1_test is predicted using X2_train library)
            # sc2: X1 -> X2 (how well X2_test is predicted using X1_train library)
            sc1, sc2 = ccm_instance.score()

            # Convergence Check
            final_sc1 = 0.0
            if len(sc1) >= self.convergence_error_num:
                errors1 = []
                for j in range(self.convergence_error_num - 1):
                    err_val = self._error(sc1[-(j + 1)], sc1[-(j + 2)])
                    errors1.append(10.0 if math.isnan(err_val) or np.isinf(err_val) else err_val)

                if np.max(errors1) < self.convergence_threshold and sc1[-1] >= 1e-4:
                    final_sc1 = np.mean(sc1[-(self.convergence_error_num // 2):]) # Avg last half of convergence window
                    # Original demo used last 3: (sc1[-1] + sc1[-2] + sc1[-3]) / 3
                    # Using a portion of convergence_error_num for averaging might be more robust.
                    # For simplicity and consistency with demo, let's use last 3 if available, else fewer.
                    num_avg_points1 = min(3, len(sc1))
                    final_sc1 = np.mean(sc1[-num_avg_points1:])


            final_sc2 = 0.0
            if len(sc2) >= self.convergence_error_num:
                errors2 = []
                for j in range(self.convergence_error_num - 1):
                    err_val = self._error(sc2[-(j + 1)], sc2[-(j + 2)])
                    errors2.append(10.0 if math.isnan(err_val) or np.isinf(err_val) else err_val)

                if np.max(errors2) < self.convergence_threshold and sc2[-1] >= 1e-4:
                    num_avg_points2 = min(3, len(sc2))
                    final_sc2 = np.mean(sc2[-num_avg_points2:])

            # Populate causal matrix based on converged scores
            # sc1 relates to X2 -> X1 (effect_idx=feat_idx1, cause_idx=feat_idx2)
            # sc2 relates to X1 -> X2 (effect_idx=feat_idx2, cause_idx=feat_idx1)
            if final_sc1 > final_sc2 and final_sc1 > 0:
                self.causal_matrix_[feat_idx1, feat_idx2] = round(final_sc1, 4)
            elif final_sc2 > final_sc1 and final_sc2 > 0:
                self.causal_matrix_[feat_idx2, feat_idx1] = round(final_sc2, 4)

            # If scores are equal and positive, or one is zero, no causal link is asserted here by this logic.
            # The original demo code implies this exclusivity.

            # print(f"Processed pair: ({self.feature_names_in_[feat_idx1]}, {self.feature_names_in_[feat_idx2]})")
            # print(f"  Scores raw sc1: {sc1}, sc2: {sc2}")
            # print(f"  Converged sc1: {final_sc1}, Converged sc2: {final_sc2}")
            # print(f"  Updated causal_matrix_[{feat_idx1}, {feat_idx2}]: {self.causal_matrix_[feat_idx1, feat_idx2]}")
            # print(f"  Updated causal_matrix_[{feat_idx2}, {feat_idx1}]: {self.causal_matrix_[feat_idx2, feat_idx1]}")


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
