# chronus_causus/tests/test_ccm.py

import unittest
import numpy as np

# Attempt to import CCMDiscoverer, handling potential path issues for tests
try:
    from chronus_causus.methods.ccm import CCMDiscoverer
except ImportError:
    # This fallback might be needed if the package isn't installed in editable mode
    # or if PYTHONPATH isn't set up for development.
    # For simplicity in this environment, we assume direct import works.
    # In a more complex setup, test runners and package structure handle this.
    # from ..methods.ccm import CCMDiscoverer # Example of relative import
    raise

class TestCCMDiscoverer(unittest.TestCase):
    """Test suite for the CCMDiscoverer class."""

    def setUp(self):
        """Set up basic data for tests."""
        self.valid_data = np.random.rand(100, 3)
        self.feature_names = ["var1", "var2", "var3"]

    def test_instantiation(self):
        """Test CCMDiscoverer instantiation with default and custom params."""
        # Default instantiation
        estimator_default = CCMDiscoverer()
        self.assertEqual(estimator_default.lag, 1)
        self.assertEqual(estimator_default.embed_dim, 3)
        self.assertEqual(estimator_default.split_percent, 0.75)
        self.assertEqual(estimator_default.max_lib_size_iter, 40)
        self.assertEqual(estimator_default.convergence_error_num, 8)
        self.assertEqual(estimator_default.convergence_threshold, 0.03)

        # Custom instantiation
        custom_params = {
            "lag": 2,
            "embed_dim": 4,
            "split_percent": 0.8,
            "max_lib_size_iter": 30,
            "convergence_error_num": 6,
            "convergence_threshold": 0.05,
        }
        estimator_custom = CCMDiscoverer(**custom_params)
        self.assertEqual(estimator_custom.lag, custom_params["lag"])
        self.assertEqual(estimator_custom.embed_dim, custom_params["embed_dim"])
        self.assertEqual(estimator_custom.split_percent, custom_params["split_percent"])
        self.assertEqual(estimator_custom.max_lib_size_iter, custom_params["max_lib_size_iter"])
        self.assertEqual(estimator_custom.convergence_error_num, custom_params["convergence_error_num"])
        self.assertEqual(estimator_custom.convergence_threshold, custom_params["convergence_threshold"])

    def test_error_method(self):
        """Test the static _error method."""
        self.assertAlmostEqual(CCMDiscoverer._error(10.0, 9.0), 0.1)
        self.assertAlmostEqual(CCMDiscoverer._error(10.0, 10.0), 0.0)
        self.assertAlmostEqual(CCMDiscoverer._error(5.0, 7.0), 0.4)
        self.assertEqual(CCMDiscoverer._error(0.0, 0.0), 0.0)
        self.assertEqual(CCMDiscoverer._error(0.0, 5.0), np.inf)
        self.assertTrue(np.isinf(CCMDiscoverer._error(0.0, 5.0))) # Explicit check for inf

    def test_validate_input(self):
        """Test the _validate_input method."""
        estimator = CCMDiscoverer()

        # Valid data
        validated_data = estimator._validate_input(self.valid_data.copy())
        np.testing.assert_array_equal(validated_data, self.valid_data)

        # Invalid type
        with self.assertRaisesRegex(TypeError, "Input X_partition must be a numpy array."):
            estimator._validate_input([[1, 2], [3, 4]]) # type: ignore

        # Invalid dimension (1D)
        with self.assertRaisesRegex(ValueError, "Input X_partition must be a 2D array"):
            estimator._validate_input(np.array([1, 2, 3, 4]))

        # Too few samples
        with self.assertRaisesRegex(ValueError, "Input X_partition must have at least 10 samples"):
            estimator._validate_input(np.random.rand(5, 2))

        # Too few features
        with self.assertRaisesRegex(ValueError, "Input X_partition must have at least 2 features"):
            estimator._validate_input(np.random.rand(20, 1))

    def test_fit_runs_basic(self):
        """Test that fit method runs and sets basic attributes."""
        estimator = CCMDiscoverer(lag=1, embed_dim=2, convergence_error_num=4) # Faster params for test

        # Using self.valid_data (100, 3)
        estimator.fit(self.valid_data.copy())

        self.assertIsNotNone(estimator.causal_matrix_)
        self.assertEqual(estimator.causal_matrix_.shape, (self.valid_data.shape[1], self.valid_data.shape[1]))
        self.assertEqual(estimator.n_features_in_, self.valid_data.shape[1])
        self.assertListEqual(list(estimator.feature_names_in_), [f"feature_{i}" for i in range(self.valid_data.shape[1])])

        # Test with provided feature names
        estimator.fit(self.valid_data.copy(), feature_names=self.feature_names)
        self.assertListEqual(list(estimator.feature_names_in_), self.feature_names)

        # Test with a slightly different shaped data to ensure robustness if any fixed indices were assumed
        data_4_features = np.random.rand(120, 4)
        estimator_4feat = CCMDiscoverer(lag=1, embed_dim=2, convergence_error_num=4)
        estimator_4feat.fit(data_4_features)
        self.assertEqual(estimator_4feat.causal_matrix_.shape, (4, 4))
        self.assertEqual(estimator_4feat.n_features_in_, 4)


if __name__ == '__main__':
    unittest.main()
