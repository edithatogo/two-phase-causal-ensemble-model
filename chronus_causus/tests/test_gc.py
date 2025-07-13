import pytest
import numpy as np
from chronus_causus.methods.gc import GCDiscoverer

@pytest.fixture
def sample_data():
    """Create a sample time series dataset."""
    # Two independent time series
    ts1 = np.random.randn(100)
    ts2 = np.random.randn(100)
    return np.vstack([ts1, ts2]).T

def test_gc_discoverer_init():
    """Test the initialization of the GCDiscoverer."""
    discoverer = GCDiscoverer(max_lag=2, signif=0.1, threshold=0.2)
    assert discoverer.max_lag == 2
    assert discoverer.signif == 0.1
    assert discoverer.threshold == 0.2
    assert discoverer.causal_matrix_ is None

def test_gc_discoverer_fit(sample_data):
    """Test the fit method of the GCDiscoverer."""
    discoverer = GCDiscoverer(max_lag=1, signif=0.05, threshold=0.3)
    discoverer.fit(sample_data)
    assert discoverer.causal_matrix_ is not None
    assert discoverer.causal_matrix_.shape == (2, 2)

def test_gc_discoverer_invalid_input():
    """Test that the GCDiscoverer raises an error for invalid input."""
    discoverer = GCDiscoverer()
    with pytest.raises(TypeError):
        discoverer.fit("not a numpy array")
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100))
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(5, 3))
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100, 1))
