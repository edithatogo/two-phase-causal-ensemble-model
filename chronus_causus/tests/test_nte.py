import pytest
import numpy as np
from chronus_causus.methods.nte import NTEDiscoverer

@pytest.fixture
def sample_data():
    """Create a sample time series dataset."""
    # Two independent time series
    ts1 = np.sin(np.linspace(0, 10, 100))
    ts2 = np.cos(np.linspace(0, 10, 100))
    return np.vstack([ts1, ts2]).T

def test_nte_discoverer_init():
    """Test the initialization of the NTEDiscoverer."""
    discoverer = NTEDiscoverer(lag=2, embed_dim=4)
    assert discoverer.lag == 2
    assert discoverer.embed_dim == 4
    assert discoverer.causal_matrix_ is None

def test_nte_discoverer_fit(sample_data):
    """Test the fit method of the NTEDiscoverer."""
    discoverer = NTEDiscoverer(lag=1, embed_dim=3)
    discoverer.fit(sample_data)
    assert discoverer.causal_matrix_ is not None
    assert discoverer.causal_matrix_.shape == (2, 2)

def test_nte_discoverer_invalid_input():
    """Test that the NTEDiscoverer raises an error for invalid input."""
    discoverer = NTEDiscoverer()
    with pytest.raises(TypeError):
        discoverer.fit("not a numpy array")
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100))
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(5, 3))
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100, 1))
