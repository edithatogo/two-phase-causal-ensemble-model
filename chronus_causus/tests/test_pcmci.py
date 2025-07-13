import pytest
import numpy as np
from chronus_causus.methods.pcmci import PCMCIDiscoverer

@pytest.fixture
def sample_data():
    """Create a sample time series dataset."""
    # Two independent time series
    ts1 = np.sin(np.linspace(0, 10, 100))
    ts2 = np.cos(np.linspace(0, 10, 100))
    return np.vstack([ts1, ts2]).T

def test_pcmci_discoverer_init():
    """Test the initialization of the PCMCIDiscoverer."""
    discoverer = PCMCIDiscoverer(max_lag=2, pc_alpha=0.1, cond_ind_test='GPDC')
    assert discoverer.max_lag == 2
    assert discoverer.pc_alpha == 0.1
    assert discoverer.cond_ind_test == 'GPDC'
    assert discoverer.causal_matrix_ is None

def test_pcmci_discoverer_fit(sample_data):
    """Test the fit method of the PCMCIDiscoverer."""
    discoverer = PCMCIDiscoverer(max_lag=1, pc_alpha=0.05)
    discoverer.fit(sample_data)
    assert discoverer.causal_matrix_ is not None
    assert discoverer.causal_matrix_.shape == (2, 2)

def test_pcmci_discoverer_invalid_input():
    """Test that the PCMCIDiscoverer raises an error for invalid input."""
    discoverer = PCMCIDiscoverer()
    with pytest.raises(TypeError):
        discoverer.fit("not a numpy array")
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100))
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(5, 3))
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100, 1))

def test_pcmci_discoverer_invalid_cond_ind_test():
    """Test that the PCMCIDiscoverer raises an error for invalid cond_ind_test."""
    discoverer = PCMCIDiscoverer(cond_ind_test='invalid_test')
    with pytest.raises(ValueError):
        discoverer.fit(np.random.rand(100, 2))
