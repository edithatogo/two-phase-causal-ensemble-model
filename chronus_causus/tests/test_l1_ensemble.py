import pytest
import numpy as np
from chronus_causus.methods.ccm import CCMDiscoverer
from chronus_causus.ensemble.l1_ensemble import L1Ensembler

@pytest.fixture
def sample_data():
    """Create a sample time series dataset."""
    # Two independent time series
    ts1 = np.random.rand(100)
    ts2 = np.random.rand(100)
    return np.vstack([ts1, ts2]).T

def test_l1_ensembler_init():
    """Test the initialization of the L1Ensembler."""
    discoverer = CCMDiscoverer()
    ensembler = L1Ensembler(discoverer, num_partitions=5)
    assert ensembler.num_partitions == 5
    assert ensembler.causal_matrix_ is None
    assert ensembler.accuracy_index_ is None

def test_l1_ensembler_fit(sample_data):
    """Test the fit method of the L1Ensembler."""
    discoverer = CCMDiscoverer()
    ensembler = L1Ensembler(discoverer, num_partitions=5)
    ensembler.fit(sample_data)
    assert ensembler.causal_matrix_ is not None
    assert ensembler.causal_matrix_.shape == (2, 2)
    assert ensembler.accuracy_index_ is not None
    assert ensembler.accuracy_index_.shape == (2, 2)
