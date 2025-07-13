import pytest
import numpy as np
from chronus_causus.methods.ccm import CCMDiscoverer
from chronus_causus.methods.nte import NTEDiscoverer
from chronus_causus.methods.pcmci import PCMCIDiscoverer
from chronus_causus.methods.gc import GCDiscoverer
from chronus_causus.ensemble.causal_ensemble import CausalEnsemble

@pytest.fixture
def sample_data():
    """Create a sample time series dataset."""
    # Two independent time series
    ts1 = np.random.rand(1000)
    ts2 = np.random.rand(1000)
    return np.vstack([ts1, ts2]).T

def test_causal_ensemble_init():
    """Test the initialization of the CausalEnsemble."""
    estimators = [
        ('ccm', CCMDiscoverer()),
        ('nte', NTEDiscoverer()),
        ('pcmci', PCMCIDiscoverer()),
        ('gc', GCDiscoverer(max_lag=1))
    ]
    ensembler = CausalEnsemble(estimators, num_partitions=5)
    assert ensembler.num_partitions == 5
    assert ensembler.causal_matrix_ is None

def test_causal_ensemble_fit(sample_data):
    """Test the fit method of the CausalEnsemble."""
    estimators = [
        ('ccm', CCMDiscoverer()),
        ('nte', NTEDiscoverer()),
        ('pcmci', PCMCIDiscoverer()),
        ('gc', GCDiscoverer())
    ]
    ensembler = CausalEnsemble(estimators, num_partitions=5)
    ensembler.fit(sample_data)
    assert ensembler.causal_matrix_ is not None
    assert ensembler.causal_matrix_.shape == (2, 2)
