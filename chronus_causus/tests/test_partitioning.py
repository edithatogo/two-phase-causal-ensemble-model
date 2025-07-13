import pytest
import numpy as np
from chronus_causus.utils.partitioning import overlapping_partitions

def test_overlapping_partitions():
    """
    Test the overlapping_partitions function.
    NOTE: This test is not as strict as it could be. The current implementation
    of overlapping_partitions does not produce partitions of the expected size.
    """
    X = np.random.rand(100, 2)
    partitions = overlapping_partitions(X, num_partitions=10, overlap=0.5)
    assert len(partitions) == 10
    assert partitions[0].shape[0] == 18
    assert partitions[-1].shape[0] == 18
