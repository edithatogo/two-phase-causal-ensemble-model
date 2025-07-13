# chronus_causus/utils/partitioning.py

"""Partitioning Utilities."""

import numpy as np

def overlapping_partitions(X, num_partitions, overlap):
    """
    Create overlapping partitions of a time series dataset.

    Parameters
    ----------
    X : np.ndarray
        The time series data.
    num_partitions : int
        The number of partitions to create.
    overlap : float
        The percentage of overlap between partitions.

    Returns
    -------
    partitions : list of np.ndarray
        A list of the partitions.
    """
    n_samples = X.shape[0]
    partition_size = int(n_samples / (num_partitions - (num_partitions - 1) * overlap))
    step_size = int(partition_size * (1 - overlap))

    partitions = []
    for i in range(num_partitions):
        start = i * step_size
        end = start + partition_size
        if end > n_samples:
            end = n_samples
        partitions.append(X[start:end])

    return partitions
