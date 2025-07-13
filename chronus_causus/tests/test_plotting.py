import pytest
import numpy as np
from chronus_causus.utils.plotting import plot_causal_graph

def test_plot_causal_graph():
    """Test the plot_causal_graph function."""
    causal_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    feature_names = ['a', 'b', 'c']
    plot_causal_graph(causal_matrix, feature_names)
