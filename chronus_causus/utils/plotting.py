# chronus_causus/utils/plotting.py

"""Plotting Utilities."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_causal_graph(causal_matrix, feature_names):
    """
    Plot the causal graph.

    Parameters
    ----------
    causal_matrix : np.ndarray
        The causal matrix.
    feature_names : list of str
        The names of the features.
    """
    G = nx.DiGraph()
    for i, name in enumerate(feature_names):
        G.add_node(name)

    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            if causal_matrix[i, j] > 0:
                G.add_edge(feature_names[j], feature_names[i], weight=causal_matrix[i, j])

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
