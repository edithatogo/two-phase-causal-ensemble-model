# chronus_causus/utils/evaluation.py

"""Evaluation Utilities."""

import numpy as np

def precision_recall_f1(y_true, y_pred):
    """
    Calculate precision, recall, and F1-score.

    Parameters
    ----------
    y_true : np.ndarray
        The true causal matrix.
    y_pred : np.ndarray
        The predicted causal matrix.

    Returns
    -------
    precision : float
        The precision score.
    recall : float
        The recall score.
    f1 : float
        The F1-score.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def evaluation_index(y_true, y_pred):
    """
    Calculate the evaluation index from the paper.
    (Placeholder for now)
    """
    pass
