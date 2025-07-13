import pytest
import numpy as np
from chronus_causus.utils.evaluation import precision_recall_f1

def test_precision_recall_f1():
    """Test the precision_recall_f1 function."""
    y_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])

    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    assert precision == 0.6666666666666666
    assert recall == 1.0
    assert f1 == 0.8
