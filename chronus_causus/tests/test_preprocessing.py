import pytest
import numpy as np
import pandas as pd
from chronus_causus.utils.preprocessing import adf_test
from statsmodels.tsa.stattools import adfuller

def test_adf_test():
    """Test the adf_test function."""
    # Create a non-stationary time series
    ts = np.cumsum(np.random.randn(100))
    df = pd.DataFrame(ts, columns=['ts'])

    # Test that the function returns a stationary time series
    df_stationary = adf_test(df)
    r = adfuller(df_stationary['ts'], autolag='AIC')
    assert r[1] < 0.05
