import numpy as np
import pandas as pd

from core.downsampling import downsample_series


def test_downsample_series_identity_when_below_threshold():
    y = pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30])
    result = downsample_series(y, None, max_points=10)
    assert result.method == "identity"
    assert result.sampled_count == 3
    assert result.original_count == 3
    pd.testing.assert_series_equal(result.y, y)
    assert result.x is None


def test_downsample_series_reduces_and_preserves_edges():
    n = 50_000
    x = pd.Series(pd.date_range("2024-01-01", periods=n, freq="ms"))
    y = pd.Series(np.sin(np.linspace(0, 200, n)), index=x.index)
    result = downsample_series(y, x, max_points=1_000)
    assert result.sampled_count <= 1_000
    assert result.y.index[0] == y.index[0]
    assert result.y.index[-1] == y.index[-1]
    assert np.isclose(result.y.iloc[0], y.iloc[0])
    assert np.isclose(result.y.iloc[-1], y.iloc[-1])
    assert result.x is not None
    assert result.x.iloc[0] == x.iloc[0]
    assert result.x.iloc[-1] == x.iloc[-1]


def test_downsample_series_handles_none_x():
    n = 10_000
    y = pd.Series(np.linspace(0, 1, n))
    result = downsample_series(y, None, max_points=500)
    assert result.sampled_count <= 500
    assert result.x is None
    assert result.y.iloc[0] == y.iloc[0]
    assert result.y.iloc[-1] == y.iloc[-1]
