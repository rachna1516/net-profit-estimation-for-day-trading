import numpy as np
import pytest

import data_eda


def test_add_indicators_columns_and_values(ohlcv_df):
    out = data_eda.add_indicators(ohlcv_df)
    for col in ["Returns", "Cumulative_Returns", "SMA_20", "SMA_50", "Volatility"]:
        assert col in out.columns
    assert "Returns" not in ohlcv_df.columns  # input not mutated
    assert out["SMA_20"].iloc[:19].isna().all()
    assert out["SMA_20"].iloc[19] == pytest.approx(ohlcv_df["Open"].iloc[:20].mean())
    np.testing.assert_allclose(
        out["Cumulative_Returns"].iloc[-1], ohlcv_df["Open"].iloc[-1] / ohlcv_df["Open"].iloc[0]
    )


def test_run_eda_creates_output_dir_and_charts(ohlcv_df, tmp_path):
    out_dir = tmp_path / "nested" / "eda"  # does not exist yet
    saved = data_eda.run_eda(ohlcv_df, out_dir)
    assert len(saved) == 8  # 4 trends + correlation + volatility + moving averages + returns
    assert all(p.exists() and p.stat().st_size > 0 for p in saved)
