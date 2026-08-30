"""Tests for return construction, focused on the gap rules."""

import numpy as np
import pandas as pd

from engine.data import MAX_GAP_CALENDAR_DAYS, to_returns


def _frame(dates, values, col="X"):
    return pd.DataFrame({col: values}, index=pd.to_datetime(dates))


def test_clean_series_matches_simple_returns():
    dates = pd.bdate_range("2020-01-06", periods=5)
    prices = _frame(dates, [100.0, 101.0, 99.0, 102.0, 102.5])
    out = to_returns(prices)["X"]

    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:], [0.01, -2 / 101, 3 / 99, 0.5 / 102], rtol=1e-12)


def test_missing_price_kills_both_adjacent_returns():
    dates = pd.bdate_range("2020-01-06", periods=5)
    prices = _frame(dates, [100.0, 101.0, np.nan, 103.0, 104.0])
    out = to_returns(prices)["X"]

    # The gap day itself and the day that would otherwise span it are both dropped.
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])
    # Days on either side of the hole are untouched.
    np.testing.assert_allclose(out.iloc[1], 0.01)
    np.testing.assert_allclose(out.iloc[4], 1 / 103)


def test_never_forward_fills_across_a_hole():
    """A filled gap would show a 0 return then a compressed jump. Neither appears."""
    dates = pd.bdate_range("2020-01-06", periods=6)
    prices = _frame(dates, [100.0, np.nan, np.nan, 130.0, 131.0, 132.0])
    out = to_returns(prices)["X"]

    assert out.iloc[1:4].isna().all()
    # The 30 percent move across the hole is never attributed to a single day.
    assert not np.isclose(out.dropna().to_numpy(), 0.30).any()
    np.testing.assert_allclose(out.iloc[4], 1 / 130)


def test_wide_calendar_gap_drops_the_bridging_return():
    """Adjacent rows in the frame can still be months apart on the calendar."""
    dates = ["2020-01-06", "2020-01-07", "2020-04-15", "2020-04-16"]
    prices = _frame(dates, [100.0, 101.0, 140.0, 141.0])
    out = to_returns(prices)["X"]

    np.testing.assert_allclose(out.iloc[1], 0.01)
    assert np.isnan(out.iloc[2]), "return bridging a 99 day gap must be dropped"
    np.testing.assert_allclose(out.iloc[3], 1 / 140)


def test_normal_weekend_and_holiday_survive():
    """A three day weekend is a gap of three calendar days and must be kept."""
    dates = ["2020-01-17", "2020-01-21"]  # Friday to Tuesday, MLK Monday closed
    prices = _frame(dates, [100.0, 102.0])
    out = to_returns(prices)["X"]

    assert (pd.to_datetime(dates[1]) - pd.to_datetime(dates[0])).days <= MAX_GAP_CALENDAR_DAYS
    np.testing.assert_allclose(out.iloc[1], 0.02)


def test_columns_are_gapped_independently():
    dates = pd.bdate_range("2020-01-06", periods=4)
    prices = pd.DataFrame(
        {"A": [10.0, 11.0, np.nan, 12.0], "B": [20.0, 21.0, 22.0, 23.0]},
        index=dates,
    )
    out = to_returns(prices)

    assert out["A"].iloc[2:4].isna().all()
    assert out["B"].iloc[1:].notna().all()
