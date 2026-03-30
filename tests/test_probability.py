"""Tests for prbtrd.probability."""

import pytest

from prbtrd.probability import ProbabilityEstimator


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_at_least_two_prices():
    with pytest.raises(ValueError, match="At least 2"):
        ProbabilityEstimator([100.0])


def test_requires_nonempty_prices():
    with pytest.raises(ValueError):
        ProbabilityEstimator([])


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------


def test_win_rate_all_up():
    prices = [1, 2, 3, 4, 5]
    est = ProbabilityEstimator(prices)
    assert est.win_rate(lookback=4) == 1.0


def test_win_rate_all_down():
    prices = [5, 4, 3, 2, 1]
    est = ProbabilityEstimator(prices)
    assert est.win_rate(lookback=4) == 0.0


def test_win_rate_half_and_half():
    # Up, down, up, down
    prices = [1, 2, 1, 2, 1]
    est = ProbabilityEstimator(prices)
    assert est.win_rate(lookback=4) == 0.5


def test_win_rate_invalid_lookback():
    est = ProbabilityEstimator([1, 2])
    with pytest.raises(ValueError, match="lookback"):
        est.win_rate(lookback=0)


def test_win_rate_lookback_larger_than_data():
    """When lookback > available bars, use all available bars."""
    prices = [1, 2, 3]
    est = ProbabilityEstimator(prices)
    # Only 2 transitions available
    assert est.win_rate(lookback=100) == 1.0


# ---------------------------------------------------------------------------
# trend_strength
# ---------------------------------------------------------------------------


def test_trend_strength_bullish():
    """Rising price series → short SMA > long SMA → probability > 0.5."""
    prices = list(range(1, 22))  # 21 values
    est = ProbabilityEstimator(prices)
    ts = est.trend_strength(short_window=5, long_window=20)
    assert ts > 0.5


def test_trend_strength_bearish():
    """Falling price series → short SMA < long SMA → probability < 0.5."""
    prices = list(range(21, 0, -1))  # 21 values, descending
    est = ProbabilityEstimator(prices)
    ts = est.trend_strength(short_window=5, long_window=20)
    assert ts < 0.5


def test_trend_strength_neutral_insufficient_data():
    """With fewer prices than long_window, returns neutral 0.5."""
    prices = [100, 101, 102]
    est = ProbabilityEstimator(prices)
    ts = est.trend_strength(short_window=2, long_window=20)
    assert ts == 0.5


def test_trend_strength_invalid_windows():
    prices = list(range(25))
    est = ProbabilityEstimator(prices)
    with pytest.raises(ValueError):
        est.trend_strength(short_window=0, long_window=20)
    with pytest.raises(ValueError):
        est.trend_strength(short_window=20, long_window=5)


def test_trend_strength_clamped_to_one():
    """Extremely steep rally should not exceed 1.0."""
    prices = [1] * 19 + [1000, 1000]
    est = ProbabilityEstimator(prices)
    assert est.trend_strength(5, 20) <= 1.0


def test_trend_strength_clamped_to_zero():
    """Extremely steep decline should not go below 0.0."""
    prices = [1000] * 19 + [1, 1]
    est = ProbabilityEstimator(prices)
    assert est.trend_strength(5, 20) >= 0.0


# ---------------------------------------------------------------------------
# composite_probability
# ---------------------------------------------------------------------------


def test_composite_probability_range():
    prices = list(range(1, 25))
    est = ProbabilityEstimator(prices)
    cp = est.composite_probability()
    assert 0.0 <= cp <= 1.0


def test_composite_equal_weights():
    prices = list(range(1, 25))
    est = ProbabilityEstimator(prices)
    wr = est.win_rate()
    ts = est.trend_strength()
    expected = (wr + ts) / 2.0
    assert abs(est.composite_probability() - expected) < 1e-9


def test_composite_zero_weights_raises():
    est = ProbabilityEstimator([1, 2])
    with pytest.raises(ValueError):
        est.composite_probability(win_rate_weight=0.0, trend_weight=0.0)
