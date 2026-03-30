"""Tests for prbtrd.trader."""

import pytest

from prbtrd.trader import Signal, Trader, TradeResult


# Strongly bullish: 24 up bars, last 5 bars much higher than previous 20
BULLISH_PRICES = list(range(1, 26))   # [1, 2, …, 25]
# Strongly bearish: 24 down bars
BEARISH_PRICES = list(range(25, 0, -1))  # [25, 24, …, 1]
# Sideways: alternating, centred around a constant
SIDEWAYS_PRICES = [100 + (i % 2) * 0.01 for i in range(25)]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_thresholds_equal():
    with pytest.raises(ValueError):
        Trader([1, 2], long_threshold=0.5, short_threshold=0.5)


def test_invalid_thresholds_inverted():
    with pytest.raises(ValueError):
        Trader([1, 2], long_threshold=0.4, short_threshold=0.6)


def test_invalid_thresholds_out_of_range():
    with pytest.raises(ValueError):
        Trader([1, 2], long_threshold=1.1, short_threshold=0.45)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def test_evaluate_returns_trade_result():
    trader = Trader(BULLISH_PRICES)
    result = trader.evaluate()
    assert isinstance(result, TradeResult)


def test_long_signal_on_bullish_prices():
    trader = Trader(BULLISH_PRICES, long_threshold=0.55, short_threshold=0.45)
    result = trader.evaluate()
    assert result.signal == Signal.LONG


def test_short_signal_on_bearish_prices():
    trader = Trader(BEARISH_PRICES, long_threshold=0.55, short_threshold=0.45)
    result = trader.evaluate()
    assert result.signal == Signal.SHORT


def test_neutral_signal_on_sideways_prices():
    trader = Trader(SIDEWAYS_PRICES, long_threshold=0.55, short_threshold=0.45)
    result = trader.evaluate()
    assert result.signal == Signal.NEUTRAL


def test_probability_in_range():
    for prices in (BULLISH_PRICES, BEARISH_PRICES, SIDEWAYS_PRICES):
        result = Trader(prices).evaluate()
        assert 0.0 <= result.probability <= 1.0


def test_thresholds_stored_on_result():
    trader = Trader(BULLISH_PRICES, long_threshold=0.6, short_threshold=0.4)
    result = trader.evaluate()
    assert result.long_threshold == 0.6
    assert result.short_threshold == 0.4


def test_wide_thresholds_force_neutral():
    """When long_threshold is very close to 1, almost everything is NEUTRAL."""
    trader = Trader(BULLISH_PRICES, long_threshold=0.99, short_threshold=0.01)
    result = trader.evaluate()
    assert result.signal == Signal.NEUTRAL


def test_narrow_thresholds_force_signal():
    """When both thresholds are near 0.5, any trend generates a signal."""
    # Slightly relaxed thresholds: long>0.501, short<0.499
    trader = Trader(BULLISH_PRICES, long_threshold=0.501, short_threshold=0.499)
    result = trader.evaluate()
    assert result.signal in (Signal.LONG, Signal.SHORT)


# ---------------------------------------------------------------------------
# Signal enum values
# ---------------------------------------------------------------------------


def test_signal_values():
    assert Signal.LONG.value == "LONG"
    assert Signal.SHORT.value == "SHORT"
    assert Signal.NEUTRAL.value == "NEUTRAL"
