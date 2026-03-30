"""Tests for prbtrd.trader."""

import numpy as np
import pandas as pd
import pytest

from prbtrd.trader import LONG, NEUTRAL, SHORT, decide, trade


def _sample_prices(n: int = 60, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, dtype=float)


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

class TestDecide:
    def test_long_signal(self):
        assert decide(0.75) == LONG

    def test_short_signal(self):
        assert decide(0.25) == SHORT

    def test_neutral_signal(self):
        assert decide(0.5) == NEUTRAL

    def test_boundary_at_long_threshold_is_neutral(self):
        # probability == threshold is NOT strictly above → NEUTRAL
        assert decide(0.6, long_threshold=0.6) == NEUTRAL

    def test_boundary_at_short_threshold_is_neutral(self):
        # probability == threshold is NOT strictly below → NEUTRAL
        assert decide(0.4, short_threshold=0.4) == NEUTRAL

    def test_custom_thresholds(self):
        assert decide(0.8, long_threshold=0.7, short_threshold=0.3) == LONG
        assert decide(0.2, long_threshold=0.7, short_threshold=0.3) == SHORT
        assert decide(0.5, long_threshold=0.7, short_threshold=0.3) == NEUTRAL


# ---------------------------------------------------------------------------
# trade
# ---------------------------------------------------------------------------

class TestTrade:
    def test_returns_valid_signal(self):
        signal = trade(_sample_prices())
        assert signal in (LONG, SHORT, NEUTRAL)

    def test_accepts_learner(self):
        from prbtrd.learner import Learner

        signal = trade(_sample_prices(), learner=Learner())
        assert signal in (LONG, SHORT, NEUTRAL)

    def test_custom_thresholds_all_long(self):
        """With thresholds at 0/0, every probability produces LONG."""
        signal = trade(
            _sample_prices(),
            long_threshold=0.0,
            short_threshold=0.0,
        )
        assert signal == LONG

    def test_custom_thresholds_all_short(self):
        """With thresholds at 1/1, every probability produces SHORT."""
        signal = trade(
            _sample_prices(),
            long_threshold=1.0,
            short_threshold=1.0,
        )
        assert signal == SHORT
