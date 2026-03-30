"""Probability estimation from historical price data.

The ``ProbabilityEstimator`` accepts a sequence of closing prices and exposes
two complementary probability metrics:

* **win_rate(lookback)** – the fraction of past ``lookback`` bars on which the
  price closed *higher* than the previous bar.  Values above 0.5 indicate
  upward momentum; values below 0.5 indicate downward momentum.

* **trend_strength(short_window, long_window)** – a 0–1 value derived from a
  simple-moving-average crossover.  Values above 0.5 mean the short SMA is
  above the long SMA (bullish); values below 0.5 mean the opposite (bearish).

Both metrics are combined by ``composite_probability``, which returns the
weighted average of the two signals.  A value above 0.5 suggests the market is
more likely to move *up* (favour a long trade); below 0.5 favours a short.
"""

from __future__ import annotations

from typing import Sequence


class ProbabilityEstimator:
    """Estimates the probability of an upward price move.

    Parameters
    ----------
    prices:
        Ordered sequence of historical closing prices (oldest first).
    """

    def __init__(self, prices: Sequence[float]) -> None:
        if len(prices) < 2:
            raise ValueError("At least 2 price points are required.")
        self._prices = list(prices)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def win_rate(self, lookback: int = 14) -> float:
        """Fraction of up-closes in the last *lookback* bars.

        Returns a value in [0, 1].  0.5 means an equal number of up and
        down closes; >0.5 is bullish, <0.5 is bearish.
        """
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        window = self._prices[-(lookback + 1):]
        if len(window) < 2:
            return 0.5
        ups = sum(1 for a, b in zip(window, window[1:]) if b > a)
        return ups / (len(window) - 1)

    def trend_strength(self, short_window: int = 5, long_window: int = 20) -> float:
        """SMA-crossover-based probability of an upward move.

        Returns a value in (0, 1).  Values above 0.5 are bullish; values
        below 0.5 are bearish.

        The formula is a soft sigmoid:
            0.5 + 0.5 * (short_sma - long_sma) / long_sma
        clamped to [0, 1].
        """
        if short_window < 1 or long_window < 1:
            raise ValueError("Window sizes must be >= 1.")
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")

        prices = self._prices
        if len(prices) < long_window:
            # Not enough data – return neutral
            return 0.5

        short_sma = sum(prices[-short_window:]) / short_window
        long_sma = sum(prices[-long_window:]) / long_window

        if long_sma == 0:
            return 0.5

        raw = 0.5 + 0.5 * (short_sma - long_sma) / long_sma
        return max(0.0, min(1.0, raw))

    def composite_probability(
        self,
        lookback: int = 14,
        short_window: int = 5,
        long_window: int = 20,
        win_rate_weight: float = 0.5,
        trend_weight: float = 0.5,
    ) -> float:
        """Weighted combination of win-rate and trend-strength.

        Returns a value in [0, 1].  Values above 0.5 favour a long; below
        0.5 favour a short.
        """
        total = win_rate_weight + trend_weight
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        wr = self.win_rate(lookback)
        ts = self.trend_strength(short_window, long_window)
        return (win_rate_weight * wr + trend_weight * ts) / total
