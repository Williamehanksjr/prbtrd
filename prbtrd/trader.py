"""Probability-based trade signal generator.

``Trader`` wraps a ``ProbabilityEstimator`` and converts the composite
probability into a discrete trading signal:

* **LONG**  – composite probability  > ``long_threshold``  (default 0.55)
* **SHORT** – composite probability  < ``short_threshold`` (default 0.45)
* **NEUTRAL** – everything in between (no strong edge detected)

Usage example::

    from prbtrd import Trader

    prices = [100, 101, 102, 101, 103, 104, 105, 106, 105, 107,
              108, 107, 109, 110, 111, 112, 111, 113, 114, 115]
    trader = Trader(prices)
    result = trader.evaluate()
    print(result.signal)       # Signal.LONG / Signal.SHORT / Signal.NEUTRAL
    print(result.probability)  # e.g. 0.623
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .probability import ProbabilityEstimator


class Signal(Enum):
    """Discrete trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class TradeResult:
    """The outcome of a single ``Trader.evaluate()`` call.

    Attributes
    ----------
    signal:
        The recommended trade direction.
    probability:
        The composite probability value that drove the decision (0–1).
    long_threshold:
        The threshold above which a LONG signal is generated.
    short_threshold:
        The threshold below which a SHORT signal is generated.
    """

    signal: Signal
    probability: float
    long_threshold: float
    short_threshold: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Signal: {self.signal.value} | "
            f"Probability: {self.probability:.4f} | "
            f"Thresholds: LONG>{self.long_threshold} / SHORT<{self.short_threshold}"
        )


class Trader:
    """Generates a long/short/neutral signal from historical price data.

    Parameters
    ----------
    prices:
        Ordered sequence of historical closing prices (oldest first).
    long_threshold:
        Composite probability must exceed this value to generate a LONG
        signal.  Default is ``0.55``.
    short_threshold:
        Composite probability must fall below this value to generate a
        SHORT signal.  Default is ``0.45``.
    lookback:
        Number of bars used for the win-rate calculation.
    short_window:
        Short SMA window for the trend-strength metric.
    long_window:
        Long SMA window for the trend-strength metric.
    win_rate_weight:
        Weight assigned to the win-rate component of the composite score.
    trend_weight:
        Weight assigned to the trend-strength component of the composite
        score.
    """

    def __init__(
        self,
        prices: Sequence[float],
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        lookback: int = 14,
        short_window: int = 5,
        long_window: int = 20,
        win_rate_weight: float = 0.5,
        trend_weight: float = 0.5,
    ) -> None:
        if long_threshold <= short_threshold:
            raise ValueError(
                "long_threshold must be greater than short_threshold."
            )
        if not (0 < short_threshold < long_threshold < 1):
            raise ValueError(
                "Thresholds must satisfy 0 < short_threshold < long_threshold < 1."
            )

        self._estimator = ProbabilityEstimator(prices)
        self._long_threshold = long_threshold
        self._short_threshold = short_threshold
        self._lookback = lookback
        self._short_window = short_window
        self._long_window = long_window
        self._win_rate_weight = win_rate_weight
        self._trend_weight = trend_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> TradeResult:
        """Compute the probability and return the corresponding signal."""
        prob = self._estimator.composite_probability(
            lookback=self._lookback,
            short_window=self._short_window,
            long_window=self._long_window,
            win_rate_weight=self._win_rate_weight,
            trend_weight=self._trend_weight,
        )

        if prob > self._long_threshold:
            signal = Signal.LONG
        elif prob < self._short_threshold:
            signal = Signal.SHORT
        else:
            signal = Signal.NEUTRAL

        return TradeResult(
            signal=signal,
            probability=prob,
            long_threshold=self._long_threshold,
            short_threshold=self._short_threshold,
        )
