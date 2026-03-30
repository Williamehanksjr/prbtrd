"""Trader that decides LONG, SHORT, or NEUTRAL based on learner probability."""

from __future__ import annotations

import pandas as pd

from .learner import Learner
from .probability import compute_probability

LONG = "LONG"
SHORT = "SHORT"
NEUTRAL = "NEUTRAL"


def decide(
    probability: float,
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
) -> str:
    """Map a probability to a trading signal.

    Parameters
    ----------
    probability:
        Estimated probability that the next price movement is positive.
    long_threshold:
        Probabilities *strictly above* this value produce a ``LONG`` signal.
    short_threshold:
        Probabilities *strictly below* this value produce a ``SHORT`` signal.

    Returns
    -------
    str
        One of :data:`LONG`, :data:`SHORT`, or :data:`NEUTRAL`.
    """
    if probability > long_threshold:
        return LONG
    if probability < short_threshold:
        return SHORT
    return NEUTRAL


def trade(
    prices: pd.Series,
    learner: Learner | None = None,
    window: int = 5,
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
) -> str:
    """Compute the trading signal for the latest bar in *prices*.

    Uses a :class:`~prbtrd.learner.Learner` to estimate the probability
    that the next price movement is positive, then passes that probability
    to :func:`decide`.

    Parameters
    ----------
    prices:
        Chronological price series.
    learner:
        Pre-configured :class:`~prbtrd.learner.Learner`.  A default
        ``Learner()`` is created when ``None`` is passed.
    window:
        Look-back window passed to feature engineering.
    long_threshold:
        Passed to :func:`decide`.
    short_threshold:
        Passed to :func:`decide`.

    Returns
    -------
    str
        One of :data:`LONG`, :data:`SHORT`, or :data:`NEUTRAL`.
    """
    probability = compute_probability(prices, learner=learner, window=window)
    return decide(probability, long_threshold=long_threshold, short_threshold=short_threshold)
