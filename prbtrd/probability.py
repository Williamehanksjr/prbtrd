"""Feature engineering and probability computation using a learner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .learner import Learner


def make_features(prices: pd.Series, window: int = 5) -> pd.DataFrame:
    """Build a feature matrix from a price series.

    Features for each bar:

    * ``return``    – single-bar percentage return
    * ``roll_mean`` – rolling mean of returns over *window* bars
    * ``roll_std``  – rolling standard deviation of returns over *window* bars
    * ``momentum``  – price change over *window* bars (``price(t) / price(t-window) - 1``)

    The first *window* rows are dropped because the rolling statistics are
    undefined there.

    Parameters
    ----------
    prices:
        Chronological price series.
    window:
        Look-back period for rolling statistics.

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns ``["return", "roll_mean", "roll_std", "momentum"]``.
    """
    df = pd.DataFrame({"price": prices.values}, index=prices.index)
    df["return"] = df["price"].pct_change()
    df["roll_mean"] = df["return"].rolling(window).mean()
    df["roll_std"] = df["return"].rolling(window).std()
    df["momentum"] = df["price"] / df["price"].shift(window) - 1
    df = df.dropna()
    return df[["return", "roll_mean", "roll_std", "momentum"]]


def make_labels(prices: pd.Series, window: int = 5) -> pd.Series:
    """Create binary labels aligned with :func:`make_features`.

    Label *i* is ``1`` if the return at bar *i + 1* is positive, ``0``
    otherwise.  The first *window* bars and the final bar (whose future
    return is unknown) are excluded so that labels align exactly with the
    training rows of :func:`make_features`.

    Parameters
    ----------
    prices:
        Chronological price series (same series passed to :func:`make_features`).
    window:
        Must match the *window* used in :func:`make_features`.

    Returns
    -------
    pd.Series
        Binary labels of length ``len(prices) - window - 1``.
    """
    future_returns = prices.pct_change().shift(-1)
    labels = (future_returns > 0).astype(int)
    # Drop the first `window` rows (no rolling stats) and the last row
    # (future return is unknown / NaN-derived).
    labels = labels.iloc[window:-1]
    return labels


def compute_probability(
    prices: pd.Series,
    learner: Learner | None = None,
    window: int = 5,
) -> float:
    """Fit *learner* on historical data and return the probability for the
    current (last) bar.

    The learner is trained on all bars except the final one, then used to
    predict the probability that the *next* return will be positive for the
    last bar.

    Parameters
    ----------
    prices:
        Chronological price series.  Must have at least ``window + 2``
        data points so that there is at least one training sample.
    learner:
        A :class:`~prbtrd.learner.Learner` instance.  A fresh
        ``Learner()`` is created when ``None`` is passed.
    window:
        Look-back window for feature engineering.

    Returns
    -------
    float
        Probability in ``[0, 1]`` that the next price movement is positive.
    """
    if learner is None:
        learner = Learner()

    features = make_features(prices, window)  # rows: window .. N-1
    labels = make_labels(prices, window)       # rows: window .. N-2

    # Training: features at window..N-2, labels at window..N-2
    X_train = features.iloc[:-1].values
    y_train = labels.values
    # Current bar (the one we want to predict)
    X_pred = features.iloc[-1:].values

    # Guard: align lengths in case of rounding edge-cases
    n = min(len(X_train), len(y_train))
    X_train = X_train[:n]
    y_train = y_train[:n]

    # If all labels are the same class the learner cannot be trained;
    # return the class mean as a degenerate probability.
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return float(unique_classes[0])

    learner.fit(X_train, y_train)
    prob = learner.get_probability(X_pred)
    return float(prob[0])
