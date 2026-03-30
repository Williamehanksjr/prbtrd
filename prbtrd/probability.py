"""Probability helpers for the prbtrd trading system.

This module provides:

* :func:`probability_from_learner` — derive win probability from a fitted
  :class:`~prbtrd.learner.Learner`.
* :func:`rolling_win_rate` — simple rolling win-rate over a returns series.
* :func:`trend_strength` — normalised trend-strength metric in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

from prbtrd.learner import Learner


def probability_from_learner(learner: Learner, X: np.ndarray) -> np.ndarray:
    """Return win probabilities predicted by a fitted *learner*.

    This is the primary entry-point for obtaining model-based trade
    probabilities.  The learner must have been trained with
    :meth:`~prbtrd.learner.Learner.fit` before calling this function.

    Parameters
    ----------
    learner:
        A fitted :class:`~prbtrd.learner.Learner` instance.
    X:
        2-D feature array of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        1-D array of win probabilities, one per sample, in ``[0, 1]``.

    Raises
    ------
    TypeError
        If *learner* is not a :class:`~prbtrd.learner.Learner` instance.
    RuntimeError
        If the learner has not been fitted yet.

    Examples
    --------
    >>> import numpy as np
    >>> from prbtrd.learner import Learner
    >>> from prbtrd.probability import probability_from_learner
    >>> rng = np.random.default_rng(1)
    >>> X_train = rng.standard_normal((80, 3))
    >>> y_train = (X_train[:, 0] > 0).astype(int)
    >>> learner = Learner()
    >>> learner.fit(X_train, y_train)
    >>> X_new = rng.standard_normal((5, 3))
    >>> probs = probability_from_learner(learner, X_new)
    >>> probs.shape
    (5,)
    >>> ((probs >= 0) & (probs <= 1)).all()
    True
    """
    if not isinstance(learner, Learner):
        raise TypeError(f"learner must be a Learner instance, got {type(learner)}")
    return learner.predict_proba(np.asarray(X, dtype=float))


def rolling_win_rate(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling win rate over a 1-D array of trade returns.

    The win rate at position *i* is the fraction of the preceding *window*
    returns (inclusive) that are strictly positive.

    Parameters
    ----------
    returns:
        1-D array of trade P&L or percentage returns.
    window:
        Look-back window size (number of bars).  Must be >= 1.

    Returns
    -------
    np.ndarray
        1-D array of the same length as *returns*, with ``NaN`` for positions
        where fewer than *window* observations are available.

    Examples
    --------
    >>> import numpy as np
    >>> from prbtrd.probability import rolling_win_rate
    >>> r = np.array([-1.0, 1.0, 1.0, -1.0, 1.0])
    >>> rolling_win_rate(r, window=3)
    array([       nan,        nan, 0.66666667, 0.33333333, 0.33333333])
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {returns.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = len(returns)
    win_rate = np.full(n, np.nan)
    wins = (returns > 0).astype(float)
    for i in range(window - 1, n):
        win_rate[i] = wins[i - window + 1 : i + 1].mean()
    return win_rate


def trend_strength(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute a normalised trend-strength metric in ``[0, 1]``.

    Trend strength is defined as the absolute value of the linear regression
    slope over the *window* most recent prices, normalised by the mean price
    in that window so it is scale-independent.

    A value close to **1** indicates a strong directional move; a value close
    to **0** indicates a flat or choppy market.

    Parameters
    ----------
    prices:
        1-D array of asset prices (e.g. close prices).
    window:
        Look-back window size.  Must be >= 2.

    Returns
    -------
    np.ndarray
        1-D array of the same length as *prices*, with ``NaN`` for positions
        where fewer than *window* observations are available.

    Examples
    --------
    >>> import numpy as np
    >>> from prbtrd.probability import trend_strength
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> ts = trend_strength(prices, window=3)
    >>> np.isnan(ts[:2]).all()
    True
    >>> ts[4] > 0
    True
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {prices.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    n = len(prices)
    strength = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, n):
        y = prices[i - window + 1 : i + 1]
        y_mean = y.mean()
        if y_mean == 0:
            strength[i] = 0.0
            continue
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        # normalise by mean price so the metric is dimensionless
        raw = abs(slope) / y_mean
        # cap at 1 for interpretability
        strength[i] = min(raw, 1.0)
    return strength
