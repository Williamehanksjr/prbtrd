"""Learner: fits on labelled price-bar data and estimates win probability."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Learner:
    """A lightweight logistic-regression learner for win-probability estimation.

    The learner expects each sample to be a 1-D feature vector derived from
    market data (e.g. rolling returns, momentum, volatility) and a binary
    label indicating whether the subsequent trade was profitable (1) or not
    (0).

    Parameters
    ----------
    random_state:
        Seed forwarded to :class:`~sklearn.linear_model.LogisticRegression`
        for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from prbtrd.learner import Learner
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] + rng.standard_normal(100) > 0).astype(int)
    >>> learner = Learner()
    >>> learner.fit(X, y)
    >>> probs = learner.predict_proba(X[:5])
    >>> probs.shape
    (5,)
    """

    def __init__(self, random_state: int = 42) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(random_state=random_state, max_iter=1000)
        self._fitted = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Learner":
        """Fit the learner on feature matrix *X* and binary labels *y*.

        Parameters
        ----------
        X:
            2-D array of shape ``(n_samples, n_features)``.
        y:
            1-D integer array of shape ``(n_samples,)`` with values 0 or 1.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1-D array with the same length as X")
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probability of a *winning* trade for each sample in *X*.

        Parameters
        ----------
        X:
            2-D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)`` with values in ``[0, 1]``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        X_scaled = self._scaler.transform(X)
        # column index 1 corresponds to the positive class (win)
        classes = list(self._model.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        proba = self._model.predict_proba(X_scaled)
        if pos_idx == -1:
            # model only saw one class — return zeros or ones accordingly
            return np.zeros(len(X)) if classes[0] == 0 else np.ones(len(X))
        return proba[:, pos_idx]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Learner has not been fitted yet. Call fit() first.")
