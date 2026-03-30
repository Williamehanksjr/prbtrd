"""ML learner that wraps a scikit-learn classifier to produce probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin


class Learner:
    """Wraps a scikit-learn classifier and exposes ``get_probability``.

    Parameters
    ----------
    clf:
        Any scikit-learn classifier that supports ``predict_proba``.
        Defaults to :class:`~sklearn.linear_model.LogisticRegression`.
    """

    def __init__(self, clf: ClassifierMixin | None = None) -> None:
        self._clf: ClassifierMixin = clf if clf is not None else LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Learner":
        """Fit the underlying classifier.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Binary labels of shape ``(n_samples,)``.

        Returns
        -------
        Learner
            The fitted learner (``self``), to allow chaining.
        """
        self._clf.fit(X, y)
        return self

    def get_probability(self, X: np.ndarray) -> np.ndarray:
        """Return the probability of the positive class (class 1) for each sample.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples,)`` with values in ``[0, 1]``.
        """
        proba = self._clf.predict_proba(X)
        return proba[:, 1]
