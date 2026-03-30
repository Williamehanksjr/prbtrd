"""Tests for prbtrd.learner."""

import numpy as np
import pytest

from prbtrd.learner import Learner


def _make_linearly_separable(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_fit_returns_self():
    X, y = _make_linearly_separable()
    learner = Learner()
    result = learner.fit(X, y)
    assert result is learner


def test_get_probability_shape():
    X, y = _make_linearly_separable()
    learner = Learner().fit(X, y)
    proba = learner.get_probability(X[:1])
    assert proba.shape == (1,)


def test_get_probability_bounds():
    X, y = _make_linearly_separable()
    learner = Learner().fit(X, y)
    proba = learner.get_probability(X)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_get_probability_is_float_array():
    X, y = _make_linearly_separable()
    learner = Learner().fit(X, y)
    proba = learner.get_probability(X[:5])
    assert proba.dtype.kind == "f"


def test_custom_classifier_is_used():
    """Learner should accept an alternative scikit-learn classifier."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = _make_linearly_separable(n=60)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    learner = Learner(clf=clf)
    learner.fit(X, y)
    proba = learner.get_probability(X[:3])
    assert proba.shape == (3,)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
