"""Unit tests for prbtrd.learner."""

import numpy as np
import pytest

from prbtrd.learner import Learner


@pytest.fixture()
def simple_dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((120, 4))
    # Label is 1 if first feature is positive, else 0
    y = (X[:, 0] > 0).astype(int)
    return X, y


class TestLearnerFit:
    def test_fit_returns_self(self, simple_dataset):
        X, y = simple_dataset
        learner = Learner()
        result = learner.fit(X, y)
        assert result is learner

    def test_fitted_flag_set_after_fit(self, simple_dataset):
        X, y = simple_dataset
        learner = Learner()
        assert not learner._fitted
        learner.fit(X, y)
        assert learner._fitted

    def test_fit_raises_on_1d_X(self, simple_dataset):
        _, y = simple_dataset
        learner = Learner()
        with pytest.raises(ValueError, match="2-D"):
            learner.fit(np.ones(10), y[:10])

    def test_fit_raises_on_mismatched_lengths(self, simple_dataset):
        X, y = simple_dataset
        learner = Learner()
        with pytest.raises(ValueError):
            learner.fit(X, y[:10])


class TestLearnerPredictProba:
    def test_output_shape(self, simple_dataset):
        X, y = simple_dataset
        learner = Learner().fit(X, y)
        probs = learner.predict_proba(X[:10])
        assert probs.shape == (10,)

    def test_output_in_unit_interval(self, simple_dataset):
        X, y = simple_dataset
        learner = Learner().fit(X, y)
        probs = learner.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_raises_before_fit(self, simple_dataset):
        X, _ = simple_dataset
        learner = Learner()
        with pytest.raises(RuntimeError, match="fitted"):
            learner.predict_proba(X)

    def test_raises_on_1d_X(self, simple_dataset):
        X, y = simple_dataset
        learner = Learner().fit(X, y)
        with pytest.raises(ValueError, match="2-D"):
            learner.predict_proba(X[0])  # 1-D row

    def test_probabilities_reflect_signal(self):
        """A perfectly linearly separable dataset should yield high accuracy."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        y = (X[:, 0] > 0).astype(int)
        learner = Learner().fit(X, y)
        probs = learner.predict_proba(X)
        predicted = (probs >= 0.5).astype(int)
        accuracy = (predicted == y).mean()
        assert accuracy > 0.85, f"Accuracy too low: {accuracy:.2f}"
