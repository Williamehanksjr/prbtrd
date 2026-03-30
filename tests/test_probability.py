"""Unit tests for prbtrd.probability."""

import numpy as np
import pytest

from prbtrd.learner import Learner
from prbtrd.probability import (
    probability_from_learner,
    rolling_win_rate,
    trend_strength,
)


@pytest.fixture()
def fitted_learner():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] > 0).astype(int)
    return Learner().fit(X, y), X


class TestProbabilityFromLearner:
    def test_returns_array_of_correct_shape(self, fitted_learner):
        learner, X = fitted_learner
        probs = probability_from_learner(learner, X[:8])
        assert probs.shape == (8,)

    def test_values_in_unit_interval(self, fitted_learner):
        learner, X = fitted_learner
        probs = probability_from_learner(learner, X)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_raises_on_wrong_type(self, fitted_learner):
        _, X = fitted_learner
        with pytest.raises(TypeError):
            probability_from_learner("not-a-learner", X)

    def test_raises_if_learner_not_fitted(self, fitted_learner):
        _, X = fitted_learner
        with pytest.raises(RuntimeError, match="fitted"):
            probability_from_learner(Learner(), X)

    def test_accepts_list_input(self, fitted_learner):
        learner, X = fitted_learner
        probs = probability_from_learner(learner, X[:5].tolist())
        assert probs.shape == (5,)


class TestRollingWinRate:
    def test_basic(self):
        r = np.array([-1.0, 1.0, 1.0, -1.0, 1.0])
        result = rolling_win_rate(r, window=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert pytest.approx(result[2], abs=1e-9) == 2 / 3
        # window at i=3: [1, 1, -1] → 2 wins → 2/3
        assert pytest.approx(result[3], abs=1e-9) == 2 / 3
        # window at i=4: [1, -1, 1] → 2 wins → 2/3
        assert pytest.approx(result[4], abs=1e-9) == 2 / 3

    def test_all_wins(self):
        r = np.ones(10)
        result = rolling_win_rate(r, window=5)
        assert np.all(np.isnan(result[:4]))
        assert np.allclose(result[4:], 1.0)

    def test_all_losses(self):
        r = -np.ones(10)
        result = rolling_win_rate(r, window=5)
        assert np.allclose(result[4:], 0.0)

    def test_window_one(self):
        r = np.array([1.0, -1.0, 1.0])
        result = rolling_win_rate(r, window=1)
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])

    def test_raises_on_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            rolling_win_rate(np.ones((3, 3)), window=2)

    def test_raises_on_invalid_window(self):
        with pytest.raises(ValueError, match="window"):
            rolling_win_rate(np.ones(5), window=0)

    def test_length_preserved(self):
        r = np.random.default_rng(99).standard_normal(50)
        result = rolling_win_rate(r, window=10)
        assert len(result) == len(r)


class TestTrendStrength:
    def test_perfect_uptrend(self):
        prices = np.arange(1.0, 11.0)  # perfectly linear uptrend
        ts = trend_strength(prices, window=5)
        assert np.all(np.isnan(ts[:4]))
        assert np.all(ts[4:] > 0)

    def test_flat_market(self):
        prices = np.ones(10)
        ts = trend_strength(prices, window=3)
        # slope = 0 for constant prices
        assert np.allclose(ts[2:], 0.0)

    def test_values_bounded(self):
        rng = np.random.default_rng(5)
        prices = np.exp(rng.standard_normal(100).cumsum())
        ts = trend_strength(prices, window=10)
        valid = ts[~np.isnan(ts)]
        assert np.all(valid >= 0) and np.all(valid <= 1)

    def test_raises_on_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            trend_strength(np.ones((3, 3)), window=2)

    def test_raises_on_window_less_than_2(self):
        with pytest.raises(ValueError, match="window"):
            trend_strength(np.ones(5), window=1)

    def test_length_preserved(self):
        prices = np.random.default_rng(1).standard_normal(60).cumsum()
        ts = trend_strength(prices, window=15)
        assert len(ts) == len(prices)
