"""Tests for prbtrd.probability."""

import numpy as np
import pandas as pd
import pytest

from prbtrd.probability import compute_probability, make_features, make_labels


def _sample_prices(n: int = 60, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, dtype=float)


# ---------------------------------------------------------------------------
# make_features
# ---------------------------------------------------------------------------

class TestMakeFeatures:
    def test_column_names(self):
        features = make_features(_sample_prices(), window=5)
        assert list(features.columns) == ["return", "roll_mean", "roll_std", "momentum"]

    def test_row_count(self):
        prices = _sample_prices(n=60)
        features = make_features(prices, window=5)
        # first `window` rows are dropped by dropna
        assert len(features) == len(prices) - 5

    def test_no_nan(self):
        features = make_features(_sample_prices(), window=5)
        assert not features.isna().any().any()


# ---------------------------------------------------------------------------
# make_labels
# ---------------------------------------------------------------------------

class TestMakeLabels:
    def test_binary_values(self):
        labels = make_labels(_sample_prices(), window=5)
        assert set(labels.unique()).issubset({0, 1})

    def test_row_count(self):
        prices = _sample_prices(n=60)
        labels = make_labels(prices, window=5)
        # length = N - window - 1
        assert len(labels) == len(prices) - 5 - 1

    def test_features_labels_alignment(self):
        """Training lengths must match."""
        prices = _sample_prices(n=60)
        features = make_features(prices, window=5)
        labels = make_labels(prices, window=5)
        # features[:-1] rows should equal labels rows
        assert len(features) - 1 == len(labels)


# ---------------------------------------------------------------------------
# compute_probability
# ---------------------------------------------------------------------------

class TestComputeProbability:
    def test_returns_float(self):
        prob = compute_probability(_sample_prices())
        assert isinstance(prob, float)

    def test_in_unit_interval(self):
        prob = compute_probability(_sample_prices())
        assert 0.0 <= prob <= 1.0

    def test_accepts_learner_argument(self):
        from prbtrd.learner import Learner

        learner = Learner()
        prob = compute_probability(_sample_prices(), learner=learner)
        assert 0.0 <= prob <= 1.0

    def test_custom_window(self):
        prob = compute_probability(_sample_prices(n=80), window=10)
        assert 0.0 <= prob <= 1.0

    def test_single_class_labels_returns_valid_probability(self):
        """A monotone price series produces only one label class; must not raise."""
        prices = pd.Series(np.arange(1.0, 21.0))  # strictly increasing → all labels 1
        prob = compute_probability(prices)
        assert 0.0 <= prob <= 1.0
