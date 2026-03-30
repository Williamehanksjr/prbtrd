"""Tests for the CLI entry point (main.py)."""

import sys
import pytest

# Ensure the repo root is on the path so ``import main`` works
sys.path.insert(0, ".")

from main import main  # noqa: E402  (import after sys.path mutation)


PRICES_BULLISH = [str(i) for i in range(1, 26)]
PRICES_BEARISH = [str(i) for i in range(25, 0, -1)]


def test_cli_bullish_exits_zero(capsys):
    rc = main(PRICES_BULLISH)
    assert rc == 0
    out = capsys.readouterr().out
    assert "LONG" in out


def test_cli_bearish_exits_zero(capsys):
    rc = main(PRICES_BEARISH)
    assert rc == 0
    out = capsys.readouterr().out
    assert "SHORT" in out


def test_cli_shows_probability(capsys):
    main(PRICES_BULLISH)
    out = capsys.readouterr().out
    assert "Probability" in out


def test_cli_custom_thresholds(capsys):
    rc = main(PRICES_BULLISH + ["--long-threshold", "0.6", "--short-threshold", "0.4"])
    assert rc == 0


def test_cli_invalid_threshold_exits_nonzero(capsys):
    rc = main(PRICES_BULLISH + ["--long-threshold", "0.4", "--short-threshold", "0.6"])
    assert rc != 0


def test_cli_too_few_prices_exits_nonzero(capsys):
    rc = main(["100"])
    assert rc != 0
