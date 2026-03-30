"""CLI entry point for prbtrd."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from prbtrd.trader import trade


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="prbtrd",
        description="Probability-based trader — prints LONG, SHORT, or NEUTRAL.",
    )
    parser.add_argument(
        "prices",
        nargs="+",
        type=float,
        metavar="PRICE",
        help="Chronological price series (at least 8 values recommended).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        metavar="N",
        help="Look-back window for rolling features (default: 5).",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.6,
        metavar="P",
        help="Probability threshold above which a LONG signal is emitted (default: 0.6).",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.4,
        metavar="P",
        help="Probability threshold below which a SHORT signal is emitted (default: 0.4).",
    )

    args = parser.parse_args(argv)
    prices = pd.Series(args.prices, dtype=float)

    signal = trade(
        prices,
        window=args.window,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )
    print(signal)


if __name__ == "__main__":
    main()
