#!/usr/bin/env python3
"""CLI entry point for prbtrd.

Usage
-----
Provide closing prices as space-separated arguments::

    python main.py 100 101 102 103 102 104 105 106 105 107 108 107 109 110 111 112 111 113 114 115

Optional flags::

    --long-threshold  FLOAT   Probability threshold for LONG  (default 0.55)
    --short-threshold FLOAT   Probability threshold for SHORT (default 0.45)
    --lookback        INT     Win-rate lookback bars          (default 14)
    --short-window    INT     Short SMA window                (default 5)
    --long-window     INT     Long SMA window                 (default 20)
"""

from __future__ import annotations

import argparse
import sys

from prbtrd import Trader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prbtrd",
        description="Trade long or short based on probability.",
    )
    parser.add_argument(
        "prices",
        nargs="+",
        type=float,
        metavar="PRICE",
        help="Historical closing prices (oldest first).",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.55,
        metavar="FLOAT",
        help="Probability threshold above which a LONG signal is issued (default: 0.55).",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.45,
        metavar="FLOAT",
        help="Probability threshold below which a SHORT signal is issued (default: 0.45).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=14,
        metavar="INT",
        help="Number of bars used for the win-rate calculation (default: 14).",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=5,
        metavar="INT",
        help="Short SMA window for trend-strength (default: 5).",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=20,
        metavar="INT",
        help="Long SMA window for trend-strength (default: 20).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        trader = Trader(
            prices=args.prices,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            lookback=args.lookback,
            short_window=args.short_window,
            long_window=args.long_window,
        )
        result = trader.evaluate()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
