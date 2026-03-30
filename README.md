# prbtrd
probability trader

`prbtrd` decides whether to go **LONG**, **SHORT**, or stay **NEUTRAL** by
computing a composite probability of upward price movement from historical
closing prices.

## How it works

Two signals are blended into a single composite probability (0–1):

| Signal | Formula | Bullish when |
|---|---|---|
| **Win-rate** | fraction of up-closes in the last N bars | > 0.5 |
| **Trend-strength** | soft-normalised SMA crossover (short vs long) | > 0.5 |

The composite probability is the weighted average of the two signals
(equal weight by default).

* `probability > long_threshold (default 0.55)` → **LONG**
* `probability < short_threshold (default 0.45)` → **SHORT**
* otherwise → **NEUTRAL**

## Quick start

```bash
# bullish series
python main.py 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
# → Signal: LONG | Probability: 0.6833 | …

# bearish series
python main.py 120 119 118 117 116 115 114 113 112 111 110 109 108 107 106 105 104 103 102 101 100
# → Signal: SHORT | Probability: 0.3167 | …
```

### Options

```
positional arguments:
  PRICE                 Historical closing prices (oldest first)

options:
  --long-threshold FLOAT   Probability > this → LONG  (default: 0.55)
  --short-threshold FLOAT  Probability < this → SHORT (default: 0.45)
  --lookback INT           Win-rate lookback bars      (default: 14)
  --short-window INT       Short SMA window            (default: 5)
  --long-window INT        Long SMA window             (default: 20)
```

## Python API

```python
from prbtrd import Trader, Signal

prices = [100, 101, 102, 101, 103, 104, 105, 106, 105, 107,
          108, 107, 109, 110, 111, 112, 111, 113, 114, 115]

trader = Trader(prices)
result = trader.evaluate()

print(result.signal)      # Signal.LONG
print(result.probability) # e.g. 0.6564
```

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
