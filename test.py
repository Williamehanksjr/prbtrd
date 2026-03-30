#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from online_learner import OnlineLearner

st.set_page_config(page_title="NasdaqTrader", layout="wide")

LOOKBACK_BARS = 600
HORIZON_OPTIONS = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]
MODEL_SPEED_OPTIONS = ["Balanced", "Fast", "Very Fast"]


@dataclass
class ProbSnapshot:
    positive: float
    negative: float
    edge: float
    long_probability: float
    short_probability: float
    direction: str
    state: str
    horizon_minutes: int
    horizon_bars: int
    quantity: float


def _interval_minutes(interval_label: str) -> int:
    mapping = {
        "1m": 1,
        "2m": 2,
        "3m": 3,
        "4m": 4,
        "5m": 5,
        "10m": 10,
        "15m": 15,
        "20m": 20,
        "30m": 30,
        "40m": 40,
        "50m": 50,
        "60m": 60,
        "1h": 60,
        "1d": 1440,
    }
    return mapping.get(interval_label, 15)


@st.cache_data(ttl=5)
def load_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()]
    return df.tail(LOOKBACK_BARS)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)

    out["close"] = close
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out["rsi"] = (100 - 100 / (1 + rs)).fillna(50.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def compute_probability(
    df: pd.DataFrame,
    interval_label: str,
    model_speed: str = "Balanced",
    horizon_minutes: int = 20,
) -> ProbSnapshot:
    if df.empty:
        return ProbSnapshot(0.0, 0.0, 0.0, 50.0, 50.0, "NEUTRAL", "UNTRADABLE", horizon_minutes, 1, 0.0)

    ind = compute_indicators(df)
    close = ind["close"]
    vol = df["Volume"].astype(float).fillna(0.0) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    ret = close.pct_change().fillna(0.0)

    bar_minutes = _interval_minutes(interval_label)
    horizon_bars = max(1, int(round(horizon_minutes / max(1, bar_minutes))))
    min_bars_required = max(40, horizon_bars * 15)
    if len(df) < min_bars_required:
        return ProbSnapshot(0.0, 0.0, 0.0, 50.0, 50.0, "NEUTRAL", "UNTRADABLE", horizon_minutes, horizon_bars, 0.0)

    bar_m = float(bar_minutes)

    prof_1m = (8.0, 21.0, 55.0, 8.0, 21.0, 5.0, 7.0, 11.0)
    prof_5m = (13.0, 34.0, 89.0, 10.0, 24.0, 6.0, 10.0, 14.0)
    prof_15m = (21.0, 50.0, 200.0, 12.0, 26.0, 9.0, 14.0, 18.0)

    def _lerp_profile(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, ...]:
        tt = min(1.0, max(0.0, t))
        return tuple(ai + (bi - ai) * tt for ai, bi in zip(a, b))

    if bar_m <= 5.0:
        t = (bar_m - 1.0) / 4.0
        ef, em, es, macd_fast, macd_slow, macd_sig, rsi_len, edge_scale = _lerp_profile(prof_1m, prof_5m, t)
    elif bar_m <= 15.0:
        t = (bar_m - 5.0) / 10.0
        ef, em, es, macd_fast, macd_slow, macd_sig, rsi_len, edge_scale = _lerp_profile(prof_5m, prof_15m, t)
    else:
        ef, em, es, macd_fast, macd_slow, macd_sig, rsi_len, edge_scale = prof_15m

    speed_cfg = {
        "Balanced": {"span_mult": 1.00, "lookback_mult": 1.00, "edge_mult": 1.00},
        "Fast": {"span_mult": 0.78, "lookback_mult": 0.75, "edge_mult": 0.82},
        "Very Fast": {"span_mult": 0.58, "lookback_mult": 0.55, "edge_mult": 0.66},
    }.get(model_speed, {"span_mult": 1.00, "lookback_mult": 1.00, "edge_mult": 1.00})

    span_mult = float(speed_cfg["span_mult"])
    look_mult = float(speed_cfg["lookback_mult"])
    edge_scale = max(4.0, edge_scale * float(speed_cfg["edge_mult"]))

    ef = max(3, int(round(ef * span_mult)))
    em = max(ef + 2, int(round(em * span_mult)))
    es = max(em + 5, int(round(es * span_mult)))
    macd_fast = max(3, int(round(macd_fast * span_mult)))
    macd_slow = max(macd_fast + 2, int(round(macd_slow * span_mult)))
    macd_sig = max(3, int(round(macd_sig * span_mult)))
    rsi_len = max(4, int(round(rsi_len * span_mult)))

    ema_fast_s = close.ewm(span=ef, adjust=False).mean()
    ema_mid_s = close.ewm(span=em, adjust=False).mean()
    ema_slow_s = close.ewm(span=es, adjust=False).mean()
    ema21 = float(ema_fast_s.iloc[-1])
    ema50 = float(ema_mid_s.iloc[-1])
    ema200 = float(ema_slow_s.iloc[-1])

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = float((100 - 100 / (1 + rs)).fillna(50.0).iloc[-1])

    ema_fast_macd = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow_macd = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast_macd - ema_slow_macd
    macd_signal = macd_line.ewm(span=macd_sig, adjust=False).mean()
    macd_hist = float((macd_line - macd_signal).iloc[-1])

    trend_fast = (ema21 - ema50) / close.iloc[-1] * 100
    trend_slow = (ema50 - ema200) / close.iloc[-1] * 100

    fast_lookback = max(2, int(round(max(2, horizon_bars) * look_mult)))
    slow_lookback = max(fast_lookback + 2, int(round(max(8, horizon_bars * 2) * look_mult)))
    momentum_fast = (close.iloc[-1] / close.iloc[-(fast_lookback + 1)] - 1) * 100 if len(close) > fast_lookback else 0.0
    momentum_slow = (close.iloc[-1] / close.iloc[-(slow_lookback + 1)] - 1) * 100 if len(close) > slow_lookback else 0.0

    vol_short_lb = max(6, int(round(max(6, horizon_bars * 3) * look_mult)))
    vol_long_lb = max(vol_short_lb + 4, int(round(max(24, horizon_bars * 8) * look_mult)))
    vol_short = ret.tail(vol_short_lb).std() * 100
    vol_long = ret.tail(vol_long_lb).std() * 100 if len(ret) >= vol_long_lb else ret.std() * 100
    vol_ratio = (vol_short / vol_long) if vol_long and vol_long > 1e-9 else 1.0

    vol_ma = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else vol.mean()
    volume_ratio = (vol.iloc[-1] / vol_ma) if vol_ma and vol_ma > 1e-9 else 1.0

    trend_long_ok = ema21 > ema50 > ema200
    trend_short_ok = ema21 < ema50 < ema200

    positive = (
        25.0
        + 50.0 * math.tanh(max(0.0, trend_fast) / 0.12)
        + 45.0 * math.tanh(max(0.0, trend_slow) / 0.18)
        + 35.0 * math.tanh(max(0.0, momentum_fast) / 0.30)
        + 30.0 * math.tanh(max(0.0, momentum_slow) / 0.70)
        + 20.0 * math.tanh(max(0.0, (rsi - 50.0)) / 12.0)
        + 15.0 * math.tanh(max(0.0, macd_hist) / 0.10)
        + 15.0 * math.tanh(max(0.0, volume_ratio - 1.0))
        + (8.0 if trend_long_ok else 0.0)
    )
    negative = (
        25.0
        + 50.0 * math.tanh(max(0.0, -trend_fast) / 0.12)
        + 45.0 * math.tanh(max(0.0, -trend_slow) / 0.18)
        + 35.0 * math.tanh(max(0.0, -momentum_fast) / 0.30)
        + 30.0 * math.tanh(max(0.0, -momentum_slow) / 0.70)
        + 20.0 * math.tanh(max(0.0, (50.0 - rsi)) / 12.0)
        + 15.0 * math.tanh(max(0.0, -macd_hist) / 0.10)
        + 15.0 * math.tanh(max(0.0, 1.0 - volume_ratio))
        + (8.0 if trend_short_ok else 0.0)
    )

    edge = max(-100.0, min(100.0, positive - negative))
    long_probability = 100.0 / (1.0 + math.exp(-edge / edge_scale))
    direction = "LONG" if edge >= 0 else "SHORT"

    tradable = (
        abs(edge) >= 12
        and 0.70 <= vol_ratio <= 2.10
        and volume_ratio >= 0.75
        and 38 <= rsi <= 72
    )
    untradable = (
        abs(edge) < 3.5
        or vol_ratio < 0.35
        or vol_ratio > 2.80
        or volume_ratio < 0.45
        or rsi < 20
        or rsi > 80
    )

    if tradable:
        state = "TRADABLE"
    elif untradable:
        state = "UNTRADABLE"
    else:
        state = "NEUTRAL"

    return ProbSnapshot(
        positive=max(0.0, min(100.0, positive)),
        negative=max(0.0, min(100.0, negative)),
        edge=edge,
        long_probability=long_probability,
        short_probability=100.0 - long_probability,
        direction=direction,
        state=state,
        horizon_minutes=horizon_minutes,
        horizon_bars=horizon_bars,
        quantity=float(vol.iloc[-1]) if len(vol) else 0.0,
    )


@st.cache_resource
def get_learner(symbol: str, horizon_minutes: int) -> OnlineLearner:
    key = f"{symbol.replace('/', '_').replace('-', '_')}_{horizon_minutes}"
    state_path = Path(__file__).resolve().parent / f"learner_state_{key}.json"
    return OnlineLearner(
        state_path=str(state_path),
        horizon_seconds=horizon_minutes * 60,
        learning_rate=0.05,
    )


def learner_step(df: pd.DataFrame, symbol: str, horizon_minutes: int) -> dict:
    if df.empty or "Close" not in df.columns:
        return {}

    close_ser = df["Close"].astype(float).dropna()
    vol_ser = (
        df["Volume"].astype(float).fillna(0.0)
        if "Volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    if len(close_ser) < 20:
        return {}

    vol_ser = vol_ser.reindex(close_ser.index).fillna(0.0)
    learner = get_learner(symbol, horizon_minutes)
    now_ts = float(pd.Timestamp(close_ser.index[-1]).timestamp())
    return learner.step(now_ts, close_ser.tolist(), vol_ser.tolist())


def render_chart(df: pd.DataFrame, symbol: str) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    ind = compute_indicators(df)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(ind.index, ind["close"], label="Close")
    ax1.plot(ind.index, ind["ema21"], label="EMA21")
    ax1.plot(ind.index, ind["ema50"], label="EMA50")
    ax1.plot(ind.index, ind["ema200"], label="EMA200")
    ax1.set_title(f"{symbol} Swing Chart")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ind.index, ind["rsi"], label="RSI14")
    ax2.axhline(70, linestyle="--")
    ax2.axhline(50, linestyle=":")
    ax2.axhline(30, linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    st.pyplot(fig)


def render_probability_panel(classic: ProbSnapshot, learner_result: dict) -> None:
    import matplotlib.pyplot as plt

    learner_ready = bool(learner_result.get("ready", False))
    p_up = float(learner_result.get("probability_up", 0.5)) if learner_result else 0.5
    edge = float(learner_result.get("edge", 0.0)) if learner_result else 0.0
    acc = float(learner_result.get("accuracy", 0.0)) if learner_result else 0.0
    pending = int(learner_result.get("pending_count", 0)) if learner_result else 0

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    state_color = (
        "green" if classic.state == "TRADABLE"
        else "orange" if classic.state == "NEUTRAL"
        else "red"
    )

    ax.text(
        0.01,
        0.92,
        f"State: {classic.state}   Volume: {int(classic.quantity)}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=state_color,
        ha="left",
    )

    if learner_result:
        if learner_ready:
            learner_txt = f"Learner P(up)={p_up:.2f}  edge={edge:+.2f}  acc={acc:.0%}  pending={pending}"
        else:
            learner_txt = f"Learner warming up... pending={pending}  acc={acc:.0%}"
    else:
        learner_txt = "Learner unavailable"

    ax.text(
        0.01,
        0.74,
        learner_txt,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        family="monospace",
    )

    edge_clip = max(-1.0, min(1.0, edge))
    if edge_clip >= 0:
        ax.barh([0.1], [edge_clip], left=0.0, height=0.22, alpha=0.85)
    else:
        ax.barh([0.1], [-edge_clip], left=edge_clip, height=0.22, alpha=0.85)

    p_down = 1.0 - p_up
    ax.barh([-0.7], [p_down], left=-p_down, height=0.22, alpha=0.9)
    ax.barh([-0.7], [p_up], left=0.0, height=0.22, alpha=0.9)
    ax.axvline(0, linestyle="--", linewidth=1.0)

    ax.set_xlabel(f"{classic.horizon_minutes}m Horizon")
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.title("NasdaqTrader")
    st.caption("Merged Streamlit app with indicator model + online learner")

    with st.sidebar:
        symbol = st.selectbox(
            "Symbol",
            ["ETH-USD", "BTC-USD", "QQQ", "SPY", "TQQQ", "SQQQ"],
            index=2,
        )
        interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "30m", "60m", "1d"],
            index=2,
        )
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=2,
        )
        horizon_minutes = st.selectbox("Horizon (minutes)", HORIZON_OPTIONS, index=6)
        model_speed = st.selectbox("Model speed", MODEL_SPEED_OPTIONS, index=1)
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_seconds = st.slider("Refresh seconds", 5, 120, 30)

    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    df = load_data(symbol, interval, period)
    if df.empty:
        st.error("No data returned.")
        return

    classic = compute_probability(
        df=df,
        interval_label=interval,
        model_speed=model_speed,
        horizon_minutes=horizon_minutes,
    )
    learner_result = learner_step(df, symbol, horizon_minutes)

    learner_ready = bool(learner_result.get("ready", False))
    learner_p_up = float(learner_result.get("probability_up", 0.5)) if learner_result else 0.5
    learner_edge = float(learner_result.get("edge", 0.0)) if learner_result else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Positive", f"{classic.positive:.1f}%")
    c2.metric("Negative", f"{classic.negative:.1f}%")
    c3.metric("Classic Edge", f"{classic.edge:.1f}")
    c4.metric("Direction", classic.direction)
    c5.metric("State", classic.state)
    c6.metric("Learner P(up)", f"{learner_p_up * 100:.1f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric("Long Prob", f"{classic.long_probability:.1f}%")
    c8.metric("Short Prob", f"{classic.short_probability:.1f}%")
    c9.metric("Learner Edge", f"{learner_edge:+.2f}")

    st.write(f"Last rerun: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Auto refresh is {'ON' if auto_refresh else 'OFF'}")
    st.write(f"Learner ready: {'Yes' if learner_ready else 'No'}")

    render_chart(df.tail(100), symbol)
    render_probability_panel(classic, learner_result)

    with st.expander("Learner raw output"):
        st.json(learner_result or {})

    with st.expander("Raw data"):
        #st.dataframe(df.tail(50), use_container_width=True)
        st.dataframe(df.tail(50), width="stretch")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()