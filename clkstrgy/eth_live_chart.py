#!/usr/bin/env python3
"""
BASELINE (v1): Live market chart with MACD, RSI, and volume-profile-based entry signals.

Features:
- Streams live spot data for BTC-USD/ETH-USD via Coinbase websocket.
- Plot refresh interval is selectable (1–15 s, default 1 s); see Plot update dropdown.
- Click once to set lower bound, once to set upper bound, then adjust nearest bound.
- Interval dropdown at top; click list (Increase/Decrease Period, Zoom In/Out) just below it.
"""

from __future__ import annotations

import json
import queue
import ssl
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import websocket
import certifi
from matplotlib.animation import FuncAnimation


WS_URL = "wss://advanced-trade-ws.coinbase.com"
DEFAULT_PLOT_UPDATE_INTERVAL_SEC = 1  # default; user can set 1–15 s via Plot update dropdown
LIVE_BUCKET = "100ms"
MAX_POINTS = 1800000  # Keep ample history with finer sub-second bucketing.
VOLUME_PROFILE_BINS = 30
VALUE_AREA_FRACTION = 0.70
DEFAULT_SYMBOL = "BTC-USD"
SYMBOL_CHOICES = ["BTC-USD", "ETH-USD"]
SESSION_STATE_PATH = Path(__file__).with_name(".last_symbol.json")
LOCAL_TZ = datetime.now().astimezone().tzinfo
LOCAL_TZ_LABEL = datetime.now().astimezone().tzname() or "Local"
INDICATOR_INTERVALS = {
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "4m": "4min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1hr": "1h",
    "4hr": "4h",
}
PROFILE_PRESETS = {
    "1": {
        "name": "Scalp",
        "view_minutes": 60,
        "interval_label": "1m",
        "price_zoom": 1.8,
        "poc_mode": "off",
        "signal_quality_min": 2,
    },
    "2": {
        "name": "Intraday",
        "view_minutes": 240,
        "interval_label": "5m",
        "price_zoom": 1.0,
        "poc_mode": "regime",
        "signal_quality_min": 3,
    },
    "3": {
        "name": "Swing",
        "view_minutes": 1440,
        "interval_label": "15m",
        "price_zoom": 0.65,
        "poc_mode": "bounce",
        "signal_quality_min": 4,
    },
}
DEFAULT_VIEW_MINUTES = 240
MIN_VIEW_MINUTES = 10
MAX_VIEW_MINUTES = 2880
MIN_PRICE_ZOOM = 0.3
MAX_PRICE_ZOOM = 6.0
PRICE_ZOOM_STEP = 1.2
MIN_MACD_ZOOM = 0.4
MAX_MACD_ZOOM = 6.0
MACD_ZOOM_STEP = 1.2
POC_FILTER_MODES = ["off", "regime", "bounce"]
SIGNAL_STYLES = ["trend", "reversal"]
POC_BOUNCE_LOOKBACK_BARS = 4
POC_BOUNCE_TOLERANCE_PCT = 0.0015
ENABLE_INITIAL_BACKFILL = True
SIGNAL_MOVE_MULTIPLIER = 1.25
SIGNAL_VOLUME_MULTIPLIER = 1.20
SIGNAL_QUALITY_MIN_LOWER = 1
SIGNAL_QUALITY_MIN_UPPER = 5
PRE_SIGNAL_NEAR_CROSS_RATIO = 0.35
EXIT_MACD_CONFIRM_BARS = 2
LONG_EXIT_RSI_TP = 72.0
LONG_EXIT_RSI_SL = 35.0
SHORT_EXIT_RSI_TP = 28.0
SHORT_EXIT_RSI_SL = 65.0


@dataclass
class Tick:
    ts: pd.Timestamp
    price: float
    size: float


class CoinbaseTickerClient:
    def __init__(self, product_id: str) -> None:
        self.product_id = product_id
        self._queue: "queue.Queue[Tick]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_price: Optional[float] = None
        self._last_volume_24h: Optional[float] = None
        self._ws: Optional[websocket.WebSocketApp] = None

    @property
    def queue(self) -> "queue.Queue[Tick]":
        return self._queue

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        subscribe_msg = {
            "type": "subscribe",
            "channel": "ticker",
            "product_ids": [self.product_id],
        }
        ws.send(json.dumps(subscribe_msg))
        print(f"[ws] subscribed to {self.product_id}")

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        channel = payload.get("channel")
        events = payload.get("events", [])
        if channel != "ticker" or not isinstance(events, list):
            return

        for event in events:
            tickers = event.get("tickers", [])
            for ticker in tickers:
                if ticker.get("product_id") != self.product_id:
                    continue
                price_str = ticker.get("price")
                size_str = ticker.get("last_size")
                vol24h_str = ticker.get("volume_24_h")
                if price_str is None:
                    continue
                try:
                    price = float(price_str)
                except (TypeError, ValueError):
                    continue

                # Prefer trade size when provided; otherwise infer from 24h volume delta.
                size = 0.0
                if size_str is not None:
                    try:
                        size = float(size_str)
                    except (TypeError, ValueError):
                        size = 0.0

                if size <= 0.0 and vol24h_str is not None:
                    try:
                        vol24h = float(vol24h_str)
                        if self._last_volume_24h is not None:
                            size = max(0.0, vol24h - self._last_volume_24h)
                        self._last_volume_24h = vol24h
                    except (TypeError, ValueError):
                        pass

                self._last_price = price
                tick = Tick(ts=pd.Timestamp.utcnow(), price=price, size=size)
                self._queue.put(tick)

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        print(f"[ws] error: {error}")

    def _on_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: Optional[int],
        close_msg: Optional[str],
    ) -> None:
        print(f"[ws] closed code={close_status_code} msg={close_msg}")

    def _run_forever(self) -> None:
        sslopt = {
            "cert_reqs": ssl.CERT_REQUIRED,
            "ca_certs": certifi.where(),
        }
        while not self._stop_event.is_set():
            self._ws = websocket.WebSocketApp(
                WS_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                self._ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    sslopt=sslopt,
                )
            except Exception as exc:
                print(f"[ws] reconnect after exception: {exc}")
            if self._stop_event.is_set():
                break
            time.sleep(2)


def load_last_symbol() -> str:
    if not SESSION_STATE_PATH.exists():
        return DEFAULT_SYMBOL
    try:
        payload = json.loads(SESSION_STATE_PATH.read_text())
    except Exception:
        return DEFAULT_SYMBOL
    symbol = payload.get("symbol")
    if symbol in SYMBOL_CHOICES:
        return symbol
    return DEFAULT_SYMBOL


def save_last_symbol(symbol: str) -> None:
    try:
        SESSION_STATE_PATH.write_text(json.dumps({"symbol": symbol}, indent=2))
    except Exception as exc:
        print(f"[session] warning: unable to save default symbol: {exc}")


def select_symbol_at_startup() -> str:
    default_symbol = load_last_symbol()
    print("\nSelect market symbol:")
    for idx, symbol in enumerate(SYMBOL_CHOICES, start=1):
        default_tag = " (default)" if symbol == default_symbol else ""
        print(f"  {idx}) {symbol}{default_tag}")

    try:
        choice = input(f"Enter 1-{len(SYMBOL_CHOICES)} or press Enter for default [{default_symbol}]: ").strip()
    except EOFError:
        choice = ""

    if choice == "":
        selected = default_symbol
    else:
        try:
            idx = int(choice) - 1
            selected = SYMBOL_CHOICES[idx]
        except (ValueError, IndexError):
            print(f"[startup] invalid choice, using default: {default_symbol}")
            selected = default_symbol

    save_last_symbol(selected)
    print(f"[startup] using symbol: {selected}")
    return selected


def build_data_client(display_symbol: str):
    if display_symbol not in {"BTC-USD", "ETH-USD"}:
        print(f"[startup] unsupported symbol '{display_symbol}', falling back to {DEFAULT_SYMBOL}")
        return CoinbaseTickerClient(DEFAULT_SYMBOL)
    return CoinbaseTickerClient(display_symbol)


def seed_history(symbol: str) -> pd.DataFrame:
    """
    Backfill recent 1-minute candles so indicators are meaningful at startup.
    Live updates still come from Coinbase websocket.
    """
    try:
        import yfinance as yf
    except Exception:
        print("[seed] yfinance not installed; starting without historical backfill")
        return pd.DataFrame(columns=["price", "volume"], dtype=float)

    try:
        bars = yf.Ticker(symbol).history(
            period="2d",
            interval="1m",
            auto_adjust=False,
        )
    except Exception as exc:
        print(f"[seed] failed to fetch history for {symbol}: {exc}")
        return pd.DataFrame(columns=["price", "volume"], dtype=float)

    if bars is None or bars.empty:
        print(f"[seed] no history for {symbol}")
        return pd.DataFrame(columns=["price", "volume"], dtype=float)

    def extract_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
        # yfinance can return either flat columns or MultiIndex columns.
        if isinstance(frame.columns, pd.MultiIndex):
            series_or_df = None
            if column_name in frame.columns.get_level_values(0):
                series_or_df = frame.xs(column_name, axis=1, level=0)
            elif column_name in frame.columns.get_level_values(1):
                series_or_df = frame.xs(column_name, axis=1, level=1)
            if series_or_df is None:
                return pd.Series(index=frame.index, dtype=float)
            if isinstance(series_or_df, pd.DataFrame):
                if series_or_df.shape[1] == 0:
                    return pd.Series(index=frame.index, dtype=float)
                return pd.to_numeric(series_or_df.iloc[:, 0], errors="coerce")
            return pd.to_numeric(series_or_df, errors="coerce")

        if column_name in frame.columns:
            return pd.to_numeric(frame[column_name], errors="coerce")
        return pd.Series(index=frame.index, dtype=float)

    price_series = extract_series(bars, "Close")
    volume_series = extract_series(bars, "Volume").fillna(0.0)
    hist = pd.concat([price_series.rename("price"), volume_series.rename("volume")], axis=1)
    hist["price"] = pd.to_numeric(hist["price"], errors="coerce")
    hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce").fillna(0.0)
    hist = hist.dropna(subset=["price"])
    hist.index = pd.to_datetime(hist.index, utc=True).floor(LIVE_BUCKET)
    return hist.tail(MAX_POINTS)


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def volume_profile_levels(
    prices: pd.Series,
    volumes: pd.Series,
    bins: int = VOLUME_PROFILE_BINS,
    value_area_fraction: float = VALUE_AREA_FRACTION,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if prices.empty or volumes.empty:
        return None, None, None
    if prices.nunique() < 2:
        p = float(prices.iloc[-1])
        return p, p, p

    pmin = float(prices.min())
    pmax = float(prices.max())
    edges = np.linspace(pmin, pmax, bins + 1)
    hist = np.zeros(bins, dtype=float)

    idx = np.clip(np.digitize(prices.to_numpy(), edges) - 1, 0, bins - 1)
    np.add.at(hist, idx, volumes.to_numpy())

    if np.all(hist == 0):
        return None, None, None

    poc_i = int(np.argmax(hist))
    poc = (edges[poc_i] + edges[poc_i + 1]) / 2

    total = hist.sum()
    target = total * value_area_fraction

    selected = {poc_i}
    running = hist[poc_i]
    left = poc_i - 1
    right = poc_i + 1

    while running < target and (left >= 0 or right < bins):
        left_vol = hist[left] if left >= 0 else -1
        right_vol = hist[right] if right < bins else -1
        if right_vol > left_vol:
            selected.add(right)
            running += right_vol
            right += 1
        else:
            selected.add(left)
            running += left_vol
            left -= 1

    val_i = min(selected)
    vah_i = max(selected)
    val = edges[val_i]
    vah = edges[vah_i + 1]
    return float(poc), float(val), float(vah)


def main() -> None:
    selected_symbol = select_symbol_at_startup()
    client = build_data_client(selected_symbol)
    is_tick_stream = isinstance(client, CoinbaseTickerClient)
    client.start()

    # Optional startup history backfill.
    df = seed_history(selected_symbol) if ENABLE_INITIAL_BACKFILL else pd.DataFrame(
        columns=["price", "volume"], dtype=float
    )

    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    view_minutes = DEFAULT_VIEW_MINUTES
    price_zoom = 1.0
    macd_zoom = 1.0

    prev_position_state: Optional[int] = None
    selected_interval_label = "5m"
    interval_menu_open = False
    interval_options = list(INDICATOR_INTERVALS.keys())
    plot_update_interval_sec = DEFAULT_PLOT_UPDATE_INTERVAL_SEC
    refresh_menu_open = False
    refresh_interval_options = list(range(1, 16))  # 1s … 15s
    last_plot_price: Optional[float] = None  # snapshot at previous animation tick (for direction)
    last_dot_color = "#ffffff"  # only updated on animation ticks; manual redraws reuse this
    ma_period = 20  # MA(20) fixed
    poc_filter_mode_idx = POC_FILTER_MODES.index("regime")
    active_profile_key = "2"
    signal_quality_min = int(PROFILE_PRESETS[active_profile_key]["signal_quality_min"])
    signal_style_idx = 0
    manual_position = 0
    manual_long_entry_times: list[pd.Timestamp] = []
    manual_long_entry_prices: list[float] = []
    manual_short_entry_times: list[pd.Timestamp] = []
    manual_short_entry_prices: list[float] = []
    manual_long_exit_times: list[pd.Timestamp] = []
    manual_long_exit_prices: list[float] = []
    manual_short_exit_times: list[pd.Timestamp] = []
    manual_short_exit_prices: list[float] = []

    fig, (ax_price, ax_macd, ax_rsi) = plt.subplots(
        3, 1, sharex=True, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    fig.canvas.manager.set_window_title(f"{selected_symbol} Live: Price, MACD, RSI")
    # Tighter top/bottom margins + smaller UI type so price trade legend and axes share one window.
    fig.subplots_adjust(left=0.05, top=0.88, bottom=0.11, right=0.92)

    # Layout: indicator interval (left) + plot update interval (right)
    interval_button = fig.text(
        0.38,
        0.995,
        f"Indicator Interval: {selected_interval_label} ▼",
        ha="center",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#2b2b2b", edgecolor="#888888"),
    )
    option_artists = []
    opt_y_step = 0.022
    for idx, label in enumerate(interval_options):
        option = fig.text(
            0.38,
            0.948 - (idx * opt_y_step),
            label,
            ha="center",
            va="top",
            color="white",
            fontsize=8,
            visible=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#3a3a3a", edgecolor="#666666"),
        )
        option_artists.append(option)

    plot_update_button = fig.text(
        0.62,
        0.995,
        f"Plot update: {plot_update_interval_sec}s ▼",
        ha="center",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#2b2b2b", edgecolor="#888888"),
    )
    refresh_option_artists = []
    for idx, sec in enumerate(refresh_interval_options):
        opt = fig.text(
            0.62,
            0.948 - (idx * opt_y_step),
            f"{sec}s",
            ha="center",
            va="top",
            color="white",
            fontsize=8,
            visible=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#3a3a3a", edgecolor="#666666"),
        )
        refresh_option_artists.append(opt)

    # Click list: top-right, compact; aligned higher with dropdown row to free vertical space
    click_list_x = 0.985
    click_list_y_start = 0.942
    click_list_dy = 0.02
    cmd_fs = 7
    command_5_hint = fig.text(
        click_list_x,
        click_list_y_start,
        "Increase Period",
        ha="right",
        va="top",
        color="white",
        fontsize=cmd_fs,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#2b2b2b", edgecolor="#666666"),
    )
    command_6_hint = fig.text(
        click_list_x,
        click_list_y_start - click_list_dy,
        "Decrease Period",
        ha="right",
        va="top",
        color="white",
        fontsize=cmd_fs,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#2b2b2b", edgecolor="#666666"),
    )
    command_7_hint = fig.text(
        click_list_x,
        click_list_y_start - 2 * click_list_dy,
        "Zoom In",
        ha="right",
        va="top",
        color="white",
        fontsize=cmd_fs,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#2b2b2b", edgecolor="#666666"),
    )
    command_8_hint = fig.text(
        click_list_x,
        click_list_y_start - 3 * click_list_dy,
        "Zoom Out",
        ha="right",
        va="top",
        color="white",
        fontsize=cmd_fs,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#2b2b2b", edgecolor="#666666"),
    )
    def refresh_timing_hints() -> None:
        return

    refresh_timing_hints()

    def apply_profile(profile_key: str) -> None:
        nonlocal view_minutes, selected_interval_label, price_zoom, poc_filter_mode_idx, active_profile_key
        nonlocal signal_quality_min
        profile = PROFILE_PRESETS[profile_key]
        view_minutes = int(profile["view_minutes"])
        selected_interval_label = str(profile["interval_label"])
        price_zoom = float(profile["price_zoom"])
        signal_quality_min = int(profile["signal_quality_min"])
        poc_mode_name = str(profile["poc_mode"])
        if poc_mode_name in POC_FILTER_MODES:
            poc_filter_mode_idx = POC_FILTER_MODES.index(poc_mode_name)
        active_profile_key = profile_key
        interval_button.set_text(f"Indicator Interval: {selected_interval_label} ▼")
        fig.canvas.draw_idle()
        print(f"[profile] {profile['name']} preset applied")

    # Mutable ref so the click handler always sees the live FuncAnimation (not a stale closure).
    plot_animation_ref: list[Optional[FuncAnimation]] = [None]

    def _apply_plot_animation_interval() -> None:
        """Sync TimedAnimation interval with plot_update_interval_sec.

        Some GUI backends (notably macOS native) do not reliably retune a running
        timer when only ``._interval`` / ``event_source.interval`` are assigned;
        stop/start applies the new cadence.
        """
        anim = plot_animation_ref[0]
        if anim is None:
            return
        ms = int(plot_update_interval_sec * 1000)
        anim._interval = ms
        es = getattr(anim, "event_source", None)
        if es is not None:
            es.stop()
            es.interval = ms
            es.start()

    def on_click(event) -> None:
        nonlocal lower_bound, upper_bound, view_minutes, price_zoom
        nonlocal selected_interval_label, interval_menu_open
        nonlocal plot_update_interval_sec, refresh_menu_open
        if event.x is None or event.y is None:
            return

        canvas = fig.canvas
        renderer = canvas.get_renderer()
        # Plot update interval menu (1–15 s)
        if refresh_menu_open:
            for sec, option in zip(refresh_interval_options, refresh_option_artists):
                opt_bbox = option.get_window_extent(renderer=renderer)
                if opt_bbox.contains(event.x, event.y):
                    plot_update_interval_sec = int(sec)
                    plot_update_button.set_text(f"Plot update: {plot_update_interval_sec}s ▼")
                    refresh_menu_open = False
                    for opt in refresh_option_artists:
                        opt.set_visible(False)
                    _apply_plot_animation_interval()
                    print(f"[timing] plot update interval set to {plot_update_interval_sec}s")
                    update(0)
                    canvas.draw_idle()
                    return
            refresh_menu_open = False
            for opt in refresh_option_artists:
                opt.set_visible(False)
            canvas.draw_idle()
            return

        # When the interval menu is open, prioritize menu option clicks over command list.
        if interval_menu_open:
            for label, option in zip(interval_options, option_artists):
                opt_bbox = option.get_window_extent(renderer=renderer)
                if opt_bbox.contains(event.x, event.y):
                    selected_interval_label = label
                    interval_button.set_text(f"Indicator Interval: {selected_interval_label} ▼")
                    interval_menu_open = False
                    for opt in option_artists:
                        opt.set_visible(False)
                    print(f"[interval] MACD/RSI interval set to {selected_interval_label}")
                    update(0)
                    canvas.draw_idle()
                    return
            interval_menu_open = False
            for opt in option_artists:
                opt.set_visible(False)
            canvas.draw_idle()
            return

        command_5_bbox = command_5_hint.get_window_extent(renderer=renderer)
        if command_5_bbox.contains(event.x, event.y):
            view_minutes = min(MAX_VIEW_MINUTES, int(view_minutes * 1.5))
            update(0)
            canvas.draw_idle()
            print(f"[view] period increased to {view_minutes}m")
            return
        command_6_bbox = command_6_hint.get_window_extent(renderer=renderer)
        if command_6_bbox.contains(event.x, event.y):
            view_minutes = max(MIN_VIEW_MINUTES, int(view_minutes / 1.5))
            update(0)
            canvas.draw_idle()
            print(f"[view] period decreased to {view_minutes}m")
            return
        command_7_bbox = command_7_hint.get_window_extent(renderer=renderer)
        if command_7_bbox.contains(event.x, event.y):
            price_zoom = min(MAX_PRICE_ZOOM, price_zoom * PRICE_ZOOM_STEP)
            update(0)
            canvas.draw_idle()
            print(f"[view] zoom in to {price_zoom:.2f}x")
            return
        command_8_bbox = command_8_hint.get_window_extent(renderer=renderer)
        if command_8_bbox.contains(event.x, event.y):
            price_zoom = max(MIN_PRICE_ZOOM, price_zoom / PRICE_ZOOM_STEP)
            update(0)
            canvas.draw_idle()
            print(f"[view] zoom out to {price_zoom:.2f}x")
            return

        interval_bbox = interval_button.get_window_extent(renderer=renderer)
        if interval_bbox.contains(event.x, event.y):
            interval_menu_open = not interval_menu_open
            refresh_menu_open = False
            for opt in refresh_option_artists:
                opt.set_visible(False)
            for option in option_artists:
                option.set_visible(interval_menu_open)
            canvas.draw_idle()
            return

        plot_update_bbox = plot_update_button.get_window_extent(renderer=renderer)
        if plot_update_bbox.contains(event.x, event.y):
            refresh_menu_open = not refresh_menu_open
            interval_menu_open = False
            for opt in option_artists:
                opt.set_visible(False)
            for opt in refresh_option_artists:
                opt.set_visible(refresh_menu_open)
            canvas.draw_idle()
            return

        if event.inaxes != ax_price or event.ydata is None:
            return

        y = float(event.ydata)
        if lower_bound is None:
            lower_bound = y
            print(f"[range] lower set to {lower_bound:.2f}")
            return

        if upper_bound is None:
            upper_bound = y
            if upper_bound < lower_bound:
                lower_bound, upper_bound = upper_bound, lower_bound
            print(f"[range] upper set to {upper_bound:.2f}")
            return

        if abs(y - lower_bound) <= abs(y - upper_bound):
            lower_bound = y
            changed = "lower"
        else:
            upper_bound = y
            changed = "upper"
        if upper_bound < lower_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
            print(f"[range] bounds swapped after {changed} adjustment")
        print(f"[range] lower={lower_bound:.2f} upper={upper_bound:.2f}")

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_key(event) -> None:
        nonlocal manual_position, view_minutes, price_zoom, macd_zoom
        nonlocal poc_filter_mode_idx, signal_quality_min, signal_style_idx
        if event.key == "1":
            if not df.empty:
                manual_long_entry_times.append(df.index[-1])
                manual_long_entry_prices.append(float(df["price"].iloc[-1]))
                manual_position = 1
            update(0)
            fig.canvas.draw_idle()
            print("[trade] manual go long")
        elif event.key == "2":
            if not df.empty and manual_position == 1:
                manual_long_exit_times.append(df.index[-1])
                manual_long_exit_prices.append(float(df["price"].iloc[-1]))
            elif not df.empty and manual_position == -1:
                manual_short_exit_times.append(df.index[-1])
                manual_short_exit_prices.append(float(df["price"].iloc[-1]))
            manual_position = 0
            update(0)
            fig.canvas.draw_idle()
            print("[trade] manual exit")
        elif event.key == "3":
            if not df.empty:
                manual_short_entry_times.append(df.index[-1])
                manual_short_entry_prices.append(float(df["price"].iloc[-1]))
                manual_position = -1
            update(0)
            fig.canvas.draw_idle()
            print("[trade] manual go short")
        elif event.key == "4":
            if not df.empty and manual_position == 1:
                manual_long_exit_times.append(df.index[-1])
                manual_long_exit_prices.append(float(df["price"].iloc[-1]))
            elif not df.empty and manual_position == -1:
                manual_short_exit_times.append(df.index[-1])
                manual_short_exit_prices.append(float(df["price"].iloc[-1]))
            manual_position = 0
            update(0)
            fig.canvas.draw_idle()
            print("[trade] manual exit")
        elif event.key == "5":
            view_minutes = min(MAX_VIEW_MINUTES, int(view_minutes * 1.5))
            update(0)
            fig.canvas.draw_idle()
            print(f"[view] period increased to {view_minutes}m")
        elif event.key == "6":
            view_minutes = max(MIN_VIEW_MINUTES, int(view_minutes / 1.5))
            update(0)
            fig.canvas.draw_idle()
            print(f"[view] period decreased to {view_minutes}m")
        elif event.key == "7":
            price_zoom = min(MAX_PRICE_ZOOM, price_zoom * PRICE_ZOOM_STEP)
            update(0)
            fig.canvas.draw_idle()
            print(f"[view] zoom in to {price_zoom:.2f}x")
        elif event.key == "8":
            price_zoom = max(MIN_PRICE_ZOOM, price_zoom / PRICE_ZOOM_STEP)
            update(0)
            fig.canvas.draw_idle()
            print(f"[view] zoom out to {price_zoom:.2f}x")
        elif event.key in {"p", "P"}:
            poc_filter_mode_idx = (poc_filter_mode_idx + 1) % len(POC_FILTER_MODES)
            update(0)
            fig.canvas.draw_idle()
            print(f"[signal] POC filter mode set to {POC_FILTER_MODES[poc_filter_mode_idx]}")
        elif event.key in {"q", "Q"}:
            signal_quality_min += 1
            if signal_quality_min > SIGNAL_QUALITY_MIN_UPPER:
                signal_quality_min = SIGNAL_QUALITY_MIN_LOWER
            update(0)
            fig.canvas.draw_idle()
            print(f"[signal] quality threshold set to >= {signal_quality_min}/5")
        elif event.key in {"m", "M"}:
            signal_style_idx = (signal_style_idx + 1) % len(SIGNAL_STYLES)
            update(0)
            fig.canvas.draw_idle()
            print(f"[signal] mode set to {SIGNAL_STYLES[signal_style_idx]}")
        elif event.key in {"k", "K"}:
            macd_zoom = min(MAX_MACD_ZOOM, macd_zoom * MACD_ZOOM_STEP)
            update(0)
            fig.canvas.draw_idle()
            print(f"[view] MACD zoom in to {macd_zoom:.2f}x")
        elif event.key in {"j", "J"}:
            macd_zoom = max(MIN_MACD_ZOOM, macd_zoom / MACD_ZOOM_STEP)
            update(0)
            fig.canvas.draw_idle()
            print(f"[view] MACD zoom out to {macd_zoom:.2f}x")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def on_scroll(event) -> None:
        nonlocal price_zoom, macd_zoom
        step = getattr(event, "step", 0)
        if step == 0:
            return
        if event.inaxes == ax_price:
            if step > 0:
                price_zoom = min(MAX_PRICE_ZOOM, price_zoom * PRICE_ZOOM_STEP)
            else:
                price_zoom = max(MIN_PRICE_ZOOM, price_zoom / PRICE_ZOOM_STEP)
        elif event.inaxes == ax_macd:
            if step > 0:
                macd_zoom = min(MAX_MACD_ZOOM, macd_zoom * MACD_ZOOM_STEP)
            else:
                macd_zoom = max(MIN_MACD_ZOOM, macd_zoom / MACD_ZOOM_STEP)
        else:
            return
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    def ingest_queue() -> None:
        nonlocal df
        has_new = False
        while True:
            try:
                tick = client.queue.get_nowait()
            except queue.Empty:
                break
            bucket_ts = tick.ts.floor(LIVE_BUCKET)
            if bucket_ts in df.index:
                df.at[bucket_ts, "price"] = tick.price
                df.at[bucket_ts, "volume"] += tick.size
            else:
                df.loc[bucket_ts, ["price", "volume"]] = [tick.price, tick.size]
            has_new = True

        if is_tick_stream and not has_new and not df.empty:
            # Hold last price and zero volume on missing ticks, keeping live cadence.
            now = pd.Timestamp.utcnow().floor(LIVE_BUCKET)
            if now > df.index[-1]:
                df.loc[now, ["price", "volume"]] = [df["price"].iloc[-1], 0.0]

        if len(df) > MAX_POINTS:
            df = df.iloc[-MAX_POINTS:]

    def update(_frame_idx: int, *, from_animation: bool = False):
        nonlocal prev_position_state, last_plot_price, last_dot_color

        ingest_queue()
        if df.empty:
            return

        # Keep indicator computations on a bounded rolling window to avoid
        # long-session rendering drift/sparsity.
        end_ts_full = df.index.max()
        analysis_minutes = max(720, int(view_minutes * 8))
        analysis_start = end_ts_full - pd.Timedelta(minutes=analysis_minutes)
        view = df.loc[df.index >= analysis_start].copy()
        if view.empty:
            view = df.copy()
        raw_close = view["price"]
        vol = view["volume"].fillna(0.0)

        interval_rule = INDICATOR_INTERVALS[selected_interval_label]
        indicator_df = view.resample(interval_rule).agg({"price": "last", "volume": "sum"}).dropna()
        ind_close = indicator_df["price"]
        plot_close = ind_close if not ind_close.empty else raw_close

        macd = pd.Series(dtype=float)
        signal = pd.Series(dtype=float)
        hist = pd.Series(dtype=float)
        rsi = pd.Series(dtype=float)
        if not ind_close.empty:
            macd, signal, hist = compute_macd(ind_close)
            rsi = compute_rsi(ind_close, period=14)
        vol_by_bar = indicator_df["volume"] if not indicator_df.empty else pd.Series(dtype=float)
        abs_move = ind_close.diff().abs() if not ind_close.empty else pd.Series(dtype=float)
        move_baseline = abs_move.rolling(20, min_periods=5).median() if not abs_move.empty else pd.Series(dtype=float)
        vol_baseline = (
            vol_by_bar.rolling(20, min_periods=5).mean() if not vol_by_bar.empty else pd.Series(dtype=float)
        )
        entry_quality_threshold = signal_quality_min
        move_multiplier = SIGNAL_MOVE_MULTIPLIER
        volume_multiplier = SIGNAL_VOLUME_MULTIPLIER
        near_cross_ratio = PRE_SIGNAL_NEAR_CROSS_RATIO

        poc, val, vah = volume_profile_levels(raw_close.tail(300), vol.tail(300))
        poc_mode = POC_FILTER_MODES[poc_filter_mode_idx]

        macd_rsi_long_entry_times: list[pd.Timestamp] = []
        macd_rsi_long_entry_prices: list[float] = []
        macd_rsi_long_exit_times: list[pd.Timestamp] = []
        macd_rsi_long_exit_prices: list[float] = []
        macd_rsi_short_entry_times: list[pd.Timestamp] = []
        macd_rsi_short_entry_prices: list[float] = []
        macd_rsi_short_exit_times: list[pd.Timestamp] = []
        macd_rsi_short_exit_prices: list[float] = []
        pre_long_times: list[pd.Timestamp] = []
        pre_long_prices: list[float] = []
        pre_short_times: list[pd.Timestamp] = []
        pre_short_prices: list[float] = []
        if len(ind_close) >= 2:
            exit_macd_confirm_bars = EXIT_MACD_CONFIRM_BARS
            long_exit_rsi_tp = LONG_EXIT_RSI_TP
            long_exit_rsi_sl = LONG_EXIT_RSI_SL
            short_exit_rsi_tp = SHORT_EXIT_RSI_TP
            short_exit_rsi_sl = SHORT_EXIT_RSI_SL
            cross_up = (macd.shift(1) <= signal.shift(1)) & (macd > signal)
            cross_down = (macd.shift(1) >= signal.shift(1)) & (macd < signal)
            macd_below_confirm = (
                (macd < signal).astype(int).rolling(exit_macd_confirm_bars, min_periods=exit_macd_confirm_bars).sum()
                >= exit_macd_confirm_bars
            )
            macd_above_confirm = (
                (macd > signal).astype(int).rolling(exit_macd_confirm_bars, min_periods=exit_macd_confirm_bars).sum()
                >= exit_macd_confirm_bars
            )
            # Combined MACD + RSI signal state machine.
            # 0 = flat, 1 = long, -1 = short
            position = 0
            for ts in ind_close.index[1:]:
                price_at_ts = float(ind_close.loc[ts])
                rsi_now = float(rsi.loc[ts]) if ts in rsi.index else 50.0
                cu = bool(cross_up.loc[ts]) if ts in cross_up.index else False
                cd = bool(cross_down.loc[ts]) if ts in cross_down.index else False

                if SIGNAL_STYLES[signal_style_idx] == "trend":
                    long_entry_base = cu and rsi_now >= 50.0 and rsi_now <= 72.0
                    short_entry_base = cd and rsi_now <= 50.0 and rsi_now >= 28.0
                    long_exit = (
                        bool(macd_below_confirm.loc[ts]) or rsi_now >= long_exit_rsi_tp or rsi_now <= long_exit_rsi_sl
                    )
                    short_exit = (
                        bool(macd_above_confirm.loc[ts]) or rsi_now <= short_exit_rsi_tp or rsi_now >= short_exit_rsi_sl
                    )
                    rsi_long_strength = 55.0 <= rsi_now <= 68.0
                    rsi_short_strength = 32.0 <= rsi_now <= 45.0
                else:
                    # Reversal mode: lean into oversold/overbought turnarounds.
                    long_entry_base = cu and rsi_now >= 30.0 and rsi_now <= 52.0
                    short_entry_base = cd and rsi_now <= 70.0 and rsi_now >= 48.0
                    long_exit = (
                        bool(macd_below_confirm.loc[ts]) or rsi_now >= long_exit_rsi_tp or rsi_now <= long_exit_rsi_sl
                    )
                    short_exit = (
                        bool(macd_above_confirm.loc[ts]) or rsi_now <= short_exit_rsi_tp or rsi_now >= short_exit_rsi_sl
                    )
                    rsi_long_strength = 35.0 <= rsi_now <= 52.0
                    rsi_short_strength = 48.0 <= rsi_now <= 65.0

                prev_price = float(ind_close.shift(1).loc[ts]) if ts in ind_close.index else price_at_ts
                bar_move = abs(price_at_ts - prev_price)
                move_ref = float(move_baseline.loc[ts]) if ts in move_baseline.index else np.nan
                move_ok = bool(np.isfinite(move_ref) and move_ref > 0 and bar_move >= (move_ref * move_multiplier))

                vol_now = float(vol_by_bar.loc[ts]) if ts in vol_by_bar.index else 0.0
                vol_ref = float(vol_baseline.loc[ts]) if ts in vol_baseline.index else np.nan
                volume_ok = bool(np.isfinite(vol_ref) and vol_ref > 0 and vol_now >= (vol_ref * volume_multiplier))

                hist_now = float(hist.loc[ts]) if ts in hist.index else 0.0
                hist_prev = float(hist.shift(1).loc[ts]) if ts in hist.index else 0.0
                momentum_long_ok = hist_now > 0 and hist_now > hist_prev
                momentum_short_ok = hist_now < 0 and hist_now < hist_prev

                long_poc_ok = True
                short_poc_ok = True
                if poc is not None and poc_mode != "off":
                    if poc_mode == "regime":
                        long_poc_ok = price_at_ts > poc
                        short_poc_ok = price_at_ts < poc
                    elif poc_mode == "bounce":
                        recent = ind_close.loc[:ts].tail(POC_BOUNCE_LOOKBACK_BARS)
                        tol = max(0.5, abs(float(poc)) * POC_BOUNCE_TOLERANCE_PCT)
                        touched = bool(((recent - float(poc)).abs() <= tol).any())
                        long_poc_ok = touched and prev_price <= float(poc) + tol and price_at_ts > float(poc)
                        short_poc_ok = touched and prev_price >= float(poc) - tol and price_at_ts < float(poc)

                long_quality = int(long_poc_ok) + int(rsi_long_strength) + int(move_ok) + int(volume_ok) + int(
                    momentum_long_ok
                )
                short_quality = int(short_poc_ok) + int(rsi_short_strength) + int(move_ok) + int(volume_ok) + int(
                    momentum_short_ok
                )
                long_entry = long_entry_base and long_quality >= entry_quality_threshold
                short_entry = short_entry_base and short_quality >= entry_quality_threshold

                # Heads-up pre-signals: show setup quality before full entry confirmation.
                macd_gap = float(macd.loc[ts] - signal.loc[ts]) if ts in macd.index and ts in signal.index else 0.0
                prev_macd_gap = (
                    float(macd.shift(1).loc[ts] - signal.shift(1).loc[ts])
                    if ts in macd.index and ts in signal.index
                    else macd_gap
                )
                near_cross = abs(macd_gap) <= max(1e-9, abs(prev_macd_gap) * near_cross_ratio)

                pre_long = (
                    position == 0
                    and not long_entry
                    and long_quality >= max(1, entry_quality_threshold - 1)
                    and long_poc_ok
                    and rsi_long_strength
                    and momentum_long_ok
                    and macd_gap <= 0
                    and macd_gap > prev_macd_gap
                    and near_cross
                )
                pre_short = (
                    position == 0
                    and not short_entry
                    and short_quality >= max(1, entry_quality_threshold - 1)
                    and short_poc_ok
                    and rsi_short_strength
                    and momentum_short_ok
                    and macd_gap >= 0
                    and macd_gap < prev_macd_gap
                    and near_cross
                )
                if pre_long:
                    pre_long_times.append(ts)
                    pre_long_prices.append(price_at_ts)
                if pre_short:
                    pre_short_times.append(ts)
                    pre_short_prices.append(price_at_ts)

                if position == 0:
                    if long_entry:
                        position = 1
                        macd_rsi_long_entry_times.append(ts)
                        macd_rsi_long_entry_prices.append(price_at_ts)
                    elif short_entry:
                        position = -1
                        macd_rsi_short_entry_times.append(ts)
                        macd_rsi_short_entry_prices.append(price_at_ts)
                elif position == 1:
                    if long_exit:
                        position = 0
                        macd_rsi_long_exit_times.append(ts)
                        macd_rsi_long_exit_prices.append(price_at_ts)
                        if short_entry:
                            position = -1
                            macd_rsi_short_entry_times.append(ts)
                            macd_rsi_short_entry_prices.append(price_at_ts)
                else:  # position == -1
                    if short_exit:
                        position = 0
                        macd_rsi_short_exit_times.append(ts)
                        macd_rsi_short_exit_prices.append(price_at_ts)
                        if long_entry:
                            position = 1
                            macd_rsi_long_entry_times.append(ts)
                            macd_rsi_long_entry_prices.append(price_at_ts)

            prev_position_state = position

        # Draw
        ax_price.cla()
        ax_macd.cla()
        ax_rsi.cla()

        last_ts = raw_close.index[-1]
        last_price = float(raw_close.iloc[-1])

        ax_price.plot(plot_close.index, plot_close.values, color="white", lw=1.4, label=selected_symbol)
        # Dot color updates only on FuncAnimation ticks (plot update interval). Manual redraws
        # (keys, clicks, etc.) keep last_dot_color so live price moves don't flip green/red between ticks.
        if from_animation:
            if last_plot_price is None:
                blip_c = "#ffffff"
            elif last_price > last_plot_price:
                blip_c = "#2ecc71"
            elif last_price < last_plot_price:
                blip_c = "#e74c3c"
            else:
                blip_c = "#ffffff"
            last_plot_price = last_price
            last_dot_color = blip_c
        else:
            blip_c = last_dot_color
        ax_price.scatter(
            [last_ts],
            [last_price],
            s=42,
            c=blip_c,
            edgecolors="#333333",
            linewidths=0.4,
            zorder=9,
            clip_on=False,
        )
        ax_price.set_ylabel("Price (USD)", fontsize=9)
        ax_price.grid(alpha=0.2)
        ax_price.set_facecolor("#111111")
        ax_price.figure.patch.set_facecolor("#1b1b1b")
        ax_price.tick_params(colors="white", labelsize=7)
        ax_price.yaxis.label.set_color("white")
        ax_price.title.set_color("white")
        for spine in ax_price.spines.values():
            spine.set_color("#666666")

        if lower_bound is not None:
            ax_price.axhline(lower_bound, color="yellow", linestyle="--", lw=1.2, label="Lower range")
        if upper_bound is not None:
            ax_price.axhline(upper_bound, color="red", linestyle="--", lw=1.2, label="Upper range")
        if poc is not None:
            ax_price.axhline(poc, color="cyan", linestyle=":", lw=1.0, label="POC")
        if val is not None:
            ax_price.axhline(val, color="#4ecdc4", linestyle=":", lw=0.9, alpha=0.7, label="VAL")
        if vah is not None:
            ax_price.axhline(vah, color="#ff9f1c", linestyle=":", lw=0.9, alpha=0.7, label="VAH")

        if macd_rsi_long_entry_times:
            ax_price.scatter(
                macd_rsi_long_entry_times,
                macd_rsi_long_entry_prices,
                marker="^",
                color="lime",
                s=44,
                edgecolors="#111111",
                linewidths=0.4,
                label=f"Long entry (MACD+RSI {selected_interval_label})",
                zorder=5,
            )
        if manual_long_entry_times:
            ax_price.scatter(
                manual_long_entry_times,
                manual_long_entry_prices,
                marker="^",
                color="#00e676",
                s=52,
                edgecolors="white",
                linewidths=0.6,
                label="Manual long",
                zorder=6,
            )
        if manual_long_exit_times:
            ax_price.scatter(
                manual_long_exit_times,
                manual_long_exit_prices,
                marker="x",
                color="#00e676",
                s=58,
                linewidths=1.4,
                label="Exit long",
                zorder=7,
            )
        if macd_rsi_short_entry_times:
            ax_price.scatter(
                macd_rsi_short_entry_times,
                macd_rsi_short_entry_prices,
                marker="v",
                color="magenta",
                s=44,
                edgecolors="#111111",
                linewidths=0.4,
                label=f"Short entry (MACD+RSI {selected_interval_label})",
                zorder=5,
            )
        if manual_short_entry_times:
            ax_price.scatter(
                manual_short_entry_times,
                manual_short_entry_prices,
                marker="v",
                color="#ff4fc3",
                s=52,
                edgecolors="white",
                linewidths=0.6,
                label="Manual short",
                zorder=6,
            )
        if manual_short_exit_times:
            ax_price.scatter(
                manual_short_exit_times,
                manual_short_exit_prices,
                marker="x",
                color="#ff5252",
                s=58,
                linewidths=1.4,
                label="Exit short",
                zorder=7,
            )
        if pre_long_times:
            ax_price.scatter(
                pre_long_times,
                pre_long_prices,
                marker="o",
                facecolors="none",
                edgecolors="#ffd54f",
                s=26,
                linewidths=1.0,
                label="Pre-long setup",
                zorder=4,
            )
        if pre_short_times:
            ax_price.scatter(
                pre_short_times,
                pre_short_prices,
                marker="o",
                facecolors="none",
                edgecolors="#80deea",
                s=26,
                linewidths=1.0,
                label="Pre-short setup",
                zorder=4,
            )

        end_ts = raw_close.index.max()
        start_ts = end_ts - pd.Timedelta(minutes=view_minutes)
        visible = plot_close[(plot_close.index >= start_ts) & (plot_close.index <= end_ts)]
        if visible.empty:
            visible = plot_close
        y_min = float(visible.min())
        y_max = float(visible.max())
        if lower_bound is not None:
            y_min = min(y_min, float(lower_bound))
            y_max = max(y_max, float(lower_bound))
        if upper_bound is not None:
            y_min = min(y_min, float(upper_bound))
            y_max = max(y_max, float(upper_bound))
        y_pad = max(0.5, (y_max - y_min) * 0.06)
        center = (y_max + y_min) / 2.0
        half_span = ((y_max - y_min) / 2.0) + y_pad
        zoomed_half_span = max(0.2, half_span / price_zoom)

        ax_price.legend(
            loc="upper left",
            bbox_to_anchor=(0.008, 1.0),
            bbox_transform=fig.transFigure,
            fontsize=6,
            facecolor="#202020",
            edgecolor="#666666",
            labelcolor="white",
            framealpha=0.95,
            ncol=2,
            columnspacing=0.55,
            handlelength=1.15,
            handletextpad=0.45,
            labelspacing=0.28,
            borderpad=0.32,
        )

        # Plot MACD on the current visible time window for stable density.
        macd_visible = macd[(macd.index >= start_ts) & (macd.index <= end_ts)] if not macd.empty else macd
        signal_visible = signal[(signal.index >= start_ts) & (signal.index <= end_ts)] if not signal.empty else signal
        hist_visible = hist[(hist.index >= start_ts) & (hist.index <= end_ts)] if not hist.empty else hist

        # Ensure MACD begins at start of view (prepend start_ts if first point is later).
        if not macd_visible.empty and macd_visible.index[0] > start_ts:
            first_val = float(macd_visible.iloc[0])
            macd_visible = pd.concat([pd.Series({start_ts: first_val}), macd_visible]).sort_index()
        if not signal_visible.empty and signal_visible.index[0] > start_ts:
            first_val = float(signal_visible.iloc[0])
            signal_visible = pd.concat([pd.Series({start_ts: first_val}), signal_visible]).sort_index()
        if not hist_visible.empty and hist_visible.index[0] > start_ts:
            first_val = float(hist_visible.iloc[0])
            hist_visible = pd.concat([pd.Series({start_ts: first_val}), hist_visible]).sort_index()

        if macd_visible.empty and not macd.empty:
            macd_visible = macd.tail(240)
        if signal_visible.empty and not signal.empty:
            signal_visible = signal.tail(240)
        if hist_visible.empty and not hist.empty:
            hist_visible = hist.tail(240)

        ax_macd.plot(
            macd_visible.index,
            macd_visible.values,
            label=f"MACD ({selected_interval_label})",
            color="#00d4ff",
            lw=1.1,
        )
        ax_macd.plot(signal_visible.index, signal_visible.values, label="Signal", color="#ff8f00", lw=1.1)
        colors = np.where(hist_visible.values >= 0, "#4caf50", "#f44336") if not hist_visible.empty else []
        bar_width = 0.004
        if len(hist_visible.index) >= 2:
            diffs_days = np.diff(hist_visible.index.asi8) / 86_400_000_000_000.0
            positive = diffs_days[diffs_days > 0]
            if positive.size:
                dt_days = float(np.median(positive))
                bar_width = max(0.0015, dt_days * 0.95)
        ax_macd.bar(hist_visible.index, hist_visible.values, color=colors, alpha=0.75, width=bar_width, label="Hist")
        ax_macd.axhline(0, color="gray", lw=0.8)
        macd_components = []
        if not macd_visible.empty:
            macd_components.append(macd_visible.values)
        if not signal_visible.empty:
            macd_components.append(signal_visible.values)
        if not hist_visible.empty:
            macd_components.append(hist_visible.values)
        if macd_components:
            macd_vals = np.concatenate(macd_components)
            finite_vals = macd_vals[np.isfinite(macd_vals)]
            if finite_vals.size:
                # Keep MACD panel centered on zero with symmetric y-limits.
                max_abs = float(np.max(np.abs(finite_vals)))
                macd_half = max(1e-4, max_abs * 1.25) / max(1e-6, macd_zoom)
                ax_macd.set_ylim(-macd_half, macd_half)
        ax_macd.set_ylabel("MACD", fontsize=9)
        ax_macd.tick_params(colors="white", labelsize=7)
        ax_macd.grid(alpha=0.2)
        ax_macd.set_facecolor("#8c8c8c")
        ax_macd.legend(
            loc="upper left",
            fontsize=7,
            facecolor="#202020",
            edgecolor="#666666",
            labelcolor="white",
            framealpha=0.95,
        )

        ax_rsi.plot(rsi.index, rsi.values, label=f"RSI(14) {selected_interval_label}", color="#8e44ad", lw=1.2)
        ax_rsi.axhline(70, color="red", linestyle="--", lw=0.9)
        ax_rsi.axhline(30, color="green", linestyle="--", lw=0.9)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", fontsize=9)
        ax_rsi.set_xlabel(f"Time ({LOCAL_TZ_LABEL})", fontsize=9)
        ax_rsi.tick_params(axis="y", colors="black", labelsize=7)
        ax_rsi.grid(alpha=0.2)
        ax_rsi.set_facecolor("#b2ffff")
        ax_rsi.legend(
            loc="upper left",
            fontsize=7,
            facecolor="#202020",
            edgecolor="#666666",
            labelcolor="white",
            framealpha=0.95,
        )

        # Shared x-axis across all panels. Small margin so plots don't jam at edges.
        span = end_ts - start_ts
        xpad = span * 0.02
        ax_price.set_xlim(start_ts - xpad, end_ts + xpad)
        # Time labels on shared x-axis (same approach as price_volume_volity).
        time_locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        time_formatter = mdates.DateFormatter("%H:%M:%S", tz=LOCAL_TZ)
        ax_rsi.xaxis.set_major_locator(time_locator)
        ax_rsi.xaxis.set_major_formatter(time_formatter)
        ax_price.tick_params(axis="x", labelbottom=False)
        ax_macd.tick_params(axis="x", labelbottom=False)
        ax_rsi.tick_params(axis="x", labelrotation=28, colors="white", labelsize=7, labelbottom=True, pad=5)

        ax_price.set_ylim(center - zoomed_half_span, center + zoomed_half_span)

        # Keep fixed subplot geometry; avoid per-frame tight_layout clipping tick labels.

    def _animation_tick(frame: int) -> None:
        update(frame, from_animation=True)

    ani = FuncAnimation(
        fig,
        _animation_tick,
        interval=int(plot_update_interval_sec * 1000),
        cache_frame_data=False,
    )
    plot_animation_ref[0] = ani
    update(0)
    fig.canvas.draw_idle()
    try:
        plt.show()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
