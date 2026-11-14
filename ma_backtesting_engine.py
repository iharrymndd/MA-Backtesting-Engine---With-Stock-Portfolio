"""EMA / SMA crossover strategy dashboard (Streamlit) - fast version with a manual Run button.

Page layout
  * Permutations (top): pick any parameters, say for each whether it is fixed, a range,
    specific values or all values, pick the metrics you want, and get one result row per
    combination (sortable table, CSV download, optional two-parameter pivot).
  * Backtest engine (below): the original section 1 / section 2 workflow.

Positional long/short backtest on Yahoo Finance data (timeframe selectable in the sidebar: 1 minute to 1 day),
for one stock or an equal-capital portfolio of any number of stocks, with:
  * sideways-market "no-trade zone" filters (ER, ADX, Vortex),
  * heat-map based parameter estimation on a training window (section 1),
    evaluated out of sample on a backtest window (section 2).

Timing convention: a signal is computed at the close of bar t, the entry price
is bar t's close, and the position earns returns from bar t+1.

Speed-ups compared with the previous version
  * one price download per run (training and backtest windows are sliced from it),
  * the grid search works on plain NumPy arrays; EMA / SMA series are
    computed once per length instead of once per (EMA, SMA) pair,
  * the position state machine is JIT-compiled when `numba` is installed
    (optional: `pip install numba`; without it the same code runs as plain Python),
  * neighbourhood statistics are vectorised,
  * no-trade bands are drawn as one shape each instead of via add_vrect,
  * results and figures are kept in session state, so touching a widget only
    re-draws the page; the heavy work runs only when "Run backtest" is pressed.

Run with:  streamlit run main.py
"""
from __future__ import annotations

import io
import itertools
import math
import re
import warnings
from dataclasses import asdict, astuple, dataclass, field, fields, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:                                    # optional JIT
    from numba import njit
except ImportError:                     # fall back to plain Python
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
TRADING_DAYS = 252
MAX_SMA_LEN = 400          # largest SMA length selectable in the sidebar
# Extra calendar days downloaded before each start date so that every indicator
# (SMA up to 400 bars, ADX smoothing ~5x period) is fully warmed up on day one.
WARMUP_DAYS = 1500
CHART_MARGIN = dict(l=0, r=0, t=30, b=0)

# Timeframe (bar size). Each entry: label, how far back Yahoo Finance serves that bar size (calendar days,
# None = no limit) and the calendar days of warm-up history downloaded before a start date.
INTERVALS = {
    "1m": ("1 minute", 29, 3),
    "5m": ("5 minutes", 59, 10),
    "15m": ("15 minutes", 59, 30),
    "30m": ("30 minutes", 59, 60),
    "1h": ("1 hour", 729, 120),
    "1d": ("1 day", None, WARMUP_DAYS),
}

M_PLATEAU = "Best neighbourhood average (stable plateau)"
M_RAW = "Highest raw return"
M_ROBUST = "Mean minus std of neighbours (robust)"
M_RATIO = "Best return / drawdown (neighbourhood average)"
M_CONSISTENT = "Most consistent year to year (mean / std of yearly PnL)"
METHOD_NAMES = [M_PLATEAU, M_RAW, M_ROBUST, M_RATIO, M_CONSISTENT]


class UserError(Exception):
    """A problem the user can fix (bad dates, no data, ...)."""


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Strategy settings shared by signals, filters, backtest and grid search."""
    allow_short: bool
    exit_filter: bool
    stop_pct: float           # stop loss in % from entry price, 0 = off
    use_er: bool
    er_n: int
    er_min: float
    use_adx: bool
    adx_n: int
    adx_min: float
    use_vi_cross: bool
    use_vi_band: bool
    vi_n: int
    vi_lo: float
    vi_hi: float
    cost_pct: float           # per side, % of trade value
    timeline: tuple = ()      # allowed trading date ranges ((from, to), ...); empty = trade throughout


@dataclass(frozen=True)
class Profile:
    """One ticker's own strategy settings: what the sidebar edits for the ticker selected in the switcher."""
    cfg: Config
    man_ema: int
    man_sma: int
    use_estimated: bool
    method: str


@dataclass(frozen=True)
class Inputs:
    """Everything the user sets in the sidebar. `cfg`, `man_ema`, `man_sma`, `use_estimated` and `method` are
    those of the ticker currently selected in the switcher; `profiles` holds every ticker's own set."""
    tickers: tuple[str, ...]
    capital: float
    rf_pct: float
    use_train: bool
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    use_estimated: bool
    method: str
    man_ema: int
    man_sma: int
    emas: list[int]
    smas: list[int]
    cfg: Config
    timeline_error: str = ""
    interval: str = "1d"
    profiles: dict = field(default_factory=dict)       # ticker -> Profile
    active: str = ""

    @property
    def label(self) -> str:
        return self.tickers[0] if len(self.tickers) == 1 else f"Portfolio of {len(self.tickers)} stocks"

    @property
    def file_tag(self) -> str:
        return self.tickers[0] if len(self.tickers) == 1 else f"portfolio_{len(self.tickers)}_stocks"


@dataclass
class Sleeve:
    """One stock's share of the portfolio: its own capital, parameters, signals, trades and metrics."""
    ticker: str
    alloc: float
    sig: pd.DataFrame
    indicators: dict
    trades: pd.DataFrame
    metrics: dict
    cfg: Config
    ema_len: int
    sma_len: int
    source: str                                   # where its EMA / SMA lengths came from


@dataclass
class SectionOne:
    """Section 1 (parameter estimation) result of one ticker."""
    methods: dict = field(default_factory=dict)
    chosen_method: str | None = None
    est_pair: tuple | None = None
    ret_grid: pd.DataFrame | None = None
    dd_grid: pd.DataFrame | None = None
    ratio_grid: pd.DataFrame | None = None


@dataclass
class Results:
    """Everything the page needs to draw itself after a run."""
    inp: Inputs
    messages: list[tuple[str, str]]               # (level, text)
    sig: pd.DataFrame                             # portfolio frame
    trades: pd.DataFrame
    metrics: dict[str, float]
    figs: dict[str, go.Figure] = field(default_factory=dict)
    sleeves: dict[str, Sleeve] = field(default_factory=dict)      # one per stock, in input order
    sections: dict[str, SectionOne] = field(default_factory=dict)
    sleeve_equity: pd.DataFrame | None = None                     # each stock's equity on the portfolio calendar
    stock_table: pd.DataFrame | None = None
    corr: pd.DataFrame | None = None
    skipped: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def earliest_allowed(interval: str) -> date | None:
    """Oldest start date Yahoo Finance serves for this bar size (None = no limit)."""
    limit = INTERVALS[interval][1]
    return None if limit is None else date.today() - timedelta(days=limit)


def interval_problems(interval: str, start: date, end: date) -> list[str]:
    earliest = earliest_allowed(interval)
    if earliest is not None and start < earliest:
        label, limit = INTERVALS[interval][0], INTERVALS[interval][1]
        return [f"Yahoo Finance only provides {label} bars for about the last {limit + 1} days, so the start date "
                f"must be {earliest} or later (it is {start}). Move the dates, or pick a longer timeframe."]
    return []


def periods_per_year(index: pd.DatetimeIndex, interval: str) -> float:
    """Bars per year, used to annualise Sharpe / Sortino / volatility. Daily keeps 252; for intraday bars it is
    measured from the data itself, so it fits any exchange's trading hours."""
    if interval == "1d" or len(index) < 2:
        return float(TRADING_DAYS)
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return len(index) / years if years > 0 else float(TRADING_DAYS)


def _download(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(show_spinner="Downloading prices...", ttl=3600)
def load_span(ticker: str, lo: date, hi: date, interval: str = "1d") -> pd.DataFrame:
    """Auto-adjusted OHLC bars of the chosen timeframe from `lo` - warm-up days to `hi` (inclusive).
    Intraday history is limited by Yahoo (see INTERVALS), so the download starts no earlier than that limit."""
    first = lo - timedelta(days=INTERVALS[interval][2])
    last = hi + timedelta(days=1)
    earliest = earliest_allowed(interval)
    if earliest is not None:
        first = max(first, earliest)
        last = min(last, date.today() + timedelta(days=1))
    if first >= last:
        return pd.DataFrame()

    if interval == "1m":                                   # Yahoo serves at most 7 days of 1-minute bars per request
        parts, cur = [], first
        while cur < last:
            nxt = min(cur + timedelta(days=7), last)
            parts.append(_download(ticker, cur, nxt, interval))
            cur = nxt
        parts = [x for x in parts if not x.empty]
        df = pd.concat(parts) if parts else pd.DataFrame()
    else:
        df = _download(ticker, first, last, interval)
    if df.empty:
        return df

    if getattr(df.index, "tz", None) is not None:           # keep exchange-local wall-clock time, drop the zone
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated()].sort_index()


def price_window(span: pd.DataFrame, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
    """`start` - warm-up .. end of day `end`, sliced from the downloaded span."""
    if span.empty:
        return span
    lo = pd.Timestamp(start - timedelta(days=INTERVALS[interval][2]))
    hi = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return span.loc[lo:hi]


# ---------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def vortex(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Vortex indicator (+VI, -VI)."""
    vm_plus = (df["High"] - df["Low"].shift(1)).abs()
    vm_minus = (df["Low"] - df["High"].shift(1)).abs()
    tr_sum = true_range(df).rolling(period).sum()
    return vm_plus.rolling(period).sum() / tr_sum, vm_minus.rolling(period).sum() / tr_sum


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """ER = |P_t - P_(t-N)| / sum(|P_i - P_(i-1)|) over the last N bars (0..1)."""
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return change / path.replace(0, np.nan)


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder's ADX."""
    up, down = df["High"].diff(), -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    alpha = 1.0 / n
    atr = true_range(df).ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def ema_values(close: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(close).ewm(span=n, adjust=False).mean().to_numpy()


def sma_values(close: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(close).rolling(n).mean().to_numpy()


# Which no-trade condition fired (bit flags, so several can be true on one bar)
NT_ER, NT_ADX, NT_VI_CROSS, NT_VI_BAND, NT_TIMELINE = 1, 2, 4, 8, 16
NT_NAMES = {
    NT_ER: "Efficiency Ratio below threshold",
    NT_ADX: "ADX below threshold",
    NT_VI_CROSS: "Vortex: +VI below -VI",
    NT_VI_BAND: "Vortex: inside the no-trade band",
    NT_TIMELINE: "Outside the trading timeline",
}


def no_trade_parts(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, dict[str, pd.Series]]:
    """Sideways-market filters. Returns an int array of NT_* bit flags per bar (0 = tradable)
    and the indicator series that were computed (for charting). Uses the full, warmed-up history."""
    bits = np.zeros(len(df), dtype=np.int64)
    ind: dict[str, pd.Series] = {}
    if cfg.use_er:
        er = efficiency_ratio(df["Close"], cfg.er_n)
        ind["ER"] = er
        bits |= ((er < cfg.er_min) | er.isna()).to_numpy().astype(np.int64) * NT_ER
    if cfg.use_adx:
        adx_series = adx(df, cfg.adx_n)
        ind["ADX"] = adx_series
        bits |= ((adx_series < cfg.adx_min) | adx_series.isna()).to_numpy().astype(np.int64) * NT_ADX
    if cfg.use_vi_cross or cfg.use_vi_band:
        vi_p, vi_m = vortex(df, cfg.vi_n)
        ind["+VI"], ind["-VI"] = vi_p, vi_m
        if cfg.use_vi_cross:
            bits |= (vi_p < vi_m).fillna(False).to_numpy().astype(np.int64) * NT_VI_CROSS
        if cfg.use_vi_band:
            in_band = vi_p.between(cfg.vi_lo, cfg.vi_hi) & vi_m.between(cfg.vi_lo, cfg.vi_hi)
            bits |= in_band.fillna(False).to_numpy().astype(np.int64) * NT_VI_BAND
    return bits, ind


def no_trade_mask(df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, dict[str, pd.Series]]:
    """True = no-trade zone (any enabled filter fires), plus the indicator series."""
    bits, ind = no_trade_parts(df, cfg)
    return pd.Series(bits != 0, index=df.index), ind


def off_timeline_mask(index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """True on bars outside every chosen trading date range (all False when no timeline is set)."""
    if not cfg.timeline:
        return np.zeros(len(index), dtype=bool)
    ok = np.zeros(len(index), dtype=bool)
    for a, b in cfg.timeline:
        ok |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b) + pd.Timedelta(days=1) - pd.Timedelta(1, "ns"))
    return ~ok


def describe_timeline(cfg: Config) -> str:
    return "; ".join(f"{a} to {b}" for a, b in cfg.timeline) if cfg.timeline else "Off (trades throughout)"


# ---------------------------------------------------------------
# Fast NumPy core (used by both the grid search and the section-2 backtest)
# ---------------------------------------------------------------
@njit(cache=True)
def _positions(buy, sell, cross_up, cross_dn, above, close, ema, bits, allow_short, exit_filter, stop_frac):
    """Position decided at each bar's close: +1 long, -1 short, 0 flat.

    * Entries only on the fresh crossover bar; a blocked bar is skipped for good.
    * Exit on the opposite crossover, optionally confirmed by close vs EMA; the
      pending exit is cancelled if EMA returns to its original side of the SMA.
    * A no-trade zone exits any running trade and blocks new entries.
    * Stop loss (stop_frac > 0): a long exits at the first close at or below entry x (1 - stop),
      a short at the first close at or above entry x (1 + stop). Checked on closes only, so the
      exit is the close of that bar (it can be worse than the stop level after a gap).
      No re-entry until the next fresh crossover.
    * An exit and an opposite entry on the same bar flip the position.

    Returns (pos, why). `bits` holds NT_* flags (non-zero = blocked bar). why[i] says why a trade
    was closed on bar i: 1 stop loss, 2 crossover (exit filter off), 3 crossover plus close beyond
    EMA on the same bar, 4 earlier crossover confirmed later by the close beyond EMA,
    1000 + bits = closed by a no-trade zone / timeline (bits tell which).
    """
    n = len(close)
    pos = np.zeros(n, dtype=np.int64)
    why = np.zeros(n, dtype=np.int64)
    cur = 0
    entry = 0.0
    pending = False                      # opposite crossover seen, awaiting exit confirmation
    for i in range(n):
        if bits[i] != 0:
            if cur != 0:
                why[i] = 1000 + bits[i]
            cur = 0
            pending = False
        else:
            if stop_frac > 0.0:
                if cur == 1 and close[i] <= entry * (1.0 - stop_frac):
                    cur = 0
                    pending = False
                    why[i] = 1
                elif cur == -1 and close[i] >= entry * (1.0 + stop_frac):
                    cur = 0
                    pending = False
                    why[i] = 1
            if cur == 1:
                was_pending = pending
                pending = (pending or cross_dn[i]) and (not above[i])
                if pending and ((not exit_filter) or close[i] < ema[i]):
                    cur = 0
                    pending = False
                    why[i] = 4 if was_pending else (3 if exit_filter else 2)
            elif cur == -1:
                was_pending = pending
                pending = (pending or cross_up[i]) and above[i]
                if pending and ((not exit_filter) or close[i] > ema[i]):
                    cur = 0
                    pending = False
                    why[i] = 4 if was_pending else (3 if exit_filter else 2)
            if cur == 0:
                if buy[i]:
                    cur = 1
                    entry = close[i]
                    pending = False
                elif sell[i] and allow_short:
                    cur = -1
                    entry = close[i]
                    pending = False
        pos[i] = cur
    return pos, why


def compute_pair(close: np.ndarray, ema: np.ndarray, sma: np.ndarray,
                 blocked: np.ndarray, i0: int, cfg: Config, capital: float,
                 bits: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Signals, positions, returns and equity for one (EMA, SMA) pair.

    All inputs cover the warmed-up history; the returned arrays are trimmed to
    bars `i0:` (the evaluation window). `bits` (optional NT_* flags) lets the trade log
    say which no-trade condition closed a trade; without it each blocked bar is flagged 1.
    """
    if bits is None:
        bits = blocked.astype(np.int64)
    valid = ~np.isnan(sma)
    above = (ema > sma) & valid
    prev_above = np.concatenate(([False], above[:-1]))
    prev_valid = np.concatenate(([False], valid[:-1]))
    cross_up = above & ~prev_above & prev_valid          # EMA crosses above SMA
    cross_dn = ~above & valid & prev_above                # EMA crosses below SMA

    buy, sell = cross_up, cross_dn

    w = slice(i0, None)
    c = close[w]
    pos, why = _positions(buy[w], sell[w], cross_up[w], cross_dn[w], above[w], c, ema[w],
                          bits[w], cfg.allow_short, cfg.exit_filter, cfg.stop_pct / 100.0)

    ret = np.zeros(len(c))
    ret[1:] = c[1:] / c[:-1] - 1.0
    pos_prev = np.zeros(len(c))
    pos_prev[1:] = pos[:-1]
    turnover = np.zeros(len(c))                           # a flip is 2 units
    turnover[1:] = np.abs(np.diff(pos_prev))
    gross = pos_prev * ret
    strat = gross - turnover * cfg.cost_pct / 100.0
    return {
        "pos": pos, "why": why, "ret": ret, "gross_ret": gross, "strat_ret": strat,
        "equity": capital * np.cumprod(1.0 + strat),
        "equity_gross": capital * np.cumprod(1.0 + gross),
        "above": above[w], "cross_up": cross_up[w], "cross_dn": cross_dn[w],
        "buy": buy[w], "sell": sell[w],
    }


def _years(first: pd.Timestamp, last: pd.Timestamp) -> float:
    return (last - first).total_seconds() / (365.25 * 86400)


def _annual_return(final_equity: float, capital: float, years: float) -> float:
    if years <= 0:
        return 0.0
    ratio = float(final_equity) / capital
    if ratio <= 0:
        return -100.0
    return (math.exp(min(math.log(ratio) / years, 50.0)) - 1.0) * 100.0


def _max_drawdown(equity: np.ndarray) -> float:
    return float((equity / np.maximum.accumulate(equity) - 1.0).min() * 100.0)


# ---------------------------------------------------------------
# Heat-map grid and parameter estimation (section 1)
# ---------------------------------------------------------------
def union_index(indexes: list[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Sorted union of several stocks' bar dates (the portfolio calendar)."""
    out = indexes[0]
    for ix in indexes[1:]:
        out = out.union(ix)
    return out


def carry_map(idx: pd.DatetimeIndex, union: pd.DatetimeIndex) -> np.ndarray:
    """For each portfolio date, the position of the stock's latest bar on or before it (-1 = not listed yet)."""
    return idx.searchsorted(union, side="right") - 1


def carry(arr: np.ndarray, take: np.ndarray, fill) -> np.ndarray:
    """Put a stock's series on the portfolio calendar: carried forward on dates it has no bar,
    `fill` (its idle cash) before its first bar."""
    return np.where(take >= 0, arr[np.clip(take, 0, None)], fill)


def returns_from_equity(equity: np.ndarray, capital: float) -> np.ndarray:
    return equity / np.concatenate(([capital], equity[:-1])) - 1.0


@st.cache_data(show_spinner="Running EMA/SMA grid on section 1...", hash_funcs={Config: astuple})
def run_grid(tickers: tuple[str, ...], span_lo: date, span_hi: date, start: date, end: date,
             emas: list[int], smas: list[int], cfg: Config, capital: float, interval: str = "1d"):
    """Backtest every EMA < SMA pair on an equal-capital portfolio of `tickers` (one sleeve per stock).
    Returns (avg annual return, max drawdown, yearly returns), all measured on the combined portfolio equity."""
    ret_grid = np.full((len(emas), len(smas)), np.nan)
    dd_grid = ret_grid.copy()
    yearly: dict[tuple[int, int], np.ndarray] = {}

    def as_frames():
        return (pd.DataFrame(ret_grid, index=emas, columns=smas),
                pd.DataFrame(dd_grid, index=emas, columns=smas), yearly)

    stocks: list[dict] = []
    for tk in tickers:
        prices = price_window(load_span(tk, span_lo, span_hi, interval), start, end, interval)
        if prices.empty:
            continue
        i0 = int(prices.index.searchsorted(pd.Timestamp(start)))
        idx = prices.index[i0:]
        if len(idx) < 2:
            continue
        close = prices["Close"].to_numpy(dtype=float)
        stocks.append({
            "idx": idx, "i0": i0, "close": close,
            "blocked": no_trade_mask(prices, cfg)[0].to_numpy(dtype=bool),
            # Each MA is computed once per stock and reused by every pair that needs it.
            "ema": {e: ema_values(close, e) for e in emas},
            "sma": {s: sma_values(close, s) for s in smas},
        })
    if not stocks:
        return as_frames()

    alloc = capital / len(stocks)
    union = union_index([d["idx"] for d in stocks])
    for d in stocks:
        d["take"] = carry_map(d["idx"], union)
    years = _years(union[0], union[-1])
    year_arr = union.year.to_numpy()
    year_starts = np.flatnonzero(np.concatenate(([True], year_arr[1:] != year_arr[:-1])))

    for i, e in enumerate(emas):
        for j, s in enumerate(smas):
            if e >= s:
                continue
            total = np.zeros(len(union))
            for d in stocks:
                r = compute_pair(d["close"], d["ema"][e], d["sma"][s], d["blocked"], d["i0"], cfg, alloc)
                total += carry(r["equity"], d["take"], alloc)
            ret_grid[i, j] = _annual_return(total[-1], capital, years)
            dd_grid[i, j] = _max_drawdown(total)
            yearly[(e, s)] = (np.multiply.reduceat(1.0 + returns_from_equity(total, capital), year_starts) - 1.0) * 100.0
    return as_frames()


def neighbourhood_stat(grid: pd.DataFrame, kind: str, min_n: int = 1) -> pd.DataFrame:
    """3x3 neighbourhood mean ("mean") or mean minus std ("mean_minus_std") of each cell, NaNs ignored."""
    v = grid.to_numpy(dtype=float)
    rows, cols = v.shape
    padded = np.pad(v, 1, constant_values=np.nan)
    stack = np.stack([padded[di:di + rows, dj:dj + cols] for di in range(3) for dj in range(3)])
    count = (~np.isnan(stack)).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(stack, axis=0)
        if kind == "mean_minus_std":
            out = out - np.nanstd(stack, axis=0)
    out = np.where(np.isnan(v) | (count < min_n), np.nan, out)
    return pd.DataFrame(out, index=grid.index, columns=grid.columns)


def best_pairs(ret_grid: pd.DataFrame, ratio_grid: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Best (EMA, SMA) pair under each grid-based selection method."""
    valid = ret_grid.stack().dropna()
    if valid.empty:
        return {}
    methods = {M_RAW: valid.idxmax()}
    candidates = {
        M_PLATEAU: neighbourhood_stat(ret_grid, "mean"),
        M_ROBUST: neighbourhood_stat(ret_grid, "mean_minus_std", min_n=2),
        M_RATIO: neighbourhood_stat(ratio_grid, "mean"),
    }
    for name, stat in candidates.items():
        scores = stat.stack().dropna()
        if not scores.empty:
            methods[name] = scores.idxmax()
    return methods


def consistency_pair(yearly: dict[tuple[int, int], np.ndarray]) -> tuple[int, int] | None:
    """Pair with the best mean / std of yearly returns."""
    best, best_score = None, -np.inf
    for pair, yr in yearly.items():
        if len(yr) > 1:
            sd = yr.std(ddof=1)
            if sd > 0:
                score = yr.mean() / sd
                if score > best_score:
                    best, best_score = pair, score
    return best


# ---------------------------------------------------------------
# Section 2: DataFrame for charts, trades and metrics
# ---------------------------------------------------------------
def section2_frame(prices: pd.DataFrame, ema_len: int, sma_len: int, cfg: Config, capital: float,
                   start: date) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    close = prices["Close"].to_numpy(dtype=float)
    i0 = int(prices.index.searchsorted(pd.Timestamp(start)))
    if len(close) - i0 < 2:
        raise UserError("Not enough bars inside the backtest range.")

    nt_bits, indicators = no_trade_parts(prices, cfg)
    filt = nt_bits != 0
    off = off_timeline_mask(prices.index, cfg)
    blocked = filt | off                       # engine: flat and no entries in either kind of zone
    all_bits = nt_bits | off.astype(np.int64) * NT_TIMELINE
    ema, sma = ema_values(close, ema_len), sma_values(close, sma_len)
    r = compute_pair(close, ema, sma, blocked, i0, cfg, capital, all_bits)

    w = slice(i0, None)
    sig = pd.DataFrame({
        "Close": close[w], "EMA": ema[w], "SMA": sma[w],
        "above": r["above"], "cross_up": r["cross_up"], "cross_dn": r["cross_dn"],
        "buy": r["buy"], "sell": r["sell"], "no_trade": filt[w], "off_timeline": off[w], "exit_why": r["why"],
        "position": r["pos"], "ret": r["ret"], "gross_ret": r["gross_ret"], "strat_ret": r["strat_ret"],
        "equity": r["equity"], "equity_gross": r["equity_gross"],
        "buyhold": capital * np.cumprod(1.0 + r["ret"]),
    }, index=prices.index[w])
    return sig, indicators


def exit_reason(code: int, side: int, entry_price: float, exit_price: float, stop_pct: float) -> str:
    """Plain-English condition that closed a trade (see _positions for the codes)."""
    down = side == 1                               # a long exits on a down-cross, a short on an up-cross
    if code == 1:
        level = entry_price * (1 - stop_pct / 100) if down else entry_price * (1 + stop_pct / 100)
        return (f"Stop loss {stop_pct:g}%: close {exit_price:,.2f} {'<=' if down else '>='} "
                f"stop level {level:,.2f}")
    if code == 2:
        return "EMA crossed below SMA" if down else "EMA crossed above SMA"
    if code == 3:
        return ("EMA crossed below SMA and close < EMA (exit filter)" if down
                else "EMA crossed above SMA and close > EMA (exit filter)")
    if code == 4:
        return ("Earlier EMA down-cross confirmed: close fell below EMA" if down
                else "Earlier EMA up-cross confirmed: close rose above EMA")
    if code >= 1000:
        bits = code - 1000
        names = [name for bit, name in NT_NAMES.items() if bits & bit]
        return ("No-trade zone: " if bits & ~NT_TIMELINE else "") + " + ".join(names) if names else "No-trade zone"
    return "Position closed"


def build_trades(sig: pd.DataFrame, cost_pct: float, stop_pct: float = 0.0) -> pd.DataFrame:
    """Trade log. Entry/exit prices are the close of the signal bar (bar t),
    matching the equity curve, which earns from bar t+1. 'Exit Reason' names the condition that
    closed the trade (needs the exit_why column from section2_frame)."""
    round_trip_cost = 2 * cost_pct
    trades: list[dict] = []
    side, entry_price, entry_date, prev = 0, np.nan, None, 0

    def record(exit_date, exit_price, status, reason):
        gross = side * (exit_price / entry_price - 1) * 100
        trades.append({
            "Side": "Long" if side == 1 else "Short", "Entry": entry_date, "Exit": exit_date,
            "Entry Price": entry_price, "Exit Price": exit_price,
            "Gross Return %": gross, "Net Return %": gross - round_trip_cost, "Status": status,
            "Exit Reason": reason,
        })

    why = sig["exit_why"].to_numpy() if "exit_why" in sig else np.zeros(len(sig), dtype=np.int64)
    for dt, pos, close, code in zip(sig.index, sig["position"].to_numpy(), sig["Close"].to_numpy(), why):
        if pos != prev:
            if prev != 0:
                record(dt, close, "Closed", exit_reason(int(code), side, entry_price, close, stop_pct))
            if pos != 0:
                side, entry_price, entry_date = pos, close, dt
        prev = pos
    if prev != 0:
        record(sig.index[-1], sig["Close"].iloc[-1], "Open", "Still open at the end of the data")
    return pd.DataFrame(trades)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def avg_annual_return(equity: pd.Series, capital: float) -> float:
    """Annualised (compound) return in %."""
    if len(equity) < 2:
        return 0.0
    years = _years(equity.index[0], equity.index[-1])
    return _annual_return(equity.iloc[-1], capital, years)


def max_drawdown(equity: pd.Series) -> float:
    return _max_drawdown(equity.to_numpy())


def trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    keys = ["n_trades", "n_long", "n_short", "win_rate", "profit_factor",
            "avg_win", "avg_loss", "expectancy", "payoff", "best", "worst"]
    if trades.empty:
        return {k: (0 if k.startswith("n_") else np.nan) for k in keys}

    t = trades["Net Return %"]
    wins, losses = t[t > 0], t[t <= 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    return {
        "n_trades": len(trades),
        "n_long": int((trades["Side"] == "Long").sum()),
        "n_short": int((trades["Side"] == "Short").sum()),
        "win_rate": (t > 0).mean() * 100,
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() < 0 else np.nan,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": t.mean(),
        "payoff": avg_win / abs(avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else np.nan,
        "best": t.max(),
        "worst": t.min(),
    }


def compute_metrics(sig: pd.DataFrame, trades: pd.DataFrame, capital: float, rf_pct: float,
                    ppy: float = TRADING_DAYS) -> dict[str, float]:
    equity, ret = sig["equity"], sig["strat_ret"]
    m: dict[str, float] = {}
    m["total_ret"] = (equity.iloc[-1] / capital - 1) * 100
    m["cagr"] = avg_annual_return(equity, capital)
    m["mdd"] = max_drawdown(equity)

    excess = ret - rf_pct / 100.0 / ppy
    std = ret.std()
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())      # downside deviation vs risk-free
    m["sharpe"] = excess.mean() / std * np.sqrt(ppy) if pd.notna(std) and std > 0 else np.nan
    m["sortino"] = excess.mean() / downside * np.sqrt(ppy) if downside > 0 else np.nan
    m["vol"] = std * np.sqrt(ppy) * 100 if pd.notna(std) else np.nan
    m["calmar"] = m["cagr"] / abs(m["mdd"]) if m["mdd"] < 0 else np.nan
    m["exposure"] = (sig["position"] != 0).mean() * 100

    gross_pnl = sig["equity_gross"].iloc[-1] - capital
    m["cost_amt"] = gross_pnl - (equity.iloc[-1] - capital)
    m["cost_pct_profit"] = m["cost_amt"] / gross_pnl * 100 if gross_pnl > 0 else np.nan

    m.update(trade_metrics(trades))

    crosses = sig["cross_up"] | sig["cross_dn"]
    m["n_cross"] = int(crosses.sum())
    m["n_cross_nz"] = int((crosses & sig["no_trade"]).sum())
    m["cross_nz_ratio"] = m["n_cross_nz"] / m["n_cross"] * 100 if m["n_cross"] > 0 else np.nan
    m["nz_time"] = sig["no_trade"].mean() * 100
    m["tl_time"] = sig["off_timeline"].mean() * 100 if "off_timeline" in sig else 0.0
    return m


def fmt(x: float | None, suffix: str = "", decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{decimals}f}{suffix}"


# ---------------------------------------------------------------
# Portfolio: equal capital per stock, one independent strategy sleeve per stock
# ---------------------------------------------------------------
def portfolio_label(tickers: tuple[str, ...]) -> str:
    return tickers[0] if len(tickers) == 1 else f"Portfolio of {len(tickers)} stocks"


MARKETS = {"NSE (.NS)": ".NS", "BSE (.BO)": ".BO", "As typed (no suffix)": ""}


def parse_tickers(text: str, suffix: str = "") -> tuple[str, ...]:
    """'reliance, tcs .NS' -> ('RELIANCE.NS', 'TCS.NS'): upper-cased, a stray '.NS' joined to the name before it,
    `suffix` added to names that have no dot, duplicates removed, order kept."""
    merged: list[str] = []
    for tok in (t for t in re.split(r"[,;\s]+", text.upper()) if t and t != "."):
        if tok.startswith("."):
            if merged and "." not in merged[-1]:
                merged[-1] += tok
            continue                                   # a suffix with nothing to attach to is ignored
        merged.append(tok)
    out = [t + suffix if suffix and "." not in t and not t.startswith("^") and "=" not in t and "-" not in t else t
           for t in merged]
    return tuple(dict.fromkeys(out))


def current_tickers() -> tuple[str, ...]:
    ss = st.session_state
    return parse_tickers(ss.get("sb_tickers", SIDEBAR_DEFAULTS["sb_tickers"]),
                         MARKETS.get(ss.get("sb_market", SIDEBAR_DEFAULTS["sb_market"]), ""))


def combine_sleeves(union: pd.DatetimeIndex, parts: list[dict], capital: float) -> dict:
    """Add the stocks' equity curves on the portfolio calendar. Every part holds `index`, `alloc`, `equity`,
    `equity_gross`, `position` (and optionally `buyhold`) for one stock. The portfolio is not rebalanced:
    each stock compounds its own share of the capital and portfolio equity is their sum."""
    n_bars = len(union)
    eq, eqg, bh = np.zeros(n_bars), np.zeros(n_bars), np.zeros(n_bars)
    invested = np.zeros(n_bars)
    sleeve_eq = np.zeros((len(parts), n_bars))
    for k, part in enumerate(parts):
        take = carry_map(part["index"], union)
        e = carry(part["equity"], take, part["alloc"])
        eq += e
        eqg += carry(part["equity_gross"], take, part["alloc"])
        if "buyhold" in part:
            bh += carry(part["buyhold"], take, part["alloc"])
        invested += e * (carry(part["position"], take, 0) != 0)
        sleeve_eq[k] = e
    return {"equity": eq, "equity_gross": eqg, "buyhold": bh, "strat_ret": returns_from_equity(eq, capital),
            "invested": invested / eq, "exposure": float(np.mean(invested / eq)) * 100.0, "sleeve_eq": sleeve_eq}


def portfolio_frame(union: pd.DatetimeIndex, comb: dict) -> pd.DataFrame:
    """Portfolio-level frame with the columns compute_metrics and the charts expect."""
    flat = np.zeros(len(union), dtype=bool)
    return pd.DataFrame({
        "equity": comb["equity"], "equity_gross": comb["equity_gross"], "buyhold": comb["buyhold"],
        "strat_ret": comb["strat_ret"], "position": (comb["invested"] > 0).astype(int),
        "cross_up": flat, "cross_dn": flat, "no_trade": flat, "off_timeline": flat,
    }, index=union)


def portfolio_metrics(psig: pd.DataFrame, trades: pd.DataFrame, capital: float, rf_pct: float, ppy: float,
                      comb: dict, per_stock: list[dict]) -> dict[str, float]:
    """Portfolio formulas: return, volatility, Sharpe, Sortino, drawdown, Calmar come from the combined
    equity curve (so diversification is reflected); trade statistics pool every stock's trades;
    exposure is the share of portfolio equity sitting in open positions; crossover counts are summed over
    stocks and the no-trade / timeline time shares are averaged over stocks."""
    m = compute_metrics(psig, trades, capital, rf_pct, ppy)
    m["exposure"] = comb["exposure"]
    m["n_cross"] = int(sum(x["n_cross"] for x in per_stock))
    m["n_cross_nz"] = int(sum(x["n_cross_nz"] for x in per_stock))
    m["cross_nz_ratio"] = m["n_cross_nz"] / m["n_cross"] * 100 if m["n_cross"] > 0 else np.nan
    m["nz_time"] = float(np.mean([x["nz_time"] for x in per_stock]))
    m["tl_time"] = float(np.mean([x["tl_time"] for x in per_stock]))
    return m


def portfolio_stats(comb: dict, tickers: list[str], allocs: np.ndarray, capital: float, ppy: float) -> dict:
    """Diversification statistics from each stock's equity on the portfolio calendar."""
    sleeve_eq, port_ret = comb["sleeve_eq"], comb["strat_ret"]
    n, n_bars = sleeve_eq.shape
    prev = np.concatenate((allocs[:, None], sleeve_eq[:, :-1]), axis=1)
    sleeve_ret = sleeve_eq / prev - 1.0
    # Contribution of stock i to the portfolio return on a bar: (change in its equity) / (portfolio equity before)
    prev_total = np.concatenate(([capital], comb["equity"][:-1]))
    contrib = (sleeve_eq - prev) / prev_total
    ann = np.sqrt(ppy) * 100.0
    vol_i = sleeve_ret.std(axis=1, ddof=1) * ann if n_bars > 1 else np.full(n, np.nan)
    vol_p = port_ret.std(ddof=1) * ann if n_bars > 1 else np.nan
    weights = allocs / allocs.sum()
    wavg_vol = float((weights * vol_i).sum())
    var_p = port_ret.var(ddof=1) if n_bars > 2 else np.nan
    if pd.notna(var_p) and var_p > 0:
        cov = ((contrib - contrib.mean(axis=1, keepdims=True)) * (port_ret - port_ret.mean())).sum(axis=1) / (n_bars - 1)
        risk_share = cov / var_p * 100.0
    else:
        risk_share = np.full(n, np.nan)
    corr = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    avg_corr = np.nan
    if n > 1 and n_bars > 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = pd.DataFrame(np.corrcoef(sleeve_ret), index=tickers, columns=tickers)
        off_diag = corr.to_numpy()[~np.eye(n, dtype=bool)]
        avg_corr = float(np.nanmean(off_diag)) if np.isfinite(off_diag).any() else np.nan
    return {"vol_i": vol_i, "wavg_vol": wavg_vol,
            "div_ratio": wavg_vol / vol_p if pd.notna(vol_p) and vol_p > 0 else np.nan,
            "risk_share": risk_share, "corr": corr, "avg_corr": avg_corr}


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
# Sidebar widgets that "copy top row" can fill: permutation parameter -> widget key
SIDEBAR_KEYS = {
    "ema_len": "sb_man_ema", "sma_len": "sb_man_sma", "stop_pct": "sb_stop_pct",
    "allow_short": "sb_allow_short", "exit_filter": "sb_exit_filter",
    "use_er": "sb_use_er", "er_n": "sb_er_n", "er_min": "sb_er_min",
    "use_adx": "sb_use_adx", "adx_n": "sb_adx_n", "adx_min": "sb_adx_min",
    "use_vi_cross": "sb_use_vi_cross", "use_vi_band": "sb_use_vi_band", "vi_n": "sb_vi_n",
    "vi_lo": "sb_vi_lo", "vi_hi": "sb_vi_hi", "cost_pct": "sb_cost_pct",
}
SIDEBAR_DEFAULTS = {
    "sb_use_estimated": True, "sb_man_ema": 20, "sb_man_sma": 50, "sb_stop_pct": 0.0,
    "sb_allow_short": True, "sb_exit_filter": True, "sb_use_er": True, "sb_er_n": 10, "sb_er_min": 0.25,
    "sb_use_adx": False, "sb_adx_n": 14, "sb_adx_min": 20.0, "sb_use_vi_cross": False,
    "sb_use_vi_band": False, "sb_vi_n": 14, "sb_vi_lo": 0.9, "sb_vi_hi": 1.1, "sb_cost_pct": 0.05,
    "sb_method": METHOD_NAMES[0], "sb_tickers": "RELIANCE", "sb_market": "NSE (.NS)",
}
# Every widget that belongs to a ticker's own profile
PROFILE_KEYS = list(SIDEBAR_KEYS.values()) + ["sb_use_estimated", "sb_method"]


def ensure_defaults() -> None:
    for k, v in SIDEBAR_DEFAULTS.items():          # defaults live in session_state so they can be overwritten
        st.session_state.setdefault(k, v)


def profile_capture() -> dict:
    """Values the sidebar profile widgets currently hold."""
    return {k: st.session_state[k] for k in PROFILE_KEYS if k in st.session_state}


def profile_load(values: dict) -> None:
    for k, v in values.items():
        st.session_state[k] = v


def sync_profiles(tickers: tuple[str, ...]) -> None:
    """Keep one stored profile per ticker. Runs before the sidebar is drawn. A new ticker starts as a copy of the
    values currently in the sidebar; a removed ticker's profile is dropped; the sidebar always shows the profile of
    the ticker in `profile_loaded`."""
    ss = st.session_state
    ensure_defaults()
    store = ss.setdefault("profiles", {})
    loaded = ss.get("profile_loaded")
    current = profile_capture()
    if loaded in store:
        store[loaded] = current
    for t in tickers:
        store.setdefault(t, dict(current))
    for t in [t for t in store if t not in tickers]:
        del store[t]
    if not tickers:
        return
    if loaded not in tickers:
        profile_load(store[tickers[0]])
        ss["profile_loaded"] = tickers[0]
    if ss.get("active_ticker") not in tickers:
        ss["active_ticker"] = ss["profile_loaded"]


def switch_profile() -> None:
    """Callback of the ticker switcher: store the sidebar values under the old ticker, load the new one's."""
    ss = st.session_state
    store, prev, new = ss["profiles"], ss.get("profile_loaded"), ss["active_ticker"]
    if prev in store:
        store[prev] = profile_capture()
    profile_load(store[new])
    ss["profile_loaded"] = new


def copy_profile_to_all() -> None:
    ss = st.session_state
    current = profile_capture()
    for t in ss["profiles"]:
        ss["profiles"][t] = dict(current)


def profile_from_values(v: dict, timeline: tuple) -> Profile:
    g = lambda k: v.get(k, SIDEBAR_DEFAULTS[k])
    cfg = Config(
        allow_short=bool(g("sb_allow_short")), exit_filter=bool(g("sb_exit_filter")), stop_pct=float(g("sb_stop_pct")),
        use_er=bool(g("sb_use_er")), er_n=int(g("sb_er_n")), er_min=float(g("sb_er_min")),
        use_adx=bool(g("sb_use_adx")), adx_n=int(g("sb_adx_n")), adx_min=float(g("sb_adx_min")),
        use_vi_cross=bool(g("sb_use_vi_cross")), use_vi_band=bool(g("sb_use_vi_band")), vi_n=int(g("sb_vi_n")),
        vi_lo=float(g("sb_vi_lo")), vi_hi=float(g("sb_vi_hi")), cost_pct=float(g("sb_cost_pct")), timeline=timeline)
    method = g("sb_method") if g("sb_method") in METHOD_NAMES else METHOD_NAMES[0]
    return Profile(cfg, int(g("sb_man_ema")), int(g("sb_man_sma")), bool(g("sb_use_estimated")), method)


def restrict_inputs(inp: Inputs, tickers: tuple[str, ...]) -> Inputs:
    """The same inputs for a subset of the tickers (each keeps its own profile)."""
    if tuple(tickers) == inp.tickers:
        return inp
    return replace(inp, tickers=tuple(tickers), profiles={t: inp.profiles[t] for t in tickers})


def inputs_changed(old: Inputs, new: Inputs) -> bool:
    """Have the settings behind a stored result changed? Compared as plain data (asdict), because Streamlit
    re-creates the dataclass types on every rerun, so instances from an earlier run never compare equal."""
    if not set(old.tickers) <= set(new.tickers):
        return True
    return asdict(canon_inputs(old)) != asdict(canon_inputs(restrict_inputs(new, old.tickers)))


def canon_inputs(inp: Inputs) -> Inputs:
    """Inputs without the 'currently selected ticker' fields, so switching tickers is not a settings change."""
    if not inp.tickers or inp.tickers[0] not in inp.profiles:
        return inp
    prof = inp.profiles[inp.tickers[0]]
    return replace(inp, cfg=prof.cfg, man_ema=prof.man_ema, man_sma=prof.man_sma,
                   use_estimated=prof.use_estimated, method=prof.method, active="")


def sidebar_inputs() -> Inputs:
    sb = st.sidebar
    ensure_defaults()
    sb.caption("Build: permutations + stop loss + copy row + timeframe + portfolio + ticker profiles")

    sb.header("Data")
    sb.text_area(
        "Tickers (one per line or comma separated)", key="sb_tickers", height=110,
        help="Add as many stocks as you like, for example RELIANCE, TCS, INFY. Every ticker gets its own parameter "
             "profile (switch between them in the 'Ticker profiles' section at the top of the page). The backtest "
             "is a portfolio: the capital is split equally and every stock trades its own profile.")
    sb.selectbox(
        "Market (suffix added to tickers that have none)", list(MARKETS), key="sb_market",
        help="Yahoo Finance needs .NS for NSE and .BO for BSE stocks. The suffix is added to every ticker that has "
             "no dot in it. Choose 'As typed' to use the symbols exactly as written (for example AAPL). "
             "A stray '.NS' typed after a name is joined to it.")
    tickers = current_tickers()
    if tickers:
        sb.caption("Using: " + ", ".join(tickers[:12]) + (f" ... (+{len(tickers) - 12} more)" if len(tickers) > 12 else ""))
    capital = sb.number_input("Initial capital (split equally across the stocks)", value=100000.0, step=10000.0)
    if len(tickers) > 1:
        sb.caption(f"{len(tickers)} stocks: {capital / len(tickers):,.0f} capital per stock.")
    interval = sb.selectbox(
        "Timeframe (bar size)", list(INTERVALS), index=list(INTERVALS).index("1d"),
        format_func=lambda k: INTERVALS[k][0], key="sb_interval",
        help="Size of each price bar. All lengths (EMA, SMA, ER, ADX, Vortex) are counted in bars of this size. "
             "Yahoo Finance only keeps recent history for intraday bars: 1 minute about 30 days, 5/15/30 minutes "
             "about 60 days, 1 hour about 2 years.")
    earliest = earliest_allowed(interval)
    if earliest is not None:
        sb.caption(f"{INTERVALS[interval][0]} bars: Yahoo data starts {earliest}. Use dates from then on. "
                   "Annualised figures are extrapolated from a short window; compare runs on total return "
                   "and max drawdown.")

    sb.header("Section 1: Historical training dates")
    use_train = sb.checkbox("Enable section 1 (use to estimate best parameters)", value=True)
    train_start = sb.date_input("Training start", date(2010, 1, 1), key="ts", disabled=not use_train)
    train_end = sb.date_input("Training end", date(2020, 1, 1), key="te", disabled=not use_train)

    sb.header("Section 2: Backtest dates")
    test_start = sb.date_input("Backtest start", date(2020, 1, 1), key="bs")
    test_end = sb.date_input("Backtest end", date(2025, 1, 1), key="be")

    sb.header("Trading timeline")
    use_tl = sb.checkbox(
        "Trade only inside chosen date ranges", value=False,
        help="Outside the ranges the strategy is flat: a running trade is closed at the first bar outside "
             "and no new entries are taken. Applies to the section 2 backtest and to Permutations. "
             "Section 1 (parameter estimation) still uses its full training dates.")
    tl_edit = sb.data_editor(
        pd.DataFrame({"From": [test_start], "To": [test_end]}), num_rows="dynamic", hide_index=True,
        disabled=not use_tl, key="timeline_editor",
        column_config={"From": st.column_config.DateColumn("From", format="YYYY-MM-DD"),
                       "To": st.column_config.DateColumn("To", format="YYYY-MM-DD")})
    timeline: tuple = ()
    timeline_error = ""
    if use_tl:
        pairs = [(pd.Timestamp(a).date(), pd.Timestamp(b).date())
                 for a, b in zip(tl_edit["From"], tl_edit["To"]) if pd.notna(a) and pd.notna(b)]
        timeline = tuple(pairs)
        if not pairs:
            timeline_error = "The trading timeline is on but has no complete From / To row. Add a range or turn it off."
    sb.caption("Add or delete rows to use several ranges.")

    active = st.session_state.get("profile_loaded") or (tickers[0] if tickers else "")
    if len(tickers) > 1:
        sb.info(f"The parameters below (EMA/SMA source and lengths, stop loss, entry/exit rules, filters, cost) "
                f"belong to **{active}**. Switch ticker in the 'Ticker profiles' section at the top of the page.")
    sb.header("Parameter source")
    use_estimated = sb.toggle("Use section 1 to estimate EMA/SMA values", key="sb_use_estimated")
    method = sb.selectbox("Estimation method", METHOD_NAMES, key="sb_method", disabled=not use_estimated)

    sb.header("Manual parameters")
    man_ema = sb.number_input("EMA length", 2, 300, key="sb_man_ema")
    man_sma = sb.number_input("SMA length", 2, MAX_SMA_LEN, key="sb_man_sma")

    sb.header("Stop loss")
    stop_pct = sb.number_input(
        "Stop loss (% from entry, 0 = off)", 0.0, 50.0, step=0.5, key="sb_stop_pct",
        help="Checked on bar closes: a long exits at the first close at or below entry x (1 - stop), "
             "a short at the first close at or above entry x (1 + stop). The exit is that bar's close, "
             "so it can be worse than the stop level after a gap. No re-entry until the next crossover.")

    sb.header("Entry / exit rules")
    allow_short = sb.checkbox("Enable sell (short) entries: EMA < SMA", key="sb_allow_short")
    exit_filter = sb.checkbox(
        "Exit filter: long exits only if close < EMA, short exits only if close > EMA", key="sb_exit_filter")

    sb.header("No trading zone (sideways market)")
    use_er = sb.checkbox("Efficiency Ratio filter (no trade when ER < threshold)", key="sb_use_er")
    er_n = sb.number_input("ER lookback N", 2, 200, key="sb_er_n", disabled=not use_er)
    er_min = sb.slider("ER threshold", 0.0, 1.0, step=0.05, key="sb_er_min", disabled=not use_er)
    use_adx = sb.checkbox("ADX filter (no trade when ADX < threshold)", key="sb_use_adx")
    adx_n = sb.number_input("ADX period", 2, 100, key="sb_adx_n", disabled=not use_adx)
    adx_min = sb.slider("ADX threshold", 5.0, 50.0, step=0.5, key="sb_adx_min", disabled=not use_adx)
    with sb.expander("Vortex filters (optional)"):
        vi_n = st.number_input("Vortex lookback period", 2, 100, key="sb_vi_n")
        use_vi_cross = st.checkbox("No trade when +VI < -VI", key="sb_use_vi_cross")
        use_vi_band = st.checkbox("No trade when VI is inside band", key="sb_use_vi_band")
        vi_lo = st.number_input("Band lower bound", step=0.05, key="sb_vi_lo", disabled=not use_vi_band)
        vi_hi = st.number_input("Band upper bound", step=0.05, key="sb_vi_hi", disabled=not use_vi_band)

    sb.header("Costs and risk")
    cost_pct = sb.number_input(
        "Trading cost per side (% of trade value)", 0.0, 5.0, step=0.01, key="sb_cost_pct",
        help="Commission + slippage charged on every entry and exit. A flip costs two sides.")
    rf_pct = sb.number_input("Risk-free rate (% per year, for Sharpe/Sortino)", 0.0, 20.0, 0.0, step=0.25)

    sb.header("Heat map pair ranges")
    ema_lo, ema_hi = sb.slider("EMA range", 5, 100, (5, 50))
    ema_step = sb.number_input("EMA step", 1, 50, 5)
    sma_lo, sma_hi = sb.slider("SMA range", 10, 250, (20, 100))
    sma_step = sb.number_input("SMA step", 1, 50, 10)

    cfg = Config(
        allow_short=allow_short, exit_filter=exit_filter, stop_pct=float(stop_pct),
        use_er=use_er, er_n=int(er_n), er_min=float(er_min),
        use_adx=use_adx, adx_n=int(adx_n), adx_min=float(adx_min),
        use_vi_cross=use_vi_cross, use_vi_band=use_vi_band, vi_n=int(vi_n),
        vi_lo=float(vi_lo), vi_hi=float(vi_hi), cost_pct=float(cost_pct), timeline=timeline,
    )
    store = st.session_state.setdefault("profiles", {})
    if active in tickers:
        store[active] = profile_capture()
    profiles = {t: profile_from_values(store.get(t, {}), timeline) for t in tickers}
    return Inputs(
        tickers=tickers, capital=float(capital), rf_pct=float(rf_pct),
        use_train=use_train, train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end, use_estimated=use_estimated, method=method,
        man_ema=int(man_ema), man_sma=int(man_sma),
        emas=list(range(int(ema_lo), int(ema_hi) + 1, int(ema_step))),
        smas=list(range(int(sma_lo), int(sma_hi) + 1, int(sma_step))),
        cfg=cfg, timeline_error=timeline_error, interval=interval, profiles=profiles, active=active,
    )


def timeline_problems(inp: Inputs, start: date, end: date) -> list[str]:
    out = [inp.timeline_error] if inp.timeline_error else []
    tl = inp.cfg.timeline
    out += [f"Timeline range {a} to {b} has its From date after its To date." for a, b in tl if a > b]
    if tl and not any(a <= b and a <= end and b >= start for a, b in tl):
        out.append("The trading timeline does not overlap the evaluation dates, so nothing could trade.")
    return out


def date_problems(inp: Inputs) -> list[str]:
    problems = []
    if not inp.tickers:
        problems.append("Enter at least one ticker.")
    if inp.test_start >= inp.test_end or (inp.use_train and inp.train_start >= inp.train_end):
        problems.append("Start date must be before end date in each enabled section.")
    problems += interval_problems(inp.interval, inp.test_start, inp.test_end)
    if inp.use_train:
        problems += [f"Section 1: {x}" for x in interval_problems(inp.interval, inp.train_start, inp.train_end)]
    problems += timeline_problems(inp, inp.test_start, inp.test_end)
    return problems


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------
def no_trade_spans(sig: pd.DataFrame, col: str = "no_trade") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of each consecutive run where sig[col] is True."""
    flag = sig[col].astype(int)
    prev = flag.shift(1, fill_value=0)
    starts = sig.index[(flag == 1) & (prev == 0)]
    ends = sig.index[(flag == 0) & (prev == 1)]
    if len(starts) > len(ends):
        ends = ends.append(pd.DatetimeIndex([sig.index[-1]]))
    return list(zip(starts, ends))


def price_figure(sig: pd.DataFrame, ind: dict[str, pd.Series], cfg: Config,
                 ema_len: int, sma_len: int) -> go.Figure:
    """Price with EMA/SMA/trades on top; every enabled filter indicator below on the same x-axis."""
    panels = []
    if "ER" in ind:
        panels.append("Efficiency Ratio")
    if "ADX" in ind:
        panels.append("ADX")
    if "+VI" in ind:
        panels.append("Vortex")

    n_rows = 1 + len(panels)
    heights = [1.0] if not panels else [0.5] + [0.5 / len(panels)] * len(panels)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=heights, subplot_titles=["Price"] + panels)

    prev_pos = sig["position"].shift(1).fillna(0)
    long_in = sig[(sig["position"] == 1) & (prev_pos != 1)]
    short_in = sig[(sig["position"] == -1) & (prev_pos != -1)]
    exits = sig[(sig["position"] == 0) & (prev_pos != 0)]

    fig.add_trace(go.Scatter(x=sig.index, y=sig["Close"], name="Close", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig.index, y=sig["EMA"], name=f"EMA {ema_len}", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig.index, y=sig["SMA"], name=f"SMA {sma_len}", line=dict(color="#ff7f0e")), row=1, col=1)
    for frame, name, symbol, color, size in [
        (long_in, "Buy entry", "triangle-up", "green", 11),
        (short_in, "Sell entry", "triangle-down", "red", 11),
        (exits, "Exit", "x", "black", 8),
    ]:
        fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"], mode="markers", name=name,
                                 marker=dict(symbol=symbol, color=color, size=size)), row=1, col=1)

    row = 1
    if "ER" in ind:
        row += 1
        fig.add_trace(go.Scatter(x=sig.index, y=ind["ER"].reindex(sig.index), name="ER", showlegend=False), row=row, col=1)
        fig.add_hline(y=cfg.er_min, line_dash="dot", line_color="red", row=row, col=1)
    if "ADX" in ind:
        row += 1
        fig.add_trace(go.Scatter(x=sig.index, y=ind["ADX"].reindex(sig.index), name="ADX", showlegend=False), row=row, col=1)
        fig.add_hline(y=cfg.adx_min, line_dash="dot", line_color="red", row=row, col=1)
    if "+VI" in ind:
        row += 1
        fig.add_trace(go.Scatter(x=sig.index, y=ind["+VI"].reindex(sig.index), name="+VI"), row=row, col=1)
        fig.add_trace(go.Scatter(x=sig.index, y=ind["-VI"].reindex(sig.index), name="-VI"), row=row, col=1)
        if cfg.use_vi_band:
            for level in (cfg.vi_lo, cfg.vi_hi):
                fig.add_hline(y=level, line_dash="dot", line_color="grey", row=row, col=1)

    # One full-height rectangle per no-trade run (yref="paper" spans every panel).
    # Building the shapes list directly is far faster than add_vrect(row="all").
    shapes = [dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1,
                   fillcolor="grey", opacity=0.15, line_width=0, layer="below")
              for a, b in no_trade_spans(sig)]
    if "off_timeline" in sig and sig["off_timeline"].any():
        shapes += [dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1,
                        fillcolor="royalblue", opacity=0.10, line_width=0, layer="below")
                   for a, b in no_trade_spans(sig, "off_timeline")]
    fig.update_layout(shapes=shapes, height=420 + 170 * len(panels), margin=CHART_MARGIN, hovermode="x unified")
    return fig


def equity_figure(sig: pd.DataFrame, title: str = "Equity curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig.index, y=sig["equity"], name="Strategy (net of costs)"))
    fig.add_trace(go.Scatter(x=sig.index, y=sig["equity_gross"], name="Strategy (gross)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=sig.index, y=sig["buyhold"], name="Buy and hold"))
    fig.update_layout(height=350, margin=CHART_MARGIN, title=title)
    return fig


def drawdown_figure(sig: pd.DataFrame) -> go.Figure:
    drawdown = (sig["equity"] / sig["equity"].cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line=dict(color="crimson")))
    fig.update_layout(height=250, margin=CHART_MARGIN, title="Drawdown %")
    return fig


def sleeves_figure(sleeve_equity: pd.DataFrame, allocs: np.ndarray, psig: pd.DataFrame, capital: float) -> go.Figure:
    """Growth of each stock's own capital and of the whole portfolio, all starting at 100."""
    fig = go.Figure()
    for col, a in zip(sleeve_equity.columns, allocs):
        fig.add_trace(go.Scatter(x=sleeve_equity.index, y=sleeve_equity[col] / a * 100, name=col, line=dict(width=1)))
    fig.add_trace(go.Scatter(x=psig.index, y=psig["equity"] / capital * 100, name="Portfolio",
                             line=dict(width=3, color="black")))
    fig.update_layout(height=380, margin=CHART_MARGIN, title="Growth of each stock's capital (start = 100)")
    return fig


def corr_figure(corr: pd.DataFrame) -> go.Figure:
    z = corr.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=[["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
        colorbar=dict(title="Correlation")))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_layout(height=max(320, 50 * len(corr) + 140), margin=CHART_MARGIN,
                      title="Correlation of the stocks' strategy returns")
    return fig


def contribution_figure(table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Contribution to portfolio return (pp)",
                                                        "Share of portfolio risk (%)"))
    fig.add_trace(go.Bar(x=table["Stock"], y=table["Contribution to portfolio return (pp)"], showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=table["Stock"], y=table["Share of portfolio risk %"], showlegend=False), row=1, col=2)
    fig.update_layout(height=340, margin=CHART_MARGIN)
    return fig


def heatmap_figure(grid: pd.DataFrame, colorbar_title: str, zmid: float | None = None,
                   decimals: int = 1) -> go.Figure:
    z = grid.to_numpy(dtype=float)
    text = [["" if np.isnan(v) else f"{v:.{decimals}f}" for v in row] for row in z]
    extra = {} if zmid is None else dict(zmid=zmid)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"SMA {c}" for c in grid.columns], y=[f"EMA {r}" for r in grid.index],
        colorscale="RdYlGn", colorbar=dict(title=colorbar_title),
        text=text, texttemplate="%{text}", **extra))
    fig.update_layout(height=500, margin=CHART_MARGIN)
    return fig


# ---------------------------------------------------------------
# Computation (runs only when the Run button is pressed)
# ---------------------------------------------------------------
def compute_all(inp: Inputs) -> Results:
    lo, hi = inp.test_start, inp.test_end
    if inp.use_train:
        lo, hi = min(lo, inp.train_start), max(hi, inp.train_end)

    spans: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for tk in inp.tickers:
        span = load_span(tk, lo, hi, inp.interval)
        if span.empty:
            skipped.append((tk, "no price data in the selected dates (check the symbol: NSE tickers end in .NS)"))
        else:
            spans[tk] = span
    if not spans:
        raise UserError(f"No {INTERVALS[inp.interval][0]} price data found for {', '.join(inp.tickers)} "
                        "in the selected dates. Check the symbols: NSE tickers end in .NS.")

    messages: list[tuple[str, str]] = []
    if inp.interval != "1d":
        messages.append(("info", f"Timeframe: {INTERVALS[inp.interval][0]} bars. Every length is counted in bars; "
                                 "annualised figures (avg annual return, Sharpe, volatility) are extrapolated "
                                 "from the selected window."))
    if inp.use_train and inp.train_end > inp.test_start:
        messages.append(("warning", "Section 1 overlaps section 2, so the backtest is not fully out of sample."))

    timeline_cfg = inp.profiles[next(iter(spans))].cfg
    if timeline_cfg.timeline:
        messages.append(("info", f"Trading timeline active in section 2: {describe_timeline(timeline_cfg)}. "
                                 "Section 1 (parameter estimation) still uses its full training dates."))

    # --- Section 1, per stock with that stock's own profile: grid and parameter estimation ---
    multi = len(spans) > 1
    grid_capital = inp.capital / len(spans)
    sections: dict[str, SectionOne] = {}
    pairs: dict[str, tuple[int, int]] = {}
    sources: dict[str, str] = {}
    for tk in spans:
        prof = inp.profiles[tk]
        tag = f"{tk}: " if multi else ""
        sec = SectionOne()
        if inp.use_train:
            sec.ret_grid, sec.dd_grid, yearly = run_grid(
                (tk,), lo, hi, inp.train_start, inp.train_end, inp.emas, inp.smas,
                replace(prof.cfg, timeline=()), grid_capital, inp.interval)
            sec.ratio_grid = sec.ret_grid / sec.dd_grid.abs().replace(0, np.nan)
            sec.methods = best_pairs(sec.ret_grid, sec.ratio_grid)
            consistent = consistency_pair(yearly)
            if consistent is not None:
                sec.methods[M_CONSISTENT] = consistent

        sec.est_pair = (prof.man_ema, prof.man_sma)
        if sec.methods:
            sec.chosen_method = (prof.method if prof.method in sec.methods
                                 else (M_PLATEAU if M_PLATEAU in sec.methods else next(iter(sec.methods))))
            sec.est_pair = sec.methods[sec.chosen_method]

        if inp.use_train and prof.use_estimated and sec.methods:
            ema_len, sma_len = int(sec.est_pair[0]), int(sec.est_pair[1])
            sources[tk] = f"Estimated from section 1 ({sec.chosen_method})"
            messages.append(("info", f"{tag}Using estimated parameters from section 1 ({sec.chosen_method}): "
                                     f"EMA {ema_len}, SMA {sma_len}"))
            if sec.chosen_method != prof.method:
                messages.append(("warning", f"{tag}'{prof.method}' was not available for this grid; "
                                            f"used '{sec.chosen_method}'."))
        else:
            ema_len, sma_len = prof.man_ema, prof.man_sma
            sources[tk] = "Manual / ticker profile"
            messages.append(("info", f"{tag}Using manual parameters: EMA {ema_len}, SMA {sma_len}"))
            if inp.use_train and prof.use_estimated and not sec.methods:
                messages.append(("warning", f"{tag}Section 1 produced no valid pairs (check dates and ranges); "
                                            "fell back to manual parameters."))
            if ema_len >= sma_len:
                messages.append(("warning", f"{tag}EMA length is not shorter than SMA length; "
                                            "the crossover logic still runs but is unusual."))
        sections[tk], pairs[tk] = sec, (ema_len, sma_len)

    # --- Section 2: out-of-sample backtest, one equal-capital sleeve per stock ---
    windows: dict[str, pd.DataFrame] = {}
    for tk, span in spans.items():
        w = price_window(span, inp.test_start, inp.test_end, inp.interval)
        i0 = int(w.index.searchsorted(pd.Timestamp(inp.test_start))) if not w.empty else 0
        if w.empty or len(w) - i0 < 2:
            skipped.append((tk, "not enough bars inside the backtest range"))
        else:
            windows[tk] = w
    if not windows:
        raise UserError("No data for the selected ticker(s) and backtest range.")
    alloc = inp.capital / len(windows)
    ppy_stock = lambda ix: periods_per_year(ix, inp.interval)

    sleeves: dict[str, Sleeve] = {}
    for tk, w in windows.items():
        cfg_t, (ema_t, sma_t) = inp.profiles[tk].cfg, pairs[tk]
        sig_i, ind_i = section2_frame(w, ema_t, sma_t, cfg_t, alloc, inp.test_start)
        trades_i = build_trades(sig_i, cfg_t.cost_pct, cfg_t.stop_pct)
        if not trades_i.empty:
            trades_i.insert(0, "Ticker", tk)
        m_i = compute_metrics(sig_i, trades_i, alloc, inp.rf_pct, ppy_stock(sig_i.index))
        sleeves[tk] = Sleeve(ticker=tk, alloc=alloc, sig=sig_i, indicators=ind_i, trades=trades_i, metrics=m_i,
                             cfg=cfg_t, ema_len=ema_t, sma_len=sma_t, source=sources[tk])

    names = list(sleeves)
    union = union_index([sl.sig.index for sl in sleeves.values()])
    parts = [{"index": sl.sig.index, "alloc": alloc, "equity": sl.sig["equity"].to_numpy(),
              "equity_gross": sl.sig["equity_gross"].to_numpy(), "position": sl.sig["position"].to_numpy(),
              "buyhold": sl.sig["buyhold"].to_numpy()} for sl in sleeves.values()]
    comb = combine_sleeves(union, parts, inp.capital)
    psig = portfolio_frame(union, comb)
    pooled = [sl.trades for sl in sleeves.values() if not sl.trades.empty]
    trades = (pd.concat(pooled, ignore_index=True).sort_values("Entry", kind="stable", ignore_index=True)
              if pooled else pd.DataFrame())
    ppy = ppy_stock(union)
    metrics = portfolio_metrics(psig, trades, inp.capital, inp.rf_pct, ppy, comb,
                                [sl.metrics for sl in sleeves.values()])

    allocs = np.full(len(names), alloc)
    stats = portfolio_stats(comb, names, allocs, inp.capital, ppy)
    metrics.update(n_stocks=len(names), div_ratio=stats["div_ratio"], avg_corr=stats["avg_corr"],
                   wavg_vol=stats["wavg_vol"])
    final_eq = comb["sleeve_eq"][:, -1]
    stock_table = pd.DataFrame({
        "Stock": names,
        "EMA length": [sl.ema_len for sl in sleeves.values()],
        "SMA length": [sl.sma_len for sl in sleeves.values()],
        "Capital allocated": allocs,
        "Final equity": final_eq,
        "Net PnL": final_eq - allocs,
        "Return %": (final_eq / allocs - 1) * 100,
        "Avg annual return %": [sl.metrics["cagr"] for sl in sleeves.values()],
        "Max drawdown %": [sl.metrics["mdd"] for sl in sleeves.values()],
        "Volatility %": stats["vol_i"],
        "Sharpe": [sl.metrics["sharpe"] for sl in sleeves.values()],
        "Trades": [sl.metrics["n_trades"] for sl in sleeves.values()],
        "Win rate %": [sl.metrics["win_rate"] for sl in sleeves.values()],
        "Contribution to portfolio return (pp)": (final_eq - allocs) / inp.capital * 100,
        "Share of portfolio risk %": stats["risk_share"],
    })
    sleeve_equity = pd.DataFrame(comb["sleeve_eq"].T, index=union, columns=names)

    if len(names) > 1:
        messages.insert(0, ("info", f"Portfolio of {len(names)} stocks: capital {inp.capital:,.0f} split equally "
                                    f"({alloc:,.0f} per stock). Every stock trades its own parameter profile "
                                    "independently; no rebalancing. Portfolio metrics use the combined equity curve."))
    for tk, why in skipped:
        messages.append(("warning", f"{tk}: {why}, so it was left out of the portfolio."))

    figs = {
        "equity": equity_figure(psig, "Portfolio equity curve" if len(names) > 1 else "Equity curve"),
        "drawdown": drawdown_figure(psig),
    }
    if len(names) > 1:
        figs["sleeves"] = sleeves_figure(sleeve_equity, allocs, psig, inp.capital)
        figs["corr"] = corr_figure(stats["corr"])
        figs["contrib"] = contribution_figure(stock_table)
    for tk, sec in sections.items():
        if sec.ret_grid is not None:
            figs[f"hm_ret:{tk}"] = heatmap_figure(sec.ret_grid, "Avg annual %", zmid=0)
            figs[f"hm_dd:{tk}"] = heatmap_figure(sec.dd_grid, "Max DD %")
            figs[f"hm_ratio:{tk}"] = heatmap_figure(sec.ratio_grid, "Return / |DD|", zmid=0, decimals=2)

    return Results(inp=inp, messages=messages, sig=psig, trades=trades, metrics=metrics, figs=figs,
                   sleeves=sleeves, sections=sections, sleeve_equity=sleeve_equity, stock_table=stock_table,
                   corr=stats["corr"], skipped=skipped)


# ---------------------------------------------------------------
# Page sections (cheap: only draw what was computed)
# ---------------------------------------------------------------
def metric_row(items: list[tuple]) -> None:
    """Render one row of st.metric; each item is (label, value[, help])."""
    for col, (label, value, *help_text) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_text[0] if help_text else None)


def render_performance(sig: pd.DataFrame, m: dict[str, float], capital: float,
                       title: str = "Strategy performance") -> None:
    st.subheader(title)
    buy_hold = (sig["buyhold"].iloc[-1] / capital - 1) * 100
    metric_row([
        ("Net PnL", f"{sig['equity'].iloc[-1] - capital:,.0f}"),
        ("Total return", fmt(m["total_ret"], "%", 1)),
        ("Avg annual return", fmt(m["cagr"], "%", 1)),
        ("Max drawdown", fmt(m["mdd"], "%", 1)),
        ("Buy and hold", f"{buy_hold:.1f}%"),
    ])
    metric_row([
        ("Win rate", fmt(m["win_rate"], "%", 1)),
        ("Sharpe ratio", fmt(m["sharpe"])),
        ("Sortino ratio", fmt(m["sortino"])),
        ("Calmar ratio", fmt(m["calmar"])),
        ("Profit factor", fmt(m["profit_factor"])),
    ])
    metric_row([
        ("Trading cost (amount)", f"{m['cost_amt']:,.0f}"),
        ("Trading cost % of profit", fmt(m["cost_pct_profit"], "%", 1),
         "Total costs / gross profit before costs. n/a when the gross result is not a profit."),
        ("Number of trades", f"{m['n_trades']}  (L {m['n_long']} / S {m['n_short']})"),
        ("Avg win / avg loss", f"{fmt(m['avg_win'], '%')} / {fmt(m['avg_loss'], '%')}"),
        ("Expectancy / trade", fmt(m["expectancy"], "%")),
    ])
    metric_row([
        ("Crossovers total", f"{m['n_cross']}"),
        ("Crossovers in no-trade zone", f"{m['n_cross_nz']}"),
        ("% crossovers in no-trade zone", fmt(m["cross_nz_ratio"], "%", 1)),
        ("Time in no-trade zone", fmt(m["nz_time"], "%", 1)),
        ("Market exposure", fmt(m["exposure"], "%", 1)),
    ])
    with st.expander("More metrics"):
        st.table(pd.DataFrame({
            "Metric": ["Annualised volatility", "Payoff ratio (avg win / |avg loss|)", "Best trade", "Worst trade",
                       "Time outside trading timeline"],
            "Value": [fmt(m["vol"], "%"), fmt(m["payoff"]), fmt(m["best"], "%"), fmt(m["worst"], "%"),
                      fmt(m["tl_time"], "%", 1)],
        }))


def render_heatmaps(res: Results, tk: str) -> None:
    inp = res.inp
    st.subheader(f"Heat maps by EMA / SMA pair ({inp.train_start} to {inp.train_end})"
                 + (f": {tk}" if len(res.sleeves) > 1 else ""))
    tab_ret, tab_dd, tab_ratio = st.tabs(["Return", "Max drawdown", "Return / drawdown"])
    with tab_ret:
        st.plotly_chart(res.figs[f"hm_ret:{tk}"], use_container_width=True)
    with tab_dd:
        st.plotly_chart(res.figs[f"hm_dd:{tk}"], use_container_width=True)
        st.caption("Closer to 0 (green) is better.")
    with tab_ratio:
        st.plotly_chart(res.figs[f"hm_ratio:{tk}"], use_container_width=True)


def render_best_pairs(res: Results, tk: str) -> None:
    sec = res.sections[tk]
    st.subheader("Best pair and why" + (f": {tk}" if len(res.sleeves) > 1 else ""))
    st.warning("The single best cell of a heat map can be a lucky spike that will not repeat out of "
               "sample. Prefer pairs sitting on a stable plateau of good neighbours.")
    rows = []
    for name, (e, s) in sec.methods.items():
        ratio = sec.ratio_grid.loc[e, s]
        rows.append({
            "Method": name, "EMA": int(e), "SMA": int(s),
            "Avg annual PnL %": round(float(sec.ret_grid.loc[e, s]), 2),
            "Max DD %": round(float(sec.dd_grid.loc[e, s]), 2),
            "Return / |DD|": round(float(ratio), 2) if pd.notna(ratio) else None,
        })
    st.table(pd.DataFrame(rows))
    st.markdown(
        "- **Highest raw return** picks the single best cell, which can be a lucky spike.\n"
        "- **Neighbourhood average** picks the pair whose surrounding pairs also perform well, "
        "a stable plateau rather than an isolated peak.\n"
        "- **Mean minus std of neighbours** rewards high returns and penalises neighbours that vary a lot.\n"
        "- **Return / drawdown** prefers pairs that earn more per unit of pain, smoothed over neighbours.\n"
        "- **Most consistent year to year** rewards pairs that earn steadily across individual years."
    )
    st.success(f"Pair used for estimation: EMA {int(sec.est_pair[0])}, SMA {int(sec.est_pair[1])} "
               f"({sec.chosen_method})")


def render_portfolio_analysis(res: Results) -> None:
    m, n = res.metrics, len(res.sleeves)
    alloc = res.inp.capital / n
    st.subheader("Portfolio analysis")
    st.caption(f"{n} stocks with {alloc:,.0f} capital each. Each stock trades the same strategy on its own capital "
               "(no rebalancing) and the portfolio equity is the sum of the stock equities. Return, volatility, "
               "Sharpe, Sortino, drawdown and Calmar above are measured on that combined equity curve, so "
               "diversification is already in them. Trade statistics pool every stock's trades.")
    metric_row([
        ("Stocks", f"{n}"),
        ("Capital per stock", f"{alloc:,.0f}"),
        ("Portfolio volatility", fmt(m["vol"], "%", 1)),
        ("Avg stock volatility", fmt(m["wavg_vol"], "%", 1), "Capital-weighted average of the stocks' own annualised volatilities."),
        ("Diversification ratio", fmt(m["div_ratio"]),
         "Average stock volatility / portfolio volatility. Above 1 means combining the stocks reduced risk."),
        ("Avg pairwise correlation", fmt(m["avg_corr"]),
         "Average correlation between the stocks' strategy returns (lower = more diversification)."),
    ])
    st.markdown("**Per stock**")
    table = res.stock_table.copy()
    num = table.select_dtypes("number").columns.difference(["Trades", "EMA length", "SMA length"])
    table[num] = table[num].astype(float).round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.plotly_chart(res.figs["sleeves"], use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(res.figs["corr"], use_container_width=True)
    c2.plotly_chart(res.figs["contrib"], use_container_width=True)
    st.caption("Share of portfolio risk = each stock's covariance with the portfolio return divided by the "
               "portfolio variance; the shares add up to 100%.")


def render_results(res: Results) -> None:
    for level, text in res.messages:
        getattr(st, level)(text)

    tickers = list(res.sleeves)
    multi = len(tickers) > 1
    sel = st.session_state.get("active_ticker")
    if sel not in res.sleeves:
        sel = tickers[0]
    sl, sec = res.sleeves[sel], res.sections.get(sel)
    if multi:
        st.info(f"Showing **{sel}** below. Pick another ticker in the 'Ticker profiles' section at the top of the page "
                "to see its chart, parameters and its own results; the portfolio figures stay the same.")

    st.subheader(f"{sel}: price, EMA, SMA, trades and filter indicators")
    key = f"price:{sel}"
    if key not in res.figs:
        res.figs[key] = price_figure(sl.sig, sl.indicators, sl.cfg, sl.ema_len, sl.sma_len)
    st.plotly_chart(res.figs[key], use_container_width=True)
    st.caption("Grey bands = no-trade zone (running trade is exited, no new entries). "
               "Blue bands = outside the trading timeline (same treatment).")
    with st.expander(f"Parameters used for {sel}"):
        shown = param_row({**asdict(sl.cfg), "ema_len": sl.ema_len, "sma_len": sl.sma_len})
        st.caption(f"EMA / SMA source: {sl.source}")
        st.table(pd.DataFrame({"Parameter": list(shown), "Value": [str(v) for v in shown.values()]}))

    if multi:
        render_performance(sl.sig, sl.metrics, sl.alloc, f"{sel}: performance on its own capital share")
        if f"eq:{sel}" not in res.figs:
            res.figs[f"eq:{sel}"] = equity_figure(sl.sig, f"{sel} equity curve")
        st.plotly_chart(res.figs[f"eq:{sel}"], use_container_width=True)
        st.divider()

    render_performance(res.sig, res.metrics, res.inp.capital,
                       "Portfolio performance" if multi else "Strategy performance")
    st.plotly_chart(res.figs["equity"], use_container_width=True)
    st.plotly_chart(res.figs["drawdown"], use_container_width=True)
    if multi:
        render_portfolio_analysis(res)

    st.subheader("Trades")
    trades = res.trades
    if multi and not trades.empty and st.checkbox(f"Only {sel}'s trades", key="bt_trades_only"):
        trades = trades[trades["Ticker"] == sel]
    if trades.empty:
        st.write("No trades generated.")
    else:
        st.dataframe(trades.round(2), use_container_width=True)

    if res.inp.use_train:
        if sec is not None and sec.ret_grid is not None and sec.methods:
            render_heatmaps(res, sel)
            render_best_pairs(res, sel)
        else:
            st.write("Not enough data to compute the best pair.")

    st.subheader("Export")
    st.caption("Parameter description and result metrics of this backtest.")
    export_buttons(backtest_export_sheets(res), f"backtest_{res.inp.file_tag}", key="bt_export")


# ---------------------------------------------------------------
# Export helpers (used by the backtest engine and by Permutations)
# ---------------------------------------------------------------
# Settings that only matter while some switch is on: key -> (is it in use?, text shown when not)
_DEPENDS = {
    "er_n": (lambda v: v["use_er"], "not used (ER filter off)"),
    "er_min": (lambda v: v["use_er"], "not used (ER filter off)"),
    "adx_n": (lambda v: v["use_adx"], "not used (ADX filter off)"),
    "adx_min": (lambda v: v["use_adx"], "not used (ADX filter off)"),
    "vi_n": (lambda v: v["use_vi_cross"] or v["use_vi_band"], "not used (Vortex filters off)"),
    "vi_lo": (lambda v: v["use_vi_band"], "not used (Vortex band off)"),
    "vi_hi": (lambda v: v["use_vi_band"], "not used (Vortex band off)"),
    "stop_pct": (lambda v: v["stop_pct"] > 0, "Off"),
}


def param_row(values: dict) -> dict:
    """Every strategy parameter as it was actually used: switches as On/Off, settings of a
    switched-off filter as 'not used', everything else as its value."""
    row: dict = {}
    for spec in PARAM_SPECS:
        v = values[spec.key]
        dep = _DEPENDS.get(spec.key)
        if dep and not dep[0](values):
            row[spec.label] = dep[1]
        elif spec.kind == "bool":
            row[spec.label] = "On" if v else "Off"
        else:
            row[spec.label] = v
    return row


def describe_values(spec: "ParamSpec", vals: list) -> str:
    items = [("On" if v else "Off") if spec.kind == "bool" else str(v) for v in vals]
    if len(items) > 25:
        return ", ".join(items[:8] + ["..."] + items[-3:]) + f" ({len(vals)} values)"
    return ", ".join(items)


def workbook_bytes(sheets: dict[str, pd.DataFrame]) -> bytes | None:
    """One Excel file with a sheet per table, or None if openpyxl is not installed."""
    try:
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            name = name[:31]
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for i, col in enumerate(df.columns, 1):
                width = max([len(str(col))] + [len(str(x)) for x in df[col].head(200)]) + 2
                ws.column_dimensions[get_column_letter(i)].width = min(width, 60)
    return buf.getvalue()


def export_buttons(sheets: dict[str, pd.DataFrame], base: str, key: str) -> None:
    data = workbook_bytes(sheets)
    if data is not None:
        c1, c2 = st.columns(2)
        c1.download_button("Download Excel (all sheets)", data, file_name=f"{base}.xlsx", key=f"{key}_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        first = next(iter(sheets))
        c2.download_button(f"Download CSV ({first} only)", sheets[first].to_csv(index=False).encode("utf-8"),
                           file_name=f"{base}.csv", mime="text/csv", key=f"{key}_csv")
    else:
        st.caption("Install openpyxl to get one Excel file with all sheets; downloading each sheet as CSV instead.")
        for i, (name, df) in enumerate(sheets.items()):
            st.download_button(f"Download CSV: {name}", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{base}_{name.lower().replace(' ', '_')}.csv", mime="text/csv",
                               key=f"{key}_csv{i}")


def perm_export_sheets(perm: dict, ids: list[int], metric_keys: list[str]) -> dict[str, pd.DataFrame]:
    """Chosen permutation rows: all parameters as used + result metrics, plus how each parameter was set up."""
    runs = perm["full"].loc[ids].reset_index(drop=True)
    runs.insert(0, "Run #", [i + 1 for i in ids])
    met = (perm["table"].loc[ids, metric_keys].astype(float).round(4)
           .rename(columns=METRIC_CATALOG).reset_index(drop=True))
    a, b = perm["window"]
    info = pd.DataFrame({
        "Item": ["Tickers", "Evaluation window", "Initial capital", "Risk-free rate (%)", "Combinations run",
                 "Runs exported", "Trading timeline", "Stop loss rule", "Timeframe", "Capital per stock"],
        "Value": [", ".join(perm["tickers"]), f"{a} to {b}", perm["capital"], perm["rf"], perm["n"], len(ids),
                  perm["timeline"], "Checked on bar closes; exit at that close",
                  INTERVALS[perm.get("interval", "1d")][0], perm["capital"] / max(1, len(perm["tickers"]))],
    })
    return {"Selected runs": pd.concat([runs, met], axis=1),
            "Parameter setup": pd.DataFrame(perm["setup"]), "Run info": info}


def backtest_export_sheets(res: Results) -> dict[str, pd.DataFrame]:
    """Parameter description and result metrics of the section 2 backtest."""
    inp, m = res.inp, res.metrics
    multi = len(res.sleeves) > 1
    first = next(iter(res.sleeves.values()))
    setup = [
        ("Tickers", ", ".join(res.sleeves)),
        ("Timeframe", INTERVALS[inp.interval][0]),
        ("Initial capital", inp.capital),
        ("Capital per stock (equal split)", inp.capital / len(res.sleeves)),
        ("Risk-free rate (%)", inp.rf_pct),
        ("Backtest dates (section 2)", f"{inp.test_start} to {inp.test_end}"),
        ("Trading timeline", describe_timeline(first.cfg)),
        ("Training dates (section 1)", f"{inp.train_start} to {inp.train_end}" if inp.use_train else "not used"),
        ("EMA / SMA source", "one profile per ticker: see the 'Ticker profiles' sheet" if multi else first.source),
    ]
    rows = [{"Group": "Setup", "Parameter": k, "Value": v} for k, v in setup]
    specs = {s.label: s for s in PARAM_SPECS}
    if not multi:
        for label, v in param_row({**asdict(first.cfg), "ema_len": first.ema_len, "sma_len": first.sma_len}).items():
            rows.append({"Group": specs[label].group, "Parameter": label, "Value": v})
        rows.append({"Group": "Entry / exit rules", "Parameter": "Stop loss rule",
                     "Value": "Checked on bar closes; exit at that close"})

    sig = res.sig
    vals = {**m, "net_pnl": sig["equity"].iloc[-1] - inp.capital,
            "buy_hold": (sig["buyhold"].iloc[-1] / inp.capital - 1) * 100}
    labels = {**METRIC_CATALOG, "buy_hold": "Buy and hold return %"}
    metrics = pd.DataFrame({
        "Metric": [labels[k] for k in labels],
        "Value": [round(float(vals[k]), 4) if pd.notna(vals.get(k, np.nan)) else None for k in labels],
    })
    sheets = {"Parameters": pd.DataFrame(rows), "Metrics": metrics}
    if multi:
        extra = pd.DataFrame({
            "Metric": ["Number of stocks", "Average stock volatility %", "Diversification ratio",
                       "Average pairwise correlation"],
            "Value": [m["n_stocks"]] + [round(float(m[k]), 4) if pd.notna(m[k]) else None
                                        for k in ("wavg_vol", "div_ratio", "avg_corr")]})
        sheets["Metrics"] = pd.concat([metrics, extra], ignore_index=True)
        sheets["Ticker profiles"] = pd.DataFrame([
            {"Ticker": tk, "EMA / SMA source": sl.source,
             **param_row({**asdict(sl.cfg), "ema_len": sl.ema_len, "sma_len": sl.sma_len})}
            for tk, sl in res.sleeves.items()])
        sheets["Stocks"] = res.stock_table.round(4)
        sheets["Correlation"] = res.corr.round(4).reset_index().rename(columns={"index": "Stock"})
    return sheets


# ---------------------------------------------------------------
# Permutations: parameter sweep (top section of the page)
# ---------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """One sweepable parameter. `bounds` limit typed values; `rng` is the default Range input;
    `all_vals` is what "All" means for numbers (default: every integer inside `bounds`)."""
    key: str
    label: str
    kind: str                                   # "int" | "float" | "bool" | "choice"
    group: str
    bounds: tuple = (0, 0)
    rng: tuple = (0, 0, 1)                      # (from, to, step)
    all_vals: tuple = ()
    choices: tuple = ()


_ER_ALL = tuple(round(0.05 * i, 2) for i in range(21))
PARAM_SPECS = [
    ParamSpec("ema_len", "EMA length", "int", "Moving averages", (2, 300), (5, 50, 5)),
    ParamSpec("sma_len", "SMA length", "int", "Moving averages", (2, MAX_SMA_LEN), (20, 100, 10)),
    ParamSpec("allow_short", "Sell (short) entries", "bool", "Entry / exit rules"),
    ParamSpec("exit_filter", "Exit filter (close vs EMA)", "bool", "Entry / exit rules"),
    ParamSpec("stop_pct", "Stop loss (% from entry, 0 = off)", "float", "Entry / exit rules", (0.0, 50.0),
              (1, 5, 1), (0, 1, 2, 3, 5, 7, 10, 15, 20)),
    ParamSpec("use_er", "Efficiency Ratio filter", "bool", "No-trade filters"),
    ParamSpec("er_n", "ER lookback", "int", "No-trade filters", (2, 200), (5, 20, 5)),
    ParamSpec("er_min", "ER threshold", "float", "No-trade filters", (0.0, 1.0), (0.1, 0.4, 0.05), _ER_ALL),
    ParamSpec("use_adx", "ADX filter", "bool", "No-trade filters"),
    ParamSpec("adx_n", "ADX period", "int", "No-trade filters", (2, 100), (10, 20, 2)),
    ParamSpec("adx_min", "ADX threshold", "float", "No-trade filters", (5, 50), (15, 30, 5),
              tuple(range(5, 51, 5))),
    ParamSpec("use_vi_cross", "Vortex: no trade when +VI < -VI", "bool", "No-trade filters"),
    ParamSpec("use_vi_band", "Vortex: no trade inside band", "bool", "No-trade filters"),
    ParamSpec("vi_n", "Vortex lookback", "int", "No-trade filters", (2, 100), (10, 20, 2)),
    ParamSpec("vi_lo", "Vortex band lower", "float", "No-trade filters", (0.5, 1.0), (0.8, 1.0, 0.05),
              (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
    ParamSpec("vi_hi", "Vortex band upper", "float", "No-trade filters", (1.0, 1.5), (1.0, 1.2, 0.05),
              (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)),
    ParamSpec("cost_pct", "Trading cost per side (%)", "float", "Costs", (0.0, 5.0), (0.0, 0.2, 0.05),
              (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)),
]
PARAM_BY_KEY = {p.key: p for p in PARAM_SPECS}

METRIC_CATALOG = {
    "net_pnl": "Net PnL", "total_ret": "Total return %", "cagr": "Avg annual return %",
    "mdd": "Max drawdown %", "sharpe": "Sharpe ratio", "sortino": "Sortino ratio",
    "calmar": "Calmar ratio", "vol": "Annualised volatility %", "exposure": "Market exposure %",
    "win_rate": "Win rate %", "n_trades": "Number of trades", "n_long": "Long trades",
    "n_short": "Short trades", "profit_factor": "Profit factor", "expectancy": "Expectancy / trade %",
    "avg_win": "Avg win %", "avg_loss": "Avg loss %", "payoff": "Payoff ratio",
    "best": "Best trade %", "worst": "Worst trade %", "cost_amt": "Trading cost (amount)",
    "cost_pct_profit": "Trading cost % of profit", "n_cross": "Crossovers total",
    "n_cross_nz": "Crossovers in no-trade zone", "cross_nz_ratio": "% crossovers in no-trade zone",
    "nz_time": "Time in no-trade zone %", "tl_time": "Time outside trading timeline %",
}
DEFAULT_METRICS = ["cagr", "mdd", "sharpe", "calmar", "win_rate", "n_trades"]


def fixed_values(inp: Inputs) -> dict:
    """Sidebar value of every sweepable parameter (used for anything not varied)."""
    d = asdict(inp.cfg)
    d["ema_len"], d["sma_len"] = inp.man_ema, inp.man_sma
    return d


def cast_value(spec: ParamSpec, v):
    return {"int": int, "float": float, "bool": bool, "choice": str}[spec.kind](v)


def all_values(spec: ParamSpec) -> list:
    if spec.kind == "bool":
        return [False, True]
    if spec.kind == "choice":
        return list(spec.choices)
    if spec.all_vals:
        return list(spec.all_vals)
    return list(range(int(spec.bounds[0]), int(spec.bounds[1]) + 1))


def range_values(lo: float, hi: float, step: float) -> list[float]:
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    return [round(lo + step * i, 6) for i in range(n)]


def canonical(c: dict) -> dict:
    """Blank (None) every setting that has no effect given the on/off switches, so
    combinations that differ only in a switched-off setting collapse into one."""
    c = dict(c)
    off: set[str] = set()
    if not c["use_er"]:
        off |= {"er_n", "er_min"}
    if not c["use_adx"]:
        off |= {"adx_n", "adx_min"}
    if not (c["use_vi_cross"] or c["use_vi_band"]):
        off.add("vi_n")
    if not c["use_vi_band"]:
        off |= {"vi_lo", "vi_hi"}
    for k in off:
        c[k] = None
    return c


def build_combos(axes: dict[str, list], fixed: dict) -> list[dict]:
    """Every unique, valid combination of the chosen values (EMA must be shorter than SMA)."""
    keys = list(axes)
    seen: set[tuple] = set()
    out: list[dict] = []
    for vals in itertools.product(*(axes[k] for k in keys)):
        c = canonical({**fixed, **dict(zip(keys, vals))})
        if c["ema_len"] >= c["sma_len"]:
            continue
        sig = tuple(c[k] for k in sorted(c))
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def make_cfg(c: dict, fixed: dict) -> Config:
    return Config(**{f.name: (fixed[f.name] if c[f.name] is None else c[f.name]) for f in fields(Config)})


def fast_trades(pos: np.ndarray, close: np.ndarray, cost_pct: float) -> pd.DataFrame:
    """Same trades and net returns as build_trades, straight from NumPy (no dates)."""
    prev = np.concatenate(([0], pos[:-1]))
    cost = 2 * cost_pct
    rets: list[float] = []
    sides: list[str] = []
    side, entry = 0, 0
    for k in np.flatnonzero(pos != prev):
        if side != 0:
            rets.append(side * (close[k] / close[entry] - 1) * 100 - cost)
            sides.append("Long" if side == 1 else "Short")
        side, entry = int(pos[k]), int(k)
    if side != 0:
        rets.append(side * (close[-1] / close[entry] - 1) * 100 - cost)
        sides.append("Long" if side == 1 else "Short")
    return pd.DataFrame({"Side": sides, "Net Return %": rets})


class SweepData:
    """Price arrays plus memoised EMA / SMA / no-trade series shared by every combination."""

    def __init__(self, prices: pd.DataFrame, start: date, interval: str = "1d"):
        self.prices = prices
        self.interval = interval
        self.close = prices["Close"].to_numpy(dtype=float)
        self.i0 = int(prices.index.searchsorted(pd.Timestamp(start)))
        if len(self.close) - self.i0 < 2:
            raise UserError("Not enough bars inside the evaluation range.")
        self.idx = prices.index[self.i0:]
        self._memo: dict = {}

    def _get(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def ema(self, n: int) -> np.ndarray:
        return self._get(("ema", n), lambda: ema_values(self.close, n))

    def sma(self, n: int) -> np.ndarray:
        return self._get(("sma", n), lambda: sma_values(self.close, n))

    def blocked(self, cfg: Config) -> np.ndarray:
        key = ("blk", cfg.use_er, cfg.er_n, cfg.er_min, cfg.use_adx, cfg.adx_n, cfg.adx_min,
               cfg.use_vi_cross, cfg.use_vi_band, cfg.vi_n, cfg.vi_lo, cfg.vi_hi)
        return self._get(key, lambda: no_trade_mask(self.prices, cfg)[0].to_numpy(dtype=bool))

    def off(self, cfg: Config) -> np.ndarray:
        return self._get(("off", cfg.timeline), lambda: off_timeline_mask(self.prices.index, cfg))


class PortfolioSweep:
    """Evaluates one parameter combination on an equal-capital portfolio of stocks (a single stock is a
    portfolio of one). Each stock gets capital / n, trades independently, and the metrics are computed on
    the combined equity curve."""

    def __init__(self, members: list[tuple[str, SweepData]], capital: float, interval: str):
        self.members = members
        self.alloc = capital / len(members)
        self.union = union_index([sd.idx for _, sd in members])
        self.takes = [carry_map(sd.idx, self.union) for _, sd in members]
        self.ppy = periods_per_year(self.union, interval)

    def evaluate(self, ema_len: int, sma_len: int, cfg: Config, capital: float, rf_pct: float) -> dict:
        n_bars = len(self.union)
        eq, eqg, invested = np.zeros(n_bars), np.zeros(n_bars), np.zeros(n_bars)
        n_cross = n_cross_nz = 0
        nz_share, tl_share, trade_frames = [], [], []
        for (_, sd), take in zip(self.members, self.takes):
            blocked, off = sd.blocked(cfg), sd.off(cfg)
            r = compute_pair(sd.close, sd.ema(ema_len), sd.sma(sma_len), blocked | off, sd.i0, cfg, self.alloc)
            e = carry(r["equity"], take, self.alloc)
            eq += e
            eqg += carry(r["equity_gross"], take, self.alloc)
            invested += e * (carry(r["pos"], take, 0) != 0)
            crosses = r["cross_up"] | r["cross_dn"]
            no_trade = blocked[sd.i0:]
            n_cross += int(crosses.sum())
            n_cross_nz += int((crosses & no_trade).sum())
            nz_share.append(no_trade.mean())
            tl_share.append(off[sd.i0:].mean())
            trade_frames.append(fast_trades(r["pos"], sd.close[sd.i0:], cfg.cost_pct))
        flat = np.zeros(n_bars, dtype=bool)
        psig = pd.DataFrame({
            "equity": eq, "equity_gross": eqg, "strat_ret": returns_from_equity(eq, capital),
            "position": (invested > 0).astype(int), "cross_up": flat, "cross_dn": flat,
            "no_trade": flat, "off_timeline": flat}, index=self.union)
        m = compute_metrics(psig, pd.concat(trade_frames, ignore_index=True), capital, rf_pct, self.ppy)
        m["exposure"] = float(np.mean(invested / eq)) * 100.0
        m["n_cross"], m["n_cross_nz"] = n_cross, n_cross_nz
        m["cross_nz_ratio"] = n_cross_nz / n_cross * 100 if n_cross > 0 else np.nan
        m["nz_time"], m["tl_time"] = float(np.mean(nz_share)) * 100.0, float(np.mean(tl_share)) * 100.0
        m["net_pnl"] = float(eq[-1] - capital)
        return m


def run_permutations(inp: Inputs, start: date, end: date, combos: list[dict], fixed: dict,
                     axes: dict[str, list], setup: list[dict]) -> dict:
    members: list[tuple[str, SweepData]] = []
    skipped: list[tuple[str, str]] = []
    for tk in inp.tickers:
        span = load_span(tk, start, end, inp.interval)
        prices = price_window(span, start, end, inp.interval) if not span.empty else span
        if prices.empty:
            skipped.append((tk, "no price data in the selected dates"))
            continue
        try:
            members.append((tk, SweepData(prices, start, inp.interval)))
        except UserError:
            skipped.append((tk, "not enough bars inside the evaluation range"))
    if not members:
        raise UserError(f"No price data found for {', '.join(inp.tickers)} in the selected dates.")
    data = PortfolioSweep(members, inp.capital, inp.interval)

    swept = [k for k, v in axes.items() if len(v) > 1]
    bar = st.progress(0.0, text="Running permutations...")
    every = max(1, len(combos) // 100)
    rows, full, values = [], [], []
    for k, c in enumerate(combos):
        cfg = make_cfg(c, fixed)
        m = data.evaluate(c["ema_len"], c["sma_len"], cfg, inp.capital, inp.rf_pct)
        eff = {**asdict(cfg), "ema_len": c["ema_len"], "sma_len": c["sma_len"]}
        full.append(param_row(eff))
        values.append(eff)
        rows.append({**{p: c[p] for p in swept}, **{mk: m.get(mk, np.nan) for mk in METRIC_CATALOG}})
        if k % every == 0:
            bar.progress((k + 1) / len(combos), text=f"Running permutations... {k + 1:,} / {len(combos):,}")
    bar.empty()

    df = pd.DataFrame(rows)
    for k in swept:                              # keep ints / bools tidy where a setting was switched off
        kind = PARAM_BY_KEY[k].kind
        if kind == "int":
            df[k] = df[k].astype("Int64")
        elif kind == "bool":
            df[k] = df[k].astype("boolean")
    st.session_state["perm_runs"] = st.session_state.get("perm_runs", 0) + 1
    return {"table": df, "full": pd.DataFrame(full), "values": values, "run_id": st.session_state["perm_runs"],
            "setup": setup, "swept": swept, "n": len(combos),
            "ticker": portfolio_label(tuple(t for t, _ in members)), "tickers": [t for t, _ in members],
            "skipped": skipped, "file_tag": members[0][0] if len(members) == 1 else f"portfolio_{len(members)}_stocks",
            "interval": inp.interval, "window": (start, end), "capital": inp.capital, "rf": inp.rf_pct,
            "timeline": describe_timeline(inp.cfg)}


def pick_values(spec: ParamSpec, current) -> tuple[list, str | None]:
    """One row of the parameter chooser. Returns (values to test, error message or None)."""
    numeric = spec.kind in ("int", "float")
    modes = ["Fixed", "Range", "Specific values", "All"] if numeric else ["Fixed", "Specific values", "All"]
    key = f"perm_{spec.key}"
    c_label, c_mode, c_vals = st.columns([3, 2, 5])
    c_label.markdown(f"**{spec.label}**")
    mode = c_mode.selectbox(spec.label, modes, key=f"{key}_mode", label_visibility="collapsed")

    with c_vals:
        if mode == "Fixed":
            st.caption(f"Profile value: {current}")
            return [current], None

        if mode == "All":
            vals = [cast_value(spec, v) for v in all_values(spec)]
            st.caption(f"{len(vals)} values")
            return vals, None

        if mode == "Range":
            lo_b, hi_b = spec.bounds
            r_lo, r_hi, r_st = spec.rng
            a, b, s = st.columns(3)
            if spec.kind == "int":
                lo = a.number_input("From", int(lo_b), int(hi_b), int(r_lo), 1, key=f"{key}_lo")
                hi = b.number_input("To", int(lo_b), int(hi_b), int(r_hi), 1, key=f"{key}_hi")
                step = s.number_input("Step", 1, max(1, int(hi_b - lo_b)), int(r_st), 1, key=f"{key}_st")
            else:
                lo = a.number_input("From", float(lo_b), float(hi_b), float(r_lo), 0.01, format="%.3f", key=f"{key}_lo")
                hi = b.number_input("To", float(lo_b), float(hi_b), float(r_hi), 0.01, format="%.3f", key=f"{key}_hi")
                step = s.number_input("Step", 0.001, float(hi_b - lo_b), float(r_st), 0.01, format="%.3f", key=f"{key}_st")
            if hi < lo:
                return [], f"{spec.label}: 'To' must not be smaller than 'From'."
            return [cast_value(spec, v) for v in range_values(lo, hi, step)], None

        # Specific values
        if numeric:
            txt = st.text_input("Values", value=str(current), key=f"{key}_txt", label_visibility="collapsed",
                                placeholder="comma separated, e.g. 10, 20, 50")
            vals = []
            for tok in txt.replace(";", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    x = float(tok)
                except ValueError:
                    return [], f"{spec.label}: '{tok}' is not a number."
                if spec.kind == "int" and not x.is_integer():
                    return [], f"{spec.label}: '{tok}' is not a whole number."
                if not spec.bounds[0] <= x <= spec.bounds[1]:
                    return [], f"{spec.label}: {tok} is outside the allowed {spec.bounds[0]} to {spec.bounds[1]}."
                vals.append(cast_value(spec, x))
            vals = list(dict.fromkeys(vals))
        else:
            options = [False, True] if spec.kind == "bool" else list(spec.choices)
            vals = st.multiselect("Values", options, default=[current] if current in options else [],
                                  key=f"{key}_ms", label_visibility="collapsed",
                                  format_func=(lambda v: "On" if v else "Off") if spec.kind == "bool" else str)
        if not vals:
            return [], f"{spec.label}: choose at least one value."
        return vals, None


def perm_sid(perm: dict) -> str:
    return f"{'-'.join(perm['tickers'])}_{perm['n']}_{'_'.join(perm['swept'])}"


def setup_rows(axes: dict[str, list]) -> list[dict]:
    """How each parameter was chosen for a permutation run (goes into the export)."""
    rows = []
    for spec in PARAM_SPECS:
        how = st.session_state.get(f"perm_{spec.key}_mode", "Fixed")
        rows.append({"Parameter": spec.label,
                     "How chosen": "Fixed (ticker profile value)" if how == "Fixed" else how,
                     "Values tested": describe_values(spec, axes[spec.key])})
    return rows


def sorted_view(df: pd.DataFrame, swept: list[str], metrics: list[str], sort_key: str, ascending: bool) -> pd.DataFrame:
    return df[swept + metrics].sort_values(sort_key, ascending=ascending, na_position="last")


def top_run_id(perm: dict) -> int:
    """Row (0-based) that is first in the permutation table as the user currently has it sorted."""
    df, swept = perm["table"], perm["swept"]
    sid = perm_sid(perm)
    metrics = [k for k in st.session_state.get("perm_metrics", DEFAULT_METRICS) if k in df.columns]
    if not metrics:
        return int(df.index[0])
    sort_key = st.session_state.get(f"perm_sort_{sid}", metrics[0])
    if sort_key not in metrics:
        sort_key = metrics[0]
    ascending = bool(st.session_state.get(f"perm_asc_{sid}", False))
    return int(sorted_view(df, swept, metrics, sort_key, ascending).index[0])


def copy_source_row(tk: str, perm: dict) -> tuple[int, str] | None:
    """(row id, label) of the permutation row to copy for this ticker, or None if there is nothing to copy yet."""
    if st.session_state.get("perm_copy_src", "Top row") == "Top row":
        return top_run_id(perm), "top row"
    run_id, ids = st.session_state.get("perm_selected", {}).get(tk, (None, []))
    if run_id != perm["run_id"] or not ids:
        return None
    return int(ids[0]), "selected row"


def apply_perm_copy() -> None:
    """If the copy checkbox is ticked, write ALL parameters of the chosen permutation row (top row of the table as
    sorted, or the first row ticked in its Select column) into the profile of the ticker whose permutations were
    run. The ticker shown in the sidebar gets its widgets filled; the others have their stored profile updated.
    Runs before the sidebar is drawn; copies once per change of source row so manual edits still work."""
    ss = st.session_state
    if not ss.get("perm_copy_top", False):
        ss.pop("perm_copy_sigs", None)
        return
    sigs = ss.setdefault("perm_copy_sigs", {})
    store, loaded = ss.get("profiles", {}), ss.get("profile_loaded")
    for tk, perm in ss.get("perms", {}).items():
        if tk not in store:
            continue
        src = copy_source_row(tk, perm)
        if src is None:
            continue
        rid, label = src
        sig = (perm["run_id"], rid, label)
        if sigs.get(tk) == sig:
            continue
        values = {widget: cast_value(PARAM_BY_KEY[key], perm["values"][rid][key]) for key, widget in SIDEBAR_KEYS.items()}
        values["sb_use_estimated"] = False       # use these EMA/SMA lengths, not section 1's estimate
        if tk == loaded:
            profile_load(values)
        else:
            store[tk].update(values)
        sigs[tk] = sig
        ss.setdefault("perm_copied", {})[tk] = (rid + 1, label)


def render_perm_results(perm: dict, metrics: list[str]) -> None:
    df, swept = perm["table"], perm["swept"]
    a, b = perm["window"]
    st.subheader("Permutation results" + (f": {perm['tickers'][0]}" if len(st.session_state.get("perms", {})) > 1 else ""))
    varied = ", ".join(PARAM_BY_KEY[k].label for k in swept) or "nothing (single run)"
    for tk, why in perm.get("skipped", []):
        st.warning(f"{tk}: {why}, so it was left out of the portfolio.")
    n_st = len(perm.get("tickers", [None]))
    st.caption(f"{perm['ticker']}{' (equal capital per stock)' if n_st > 1 else ''}, {INTERVALS[perm.get('interval', '1d')][0]} bars, {a} to {b}. {perm['n']:,} combinations. Varied: {varied}. "
               "A blank parameter cell means that setting was switched off in that row.")
    k1, k2 = st.columns([3, 2])
    tk = perm["tickers"][0]
    k1.checkbox("Copy a permutation row's parameters into that ticker's profile (best fit)", key="perm_copy_top",
                help="Fills ALL parameters of the ticker's profile (EMA/SMA lengths, stop loss, entry/exit rules, "
                     "no-trade filters, costs) from the chosen row of its results table, and turns off 'Use section 1 "
                     "to estimate EMA/SMA values' for it. This is how each ticker gets its own best-fit values. "
                     "Press Run backtest to run it.")
    k2.radio("Copy from", ["Top row", "Selected row"], horizontal=True, key="perm_copy_src",
             disabled=not st.session_state.get("perm_copy_top", False),
             help="Top row: the first row of the table as currently sorted (updates after every run or re-sort). "
                  "Selected row: the row whose Select box you tick in the table (first one if several).")
    if st.session_state.get("perm_copy_top"):
        copied = st.session_state.get("perm_copied", {}).get(tk)
        if copied:
            st.caption(f"The profile of {tk} now uses run #{copied[0]} ({copied[1]}). "
                       "Press Run backtest to run it.")
        if (st.session_state.get("perm_copy_src") == "Selected row"
                and not st.session_state.get("perm_selected", {}).get(tk, (0, []))[1]):
            st.info("Tick the Select box of a row in the table below to copy that row into this ticker's profile.")
    if not metrics:
        st.info("Pick at least one metric above to see results.")
        return

    sid = perm_sid(perm)
    c1, c2, c3 = st.columns([3, 2, 2])
    sort_key = c1.selectbox("Sort by", metrics, format_func=METRIC_CATALOG.get, key=f"perm_sort_{sid}")
    ascending = c2.checkbox("Ascending", value=False, key=f"perm_asc_{sid}")
    top_n = c3.number_input("Show top N (0 = all)", 0, len(df), 0, key=f"perm_top_{sid}")

    view = sorted_view(df, swept, metrics, sort_key, ascending)
    if top_n:
        view = view.head(int(top_n))
    view[metrics] = view[metrics].astype(float).round(2)
    shown = view.rename(columns={**{k: PARAM_BY_KEY[k].label for k in swept}, **METRIC_CATALOG})
    ids = [int(i) for i in view.index]

    select_all = st.checkbox("Select all rows shown", value=False, key=f"perm_all_{sid}")
    table = shown.copy()
    table.insert(0, "Run #", [i + 1 for i in ids])
    table.insert(0, "Select", select_all)
    edited = st.data_editor(
        table, use_container_width=True, hide_index=True,
        disabled=[c for c in table.columns if c != "Select"],
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
        key=f"perm_edit_{sid}_{sort_key}_{int(ascending)}_{int(top_n)}_{int(select_all)}")
    chosen = [ids[i] for i in np.flatnonzero(edited["Select"].to_numpy(dtype=bool))]
    selected = st.session_state.setdefault("perm_selected", {})
    if chosen != selected.get(tk, (None, []))[1]:
        selected[tk] = (perm["run_id"], chosen)
        if (st.session_state.get("perm_copy_top") and st.session_state.get("perm_copy_src") == "Selected row"
                and chosen):
            st.rerun()                          # sidebar is already drawn: re-run so it picks up the selected row
    if len(chosen) > 1 and st.session_state.get("perm_copy_top") and st.session_state.get("perm_copy_src") == "Selected row":
        st.caption("More than one row is ticked: the first ticked row (highest in the table) is copied.")

    st.markdown("**Export selected rows**")
    all_metrics = st.checkbox("Include all metrics in the export (not only the ones shown)", value=True,
                              key=f"perm_allm_{sid}")
    if not chosen:
        st.info("Tick the Select box on the rows you want to export. The export lists every parameter "
                "(On/Off or the value used) and the result metrics for each chosen row.")
    else:
        st.caption(f"{len(chosen)} row(s) selected.")
        sheets = perm_export_sheets(perm, chosen, list(METRIC_CATALOG) if all_metrics else metrics)
        export_buttons(sheets, f"permutation_runs_{perm.get('file_tag', 'run')}", key=f"perm_exp_{sid}")

    if len(swept) >= 2:
        with st.expander("Pivot two parameters"):
            p1, p2, p3, p4 = st.columns(4)
            row_k = p1.selectbox("Rows", swept, format_func=lambda k: PARAM_BY_KEY[k].label, key=f"pv_r_{sid}")
            col_k = p2.selectbox("Columns", [k for k in swept if k != row_k],
                                 format_func=lambda k: PARAM_BY_KEY[k].label, key=f"pv_c_{sid}")
            met = p3.selectbox("Metric", metrics, format_func=METRIC_CATALOG.get, key=f"pv_m_{sid}")
            agg = p4.selectbox("Combine the other parameters by", ["mean", "max", "min", "median"], key=f"pv_a_{sid}")
            piv = df.pivot_table(index=row_k, columns=col_k, values=met, aggfunc=agg)
            if piv.empty:
                st.write("Nothing to show for this choice.")
            else:
                z = piv.to_numpy(dtype=float)
                fig = go.Figure(go.Heatmap(
                    z=z, x=[str(c) for c in piv.columns], y=[str(r) for r in piv.index], colorscale="RdYlGn",
                    text=[["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z],
                    texttemplate="%{text}", colorbar=dict(title=METRIC_CATALOG[met])))
                fig.update_xaxes(type="category", title=PARAM_BY_KEY[col_k].label)
                fig.update_yaxes(type="category", title=PARAM_BY_KEY[row_k].label)
                fig.update_layout(height=max(320, 40 * len(piv) + 140), margin=CHART_MARGIN)
                st.plotly_chart(fig, use_container_width=True)


def render_best_fit_summary(perms: dict, metrics: list[str]) -> None:
    """Top row of every ticker's latest permutation run, side by side."""
    rows = []
    for tk, perm in perms.items():
        df, swept = perm["table"], perm["swept"]
        rid = top_run_id(perm)
        row: dict = {"Ticker": tk, "Run #": rid + 1}
        for k in swept:
            v = df.loc[rid, k]
            row[PARAM_BY_KEY[k].label] = None if pd.isna(v) else (bool(v) if PARAM_BY_KEY[k].kind == "bool" else v)
        for k in [k for k in metrics if k in df.columns]:
            row[METRIC_CATALOG[k]] = round(float(df.loc[rid, k]), 2)
        rows.append(row)
    with st.expander("Best fit per ticker (top row of each ticker's latest permutation run)", expanded=len(rows) > 1):
        st.dataframe(pd.DataFrame(rows).astype(object), use_container_width=True, hide_index=True)
        st.caption("Rows are ranked by the first metric you chose (or the sort you picked for the ticker shown). "
                   "Tick 'Copy a permutation row's parameters ...' below to push each ticker's best fit into its profile.")


def render_permutations(inp: Inputs) -> None:
    multi = len(inp.tickers) > 1
    st.header(f"Permutations: {inp.active}" if multi else "Permutations")
    st.caption("For each parameter choose Fixed (the selected ticker's profile value), a Range, Specific values or "
               "All. Every combination of what you choose is backtested on that ticker alone, on the window below, "
               "with the same engine as the section further down. Run it for each ticker (or all at once) to find "
               "the parameters that fit each one best.")
    fixed = fixed_values(inp)
    axes: dict[str, list] = {}
    setup: list[dict] = []
    errors: list[str] = []

    groups: dict[str, list[ParamSpec]] = {}
    for spec in PARAM_SPECS:
        groups.setdefault(spec.group, []).append(spec)
    for i, (group, specs) in enumerate(groups.items()):
        with st.expander(group, expanded=(i == 0)):
            for spec in specs:
                vals, err = pick_values(spec, fixed[spec.key])
                axes[spec.key] = vals
                if err:
                    errors.append(err)

    metrics = st.multiselect("Metrics to show in the results", list(METRIC_CATALOG), default=DEFAULT_METRICS,
                             format_func=METRIC_CATALOG.get, key="perm_metrics")

    where = st.radio("Evaluate on", ["Backtest dates (section 2)", "Training dates (section 1)", "Custom dates"],
                     horizontal=True, key="perm_window")
    if where.startswith("Backtest"):
        start, end = inp.test_start, inp.test_end
    elif where.startswith("Training"):
        start, end = inp.train_start, inp.train_end
    else:
        d1, d2 = st.columns(2)
        start = d1.date_input("From", date(2020, 1, 1), key="perm_from")
        end = d2.date_input("To", date(2025, 1, 1), key="perm_to")
    if start >= end:
        errors.append("The evaluation start date must be before the end date.")
    if not inp.tickers:
        errors.append("Enter at least one ticker in the sidebar.")
    errors += interval_problems(inp.interval, start, end)
    errors += timeline_problems(inp, start, end)

    max_combos = st.number_input("Maximum combinations allowed", 1, 100000, 5000, step=500, key="perm_max",
                                 help="Safety limit on the number of raw combinations, "
                                      "to avoid accidentally launching a huge run.")
    raw = math.prod(len(v) for v in axes.values())
    combos: list[dict] = []
    if errors:
        for e in errors:
            st.error(e)
    elif raw > max_combos:
        st.error(f"{raw:,} raw combinations is above the limit of {max_combos:,}. "
                 "Narrow a range, use fewer values, or raise the limit.")
    else:
        combos = build_combos(axes, fixed)
        if combos:
            st.info(f"{len(combos):,} unique combinations to run ({raw:,} raw; combinations with EMA not "
                    "shorter than SMA, and ones that differ only in a switched-off setting, are removed).")
        else:
            st.warning("No valid combinations: EMA length must be shorter than SMA length.")

    def run_for(tk: str, show_errors: bool = True) -> bool:
        """Run the sweep on one ticker with that ticker's own profile as the 'Fixed' values."""
        prof = inp.profiles[tk]
        inp_t = replace(inp, tickers=(tk,), cfg=prof.cfg, man_ema=prof.man_ema, man_sma=prof.man_sma,
                        use_estimated=prof.use_estimated, method=prof.method, profiles={tk: prof}, active=tk)
        fixed_t = fixed_values(inp_t)
        axes_t = {k: ([fixed_t[k]] if st.session_state.get(f"perm_{k}_mode", "Fixed") == "Fixed" else v)
                  for k, v in axes.items()}
        combos_t = build_combos(axes_t, fixed_t)
        try:
            if not combos_t:
                raise UserError(f"{tk}: no valid combinations (EMA length must be shorter than SMA length).")
            st.session_state.setdefault("perms", {})[tk] = run_permutations(
                inp_t, start, end, combos_t, fixed_t, axes_t, setup_rows(axes_t))
            return True
        except UserError as exc:
            if show_errors:
                st.error(str(exc))
            return False

    b1, b2 = st.columns(2)
    run = b1.button(f"▶ Run permutations for {inp.active}" if multi else "▶ Run permutations", type="primary",
                    disabled=not combos, key="perm_run")
    run_all = b2.button(f"▶ Run permutations for all {len(inp.tickers)} tickers", disabled=not combos,
                        key="perm_run_all", help="Runs the same sweep once per ticker; each ticker's 'Fixed' "
                        "parameters come from its own profile.") if multi else False
    if run or run_all:
        done, failed = [], []
        for tk in (inp.tickers if run_all else (inp.active,)):
            (done if run_for(tk, show_errors=not run_all) else failed).append(tk)
        for tk in failed if run_all else []:
            st.warning(f"{tk}: could not run permutations (no data or no valid combination).")
        if done and st.session_state.get("perm_copy_top"):
            st.rerun()                          # let the profiles pick up the new best fit

    perms = st.session_state.get("perms", {})
    perms = {tk: v for tk, v in perms.items() if tk in inp.tickers}
    if perms:
        render_best_fit_summary(perms, metrics)
    perm = perms.get(inp.active)
    if perm is not None:
        render_perm_results(perm, metrics)
    elif perms:
        st.info(f"No permutation run for {inp.active} yet. Press the Run button above.")


# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
def render_profile_section(inp: Inputs) -> bool:
    """Ticker switcher + overview of every ticker's parameters. Returns True if the 'run this ticker only'
    button was pressed."""
    st.header("Ticker profiles")
    if not inp.tickers:
        st.info("Enter at least one ticker in the sidebar.")
        return False
    n = len(inp.tickers)
    if n > 1:
        st.caption("Each ticker has its own parameter profile (EMA/SMA, stop loss, entry/exit rules, filters, cost). "
                   "Pick a ticker here: the sidebar then shows and edits its parameters, Permutations runs on it, "
                   "and the backtest results below show it. The portfolio backtest uses every ticker's own profile.")
        if n <= 8:
            st.radio("Ticker", list(inp.tickers), horizontal=True, key="active_ticker", on_change=switch_profile)
        else:
            st.selectbox("Ticker", list(inp.tickers), key="active_ticker", on_change=switch_profile)
    else:
        st.caption(f"One ticker: **{inp.tickers[0]}**. Add more in the sidebar to build a portfolio with a separate "
                   "parameter profile per ticker.")

    perms, copied = st.session_state.get("perms", {}), st.session_state.get("perm_copied", {})
    rows = []
    for tk in inp.tickers:
        p = inp.profiles[tk]
        c = p.cfg
        rows.append({
            "Ticker": ("▶ " if tk == inp.active and n > 1 else "") + tk,
            "EMA": p.man_ema, "SMA": p.man_sma,
            "EMA/SMA from": "Section 1 estimate" if (p.use_estimated and inp.use_train) else "Manual / best fit",
            "Stop %": c.stop_pct, "Short": "On" if c.allow_short else "Off",
            "ER": "On" if c.use_er else "Off", "ADX": "On" if c.use_adx else "Off",
            "Vortex": "On" if (c.use_vi_cross or c.use_vi_band) else "Off", "Cost %": c.cost_pct,
            "Permutations run": "yes" if tk in perms else "no",
            "Best fit applied": f"run #{copied[tk][0]}" if tk in copied else "",
        })
    with st.expander("Parameters of every ticker", expanded=n > 1):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    run_one = False
    if n > 1:
        c1, c2 = st.columns(2)
        run_one = c1.button(f"▶ Run backtest for {inp.active} only", key="run_one",
                            help="Backtests just this ticker with its own profile, using all of the capital.")
        c2.button("Copy this ticker's parameters to all tickers", on_click=copy_profile_to_all,
                  help="Overwrites every other ticker's profile with the parameters now in the sidebar.")
    return run_one


def main() -> None:
    st.set_page_config(page_title="EMA/SMA Crossover Strategy", layout="wide")
    st.title("EMA / SMA Crossover Strategy Dashboard")

    run = st.sidebar.button("▶ Run backtest", type="primary", use_container_width=True)
    st.sidebar.caption("Nothing is calculated until you press a Run button. This one backtests every ticker as a "
                       "portfolio, each with its own parameter profile; Permutations has its own Run buttons. "
                       "Sidebar values are the selected ticker's profile.")
    sync_profiles(current_tickers())
    apply_perm_copy()
    inp = sidebar_inputs()

    run_one = render_profile_section(inp)
    st.divider()
    render_permutations(inp)
    st.divider()
    st.header("Backtest engine")

    if run or run_one:
        inp_run = inp if run else restrict_inputs(inp, (inp.active,))
        problems = date_problems(inp_run)
        for p in problems:
            st.error(p)
        if not problems:
            try:
                with st.spinner("Running backtest..."):
                    st.session_state["results"] = compute_all(inp_run)
            except UserError as exc:
                st.error(str(exc))

    res: Results | None = st.session_state.get("results")
    if res is None:
        st.info("Set your parameters in the sidebar, then press **▶ Run backtest**.")
        return
    if inputs_changed(res.inp, inp):
        st.warning("Settings changed since the last run. The results below are from the previous settings; "
                   "press **▶ Run backtest** to update them.")
    render_results(res)


if __name__ == "__main__":
    main()