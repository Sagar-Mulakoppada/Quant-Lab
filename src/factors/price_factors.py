"""
price_factors.py
================
Library of cross-sectional price-based alpha factors.

Every function accepts a prices DataFrame (T × N) and returns a
signal DataFrame of the same shape — higher = more bullish.
All factors are computed without lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cross_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to cross-sectional percentile rank each day (0..1)."""
    return df.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Classic 12-1 momentum: 12-month trailing return, skipping the last month.
    Fama-French / Jegadeesh-Titman (1993).
    """
    r = prices.pct_change(252) - prices.pct_change(21)  # 12m minus 1m
    return _cross_rank(r)


def short_term_reversal(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Short-term mean reversion: negative of N-day return.
    Stocks that fell hard tend to bounce.
    """
    r = -prices.pct_change(window)
    return _cross_rank(r)


# ---------------------------------------------------------------------------
# Volatility / risk
# ---------------------------------------------------------------------------

def realized_volatility(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Negative realized volatility — low-vol anomaly (Baker et al.).
    Low-vol stocks tend to outperform on a risk-adjusted basis.
    """
    log_r = np.log(prices).diff()
    vol = log_r.rolling(window).std() * np.sqrt(252)
    return _cross_rank(-vol)   # invert: lower vol = higher rank


# ---------------------------------------------------------------------------
# Technical / oscillator
# ---------------------------------------------------------------------------

def rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (Wilder, 1978).
    Returns cross-sectional rank — high RSI = overbought (contrarian short signal),
    so caller should invert for mean-reversion.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - 100 / (1 + rs)
    return _cross_rank(rsi_val)


def rsi_divergence(prices: pd.DataFrame, rsi_period: int = 14,
                   corr_window: int = 20) -> pd.DataFrame:
    """
    RSI–price divergence.
    Positive divergence (RSI rising, price falling) → bullish.
    Signal = rolling corr(rsi, price) — lower (negative) = stronger bullish divergence.
    """
    rsi_val = rsi(prices, rsi_period)
    price_rank = _cross_rank(prices)

    divergence = rsi_val.rolling(corr_window).corr(price_rank)
    return _cross_rank(-divergence)   # negative corr = bullish


def macd_signal(prices: pd.DataFrame,
                fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    """
    MACD histogram as a momentum/trend signal.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    histogram = macd_line - signal_line
    return _cross_rank(histogram)


# ---------------------------------------------------------------------------
# Volume-based
# ---------------------------------------------------------------------------

def price_volume_trend(prices: pd.DataFrame,
                       volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative Price-Volume Trend (PVT).
    High PVT = price rising with volume confirmation → bullish.
    """
    pct_chg = prices.pct_change()
    pvt = (pct_chg * volumes).cumsum()
    return _cross_rank(pvt)


def amihud_illiquidity(prices: pd.DataFrame,
                       volumes: pd.DataFrame,
                       window: int = 20) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity ratio: |return| / dollar_volume.
    Here we flip sign — more *liquid* stocks = higher rank (liquidity premium).
    """
    dollar_vol = prices * volumes
    abs_ret = prices.pct_change().abs()
    illiq = (abs_ret / dollar_vol.replace(0, np.nan)).rolling(window).mean()
    return _cross_rank(-illiq)   # invert: liquid = high rank


# ---------------------------------------------------------------------------
# Trend / breakout
# ---------------------------------------------------------------------------

def price_to_52w_high(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price / 52-week high ratio.
    George & Hwang (2004) — proximity to 52-week high predicts momentum.
    """
    high_52 = prices.rolling(252).max()
    ratio = prices / high_52.replace(0, np.nan)
    return _cross_rank(ratio)


def bollinger_position(prices: pd.DataFrame,
                       window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Position within Bollinger Bands (0 = at lower band, 1 = at upper band).
    Use as contrarian (short) or trend-following (long) depending on regime.
    """
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    pos = (prices - lower) / (upper - lower).replace(0, np.nan)
    return _cross_rank(pos)
