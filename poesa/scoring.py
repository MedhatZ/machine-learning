"""درجة توقع 20 جلسة. الحساب سببي: كل يوم يستخدم اللي قبله فقط."""

from __future__ import annotations

import numpy as np
import pandas as pd

from poesa import config
from poesa.benchmark import trailing_compound


def band_name(score: float) -> str:
    for low, high, name in config.BANDS:
        if low <= score < high:
            return name
    return config.BANDS[-1][2]


def map_dist(series: pd.Series, scale: float) -> pd.Series:
    return (0.5 + series / (2.0 * scale)).clip(0.0, 1.0) * 100.0


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative))


def _atr_pct(price: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    previous = price.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=window).mean()
    return atr / price


def _extension_points(distance: pd.Series, start: float, full: float) -> pd.Series:
    span = full - start
    return ((distance - start) / span).clip(0.0, 1.0) * config.EXTENSION_PENALTY_POINTS


def score_panel(
    price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    bench_daily: pd.Series,
) -> pd.DataFrame:
    stock_daily = price.pct_change()
    bench_daily = bench_daily.reindex(price.index)
    sma20 = price.rolling(20, min_periods=20).mean()
    sma50 = price.rolling(50, min_periods=50).mean()
    sma200 = price.rolling(200, min_periods=200).mean()
    # الاتجاه ترتيب المتوسطات. البعد بيتشبع عند 2% عشان الزيادة بعده متزودش النقاط.
    trend = pd.concat(
        [
            map_dist(price / sma50 - 1.0, 0.02),
            map_dist(sma50 / sma200 - 1.0, 0.02),
        ],
        axis=1,
    ).mean(axis=1)

    stock5 = trailing_compound(stock_daily, 5)
    stock20 = trailing_compound(stock_daily, 20)
    momentum = pd.concat(
        [map_dist(stock5, 0.06), map_dist(stock20, 0.12)],
        axis=1,
    ).mean(axis=1)

    bench5 = trailing_compound(bench_daily, 5)
    bench20 = trailing_compound(bench_daily, 20)
    relative = pd.concat(
        [map_dist(stock5 - bench5, 0.04), map_dist(stock20 - bench20, 0.08)],
        axis=1,
    ).mean(axis=1)

    high20 = price.rolling(20, min_periods=20).max()
    proximity = price / high20
    proximity_score = ((proximity - 0.90) / 0.10).clip(0.0, 1.0) * 100.0
    volume_ratio = volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean()
    volume_confirm = map_dist(volume_ratio - 1.0, 0.8).where(proximity >= 0.97, 0.0)
    volume_score = pd.concat([proximity_score, volume_confirm], axis=1).mean(axis=1)

    atr_pct = _atr_pct(price, high, low)
    risk_atr = ((0.045 - atr_pct) / 0.035).clip(0.0, 1.0) * 100.0
    drawdown = price / price.rolling(120, min_periods=120).max() - 1.0
    risk_dd = (1.0 + drawdown / 0.30).clip(0.0, 1.0) * 100.0
    risk = pd.concat([risk_atr, risk_dd], axis=1).mean(axis=1)

    finite_vol = stock_daily.notna().rolling(60, min_periods=60).sum() == 60
    ann_vol = (stock_daily.rolling(60, min_periods=60).std() * np.sqrt(252)).where(finite_vol)
    dist_sma50 = price / sma50 - 1.0
    dist_sma200 = price / sma200 - 1.0
    penalty = pd.concat(
        [
            _extension_points(dist_sma200, config.MA200_EXTENSION_START, config.MA200_EXTENSION_FULL),
            _extension_points(dist_sma50, config.MA50_EXTENSION_START, config.MA50_EXTENSION_FULL),
        ],
        axis=1,
    ).max(axis=1)

    factors = pd.DataFrame(
        {
            "momentum": momentum,
            "relative_strength": relative,
            "trend": trend,
            "volume": volume_score,
            "risk": risk,
        }
    )
    raw = sum(factors[name] * weight for name, weight in config.WEIGHTS.items())
    score = (raw - penalty).clip(0.0, 100.0)
    ready = factors.notna().all(axis=1) & penalty.notna()
    score = score.where(ready)
    return pd.DataFrame(
        {
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "rsi": _rsi(price),
            "atr_pct": atr_pct,
            "drawdown": drawdown,
            "ann_vol": ann_vol,
            "dist_sma50": dist_sma50,
            "dist_sma200": dist_sma200,
            "extension_penalty": penalty,
            **{name: factors[name] for name in config.WEIGHTS},
            "score": score,
        }
    )
