"""خط الأساس: المؤشر لو تاريخه كافي، وغير كده متوسط باقي القائمة."""

from __future__ import annotations

import numpy as np
import pandas as pd

from poesa import config


def leave_one_out(returns: pd.DataFrame, min_peers: int | None = None) -> pd.DataFrame:
    needed = config.MIN_PEERS if min_peers is None else min_peers
    present = returns.notna()
    count = present.sum(axis=1)
    total = returns.sum(axis=1, min_count=1)
    out = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for column in returns.columns:
        peers = count - present[column].astype(int)
        summed = total - returns[column].fillna(0.0)
        series = summed / peers.replace(0, np.nan)
        out[column] = series.where(present[column] & (peers >= needed))
    return out


def select_benchmarks(
    index_returns: pd.Series | None,
    stock_returns: pd.DataFrame,
    min_peers: int | None = None,
) -> tuple[str, pd.DataFrame]:
    if index_returns is not None and int(index_returns.notna().sum()) >= config.MIN_CALENDAR_BARS:
        aligned = index_returns.reindex(stock_returns.index)
        bench = pd.DataFrame({column: aligned for column in stock_returns.columns})
        return "official", bench
    return "basket", leave_one_out(stock_returns, min_peers)


def trailing_compound(daily: pd.Series, window: int) -> pd.Series:
    clean = daily.notna()
    filled = daily.fillna(0.0)
    product = (1.0 + filled).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
    complete = clean.rolling(window, min_periods=window).sum() == window
    return product.where(complete)
