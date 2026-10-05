"""عائد زائد بعينات غير متداخلة. النتيجة بتتخبى لو عدد الحالات قليل."""

from __future__ import annotations

import numpy as np
import pandas as pd

from poesa import config
from poesa.scoring import band_name


def sample_positions(length: int, start: int, horizon: int) -> list[int]:
    if horizon <= 0 or start < 0 or start >= length - horizon:
        return []
    return list(range(start, length - horizon, horizon))


def holding_return(daily: pd.Series, origin: int, horizon: int) -> float | None:
    window = daily.iloc[origin + 1 : origin + horizon + 1]
    if len(window) != horizon or bool(window.isna().any()):
        return None
    return float((1.0 + window).prod() - 1.0)


def excess_table(
    score: pd.Series,
    stock_daily: pd.Series,
    bench_daily: pd.Series,
    current_score: float,
    horizons: tuple[int, ...] = config.HORIZONS,
    min_n: int = config.MIN_N,
    cost: float = config.ROUND_TRIP_COST,
) -> dict[int, dict]:
    aligned = pd.DataFrame(
        {
            "score": score,
            "stock": stock_daily.reindex(score.index),
            "bench": bench_daily.reindex(score.index),
        }
    )
    values = aligned["score"].to_numpy(dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    current_band = band_name(float(current_score))
    table: dict[int, dict] = {}
    if len(finite) == 0:
        for horizon in horizons:
            table[horizon] = _hidden(0)
        return table

    start = int(finite[0])
    for horizon in horizons:
        excesses: list[float] = []
        for origin in sample_positions(len(aligned), start, horizon):
            point = values[origin]
            if not np.isfinite(point) or band_name(float(point)) != current_band:
                continue
            stock = holding_return(aligned["stock"], origin, horizon)
            bench = holding_return(aligned["bench"], origin, horizon)
            if stock is None or bench is None:
                continue
            excesses.append(stock - bench)
        count = len(excesses)
        if count < min_n:
            table[horizon] = _hidden(count)
            continue
        gross = float(np.mean(excesses))
        table[horizon] = {
            "n": count,
            "hidden": False,
            "gross": gross,
            "net": gross - cost,
            "beat": float(np.mean(np.array(excesses) > 0.0)),
        }
    return table


def _hidden(count: int) -> dict:
    return {"n": count, "hidden": True, "gross": None, "net": None, "beat": None}
