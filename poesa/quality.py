"""فحص جودة السلسلة قبل ما السهم يدخل الترتيب."""

from __future__ import annotations

import pandas as pd

from poesa import config
from poesa.data import naive_dates


def assess_calendar(calendar: pd.DatetimeIndex, today: pd.Timestamp | None = None) -> tuple[bool, str]:
    if calendar is None or len(calendar) < config.MIN_CALENDAR_BARS:
        count = 0 if calendar is None else len(calendar)
        return False, f"تقويم الجلسات قصير ({count} جلسة). الترتيب واقف."
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    last = pd.Timestamp(calendar[-1]).normalize()
    age = int((today - last).days)
    if age > config.MAX_CALENDAR_AGE_DAYS:
        return False, f"آخر جلسة في البيانات عمرها {age} يوم. الترتيب واقف لحد ما البيانات تتجدد."
    gaps = pd.Series(calendar).diff().dt.days
    recent = gaps.tail(config.MISSING_WINDOW)
    if (recent > config.MAX_CALENDAR_GAP_DAYS).any():
        return False, "في فجوة طويلة في تقويم الجلسات خلال آخر سنة. الترتيب واقف."
    return True, "تقويم الجلسات سليم."


def trading_calendar(frames: dict[str, pd.DataFrame], min_names: int | None = None) -> pd.DatetimeIndex:
    needed = config.MIN_NAMES_FOR_CALENDAR if min_names is None else min_names
    counts: pd.Series | None = None
    for frame in frames.values():
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        present = frame["close"].notna().astype(int)
        present.index = naive_dates(present.index)
        present = present[~present.index.duplicated(keep="last")]
        counts = present if counts is None else counts.add(present, fill_value=0)
    if counts is None or counts.empty:
        return pd.DatetimeIndex([])
    dates = counts[counts >= needed].index.sort_values()
    return pd.DatetimeIndex(dates)


def assess(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if frame is None or frame.empty:
        return False, ["لا توجد بيانات من المصدر"]
    if calendar is None or len(calendar) == 0:
        return False, ["مفيش تقويم جلسات نقيس عليه"]

    adj = _align(frame["adj"], calendar)
    close = _align(frame["close"], calendar)
    high = _align(frame["high"], calendar)
    low = _align(frame["low"], calendar)
    volume = _align(frame["volume"], calendar)
    valid = adj.dropna()
    if len(valid) < config.MIN_BARS:
        reasons.append(f"التاريخ أقصر من {config.MIN_BARS} جلسة")
        return False, reasons

    cutoff = pd.Timestamp(calendar[-(config.MAX_STALE_SESSIONS + 1)])
    if valid.index[-1] < cutoff:
        reasons.append("آخر سعر أقدم من السوق")

    recent_calendar = calendar[-config.MISSING_WINDOW :]
    missing = float(adj.reindex(recent_calendar).isna().mean())
    if missing > config.MAX_MISSING_RATIO:
        reasons.append(f"أيام ناقصة {missing:.0%} في آخر {config.MISSING_WINDOW} جلسة")

    returns = adj.pct_change()
    jumps = returns.abs().tail(config.JUMP_LOOKBACK)
    if int((jumps > config.JUMP_ABS).sum()) > 0:
        reasons.append("قفزة يومية أكبر من حدود الجلسة المعتادة")

    recent_volume = volume.tail(config.LIQUIDITY_WINDOW)
    if recent_volume.fillna(0).sum() <= 0:
        reasons.append("حجم التداول غير متاح من المصدر")
    else:
        traded = (close * volume).dropna().tail(config.LIQUIDITY_WINDOW)
        median_value = float(traded.median()) if len(traded) else 0.0
        if median_value < config.MIN_MEDIAN_VALUE:
            reasons.append(f"سيولة ضعيفة (وسيط قيمة التداول {median_value:,.0f} جنيه)")

    span = (high - low) / close.replace(0, pd.NA)
    at_limit = (returns.abs() >= config.LIMIT_MOVE) & (span <= config.LIMIT_RANGE)
    recent_returns = returns.tail(config.LIMIT_WINDOW)
    recent_limits = at_limit.tail(config.LIMIT_WINDOW)
    finite = recent_returns.notna()
    if int(finite.sum()) > 0:
        fraction = float(recent_limits[finite].mean())
        if fraction > config.MAX_LIMIT_FRACTION:
            reasons.append(f"أيام حدود جلسة كثيرة ({fraction:.0%} من آخر {config.LIMIT_WINDOW})")

    return len(reasons) == 0, reasons


def _align(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    values = series.copy()
    values.index = naive_dates(values.index)
    values = values[~values.index.duplicated(keep="last")]
    return values.reindex(calendar)
