"""ووك-فورورد لدرجة 20 جلسة. الاختيار عند الإغلاق، والعائد بعده. الأوزان مش بتتغير هنا."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from poesa import config, data, quality
from poesa.benchmark import select_benchmarks
from poesa.engine import _series
from poesa.membership import ANCHOR_NOTE, gap_notes, membership_frame, membership_intervals, research_symbols
from poesa.scoring import score_panel
from poesa.stats import holding_return

SIGNAL_COLUMNS = (
    "momentum",
    "relative_strength",
    "trend",
    "volume",
    "risk",
    "dist_sma50",
    "dist_sma200",
    "extension_penalty",
)


@dataclass
class Window:
    date: str
    top: list[str]
    top_return: float
    rest_return: float
    excess: float


@dataclass
class SliceSummary:
    years: int
    n: int
    hidden: bool
    top: float | None
    rest: float | None
    gross: float | None
    net: float | None
    beat: float | None
    median: float | None


@dataclass
class SignalSummary:
    name: str
    n: int
    hidden: bool
    mean: float | None
    median: float | None
    positive: float | None
    spread_n: int
    spread_hidden: bool
    spread: float | None


@dataclass
class QuintileSummary:
    n: int
    hidden: bool
    means: tuple[float | None, ...]
    monotonic: int


@dataclass
class BucketCell:
    label: str
    n: int
    hidden: bool
    mean: float | None


@dataclass
class FactorCurve:
    name: str
    buckets: tuple[BucketCell, ...]


@dataclass
class HorizonIc:
    horizon: int
    n: int
    signals: list[SignalSummary]


@dataclass
class WalkForwardResult:
    ok: bool
    message: str
    windows: list[Window] = field(default_factory=list)
    slices: list[SliceSummary] = field(default_factory=list)
    skipped: int = 0
    gaps: int = 0
    gap_notes: list[str] = field(default_factory=list)
    missing_yahoo: list[str] = field(default_factory=list)
    anchor_note: str = ""
    signals: list[SignalSummary] = field(default_factory=list)
    quintiles: QuintileSummary | None = None
    curves: list[FactorCurve] = field(default_factory=list)
    horizon_ics: list[HorizonIc] = field(default_factory=list)
    horizon: int = config.DECISION_HORIZON


def run_walk_forward(refresh: bool = False, progress=None) -> WalkForwardResult:
    symbols = research_symbols()
    frames = data.load_many([f"{symbol}.CA" for symbol in symbols], refresh=refresh, progress=progress)
    stock_frames = {symbol: frames.get(f"{symbol}.CA", data.empty_frame()) for symbol in symbols}
    calendar = quality.trading_calendar(stock_frames)
    healthy, message = quality.assess_calendar(calendar)
    intervals = membership_intervals()
    if not healthy:
        return WalkForwardResult(ok=False, message=message, gap_notes=gap_notes(intervals), anchor_note=ANCHOR_NOTE)

    prices = pd.DataFrame({symbol: _series(stock_frames[symbol], "adj", calendar) for symbol in symbols})
    returns = prices.pct_change()
    _, benches = select_benchmarks(None, returns)
    member, known = membership_frame(pd.DatetimeIndex(prices.index), symbols, intervals)
    scores: dict[str, pd.Series] = {}
    eligible: dict[str, pd.Series] = {}
    factors: dict[str, dict[str, pd.Series]] = {name: {} for name in SIGNAL_COLUMNS}
    missing = [symbol for symbol in symbols if stock_frames[symbol] is None or stock_frames[symbol].empty]
    for symbol in symbols:
        frame = stock_frames[symbol]
        high = _series(frame, "high", calendar)
        low = _series(frame, "low", calendar)
        volume = _series(frame, "volume", calendar)
        close = _series(frame, "close", calendar)
        panel = score_panel(prices[symbol], high, low, volume, benches[symbol])
        scores[symbol] = panel["score"]
        eligible[symbol] = eligible_mask(prices[symbol], close, high, low, volume) & member[symbol]
        for name in SIGNAL_COLUMNS:
            factors[name][symbol] = panel[name]
    return evaluate_walk_forward(
        pd.DataFrame(scores),
        returns,
        pd.DataFrame(eligible),
        known=known,
        factors={name: pd.DataFrame(columns) for name, columns in factors.items()},
        gap_notes_text=gap_notes(intervals),
        missing_yahoo=missing,
        anchor_note=ANCHOR_NOTE,
    )


def eligible_mask(
    adj: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    returns = adj.pct_change()
    recent_jump = (
        (returns.abs() > config.JUMP_ABS)
        .fillna(False)
        .rolling(config.JUMP_LOOKBACK, min_periods=1)
        .max()
        .astype(bool)
    )
    traded = close * volume
    liquid = traded.rolling(config.LIQUIDITY_WINDOW, min_periods=config.LIQUIDITY_WINDOW).median() >= config.MIN_MEDIAN_VALUE
    volume_missing = volume.fillna(0).rolling(config.LIQUIDITY_WINDOW, min_periods=config.LIQUIDITY_WINDOW).sum() <= 0
    span = (high - low) / close.replace(0, pd.NA)
    at_limit = (returns.abs() >= config.LIMIT_MOVE) & (span <= config.LIMIT_RANGE)
    limit_fraction = at_limit.fillna(False).rolling(config.LIMIT_WINDOW, min_periods=config.LIMIT_WINDOW).mean()
    limits_ok = limit_fraction <= config.MAX_LIMIT_FRACTION
    missing_ok = adj.isna().rolling(config.MISSING_WINDOW, min_periods=config.MISSING_WINDOW).mean() <= config.MAX_MISSING_RATIO
    enough_history = adj.notna().cumsum() >= config.MIN_BARS
    return adj.notna() & enough_history & missing_ok & ~recent_jump & liquid & ~volume_missing & limits_ok


def decision_origins(index: pd.DatetimeIndex, horizon: int, years: int) -> list[int]:
    if horizon <= 0 or len(index) <= horizon:
        return []
    cutoff = pd.Timestamp(index[-1]) - pd.DateOffset(years=years)
    start = None
    last_origin = len(index) - horizon - 1
    for position, stamp in enumerate(index):
        if position > last_origin:
            break
        if pd.Timestamp(stamp) >= cutoff:
            start = position
            break
    if start is None:
        return []
    return list(range(start, last_origin + 1, horizon))


def evaluate_walk_forward(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
    horizon: int = config.DECISION_HORIZON,
    top_n: int = config.TOP_N,
    years: tuple[int, ...] = (3, 2),
    min_names: int = config.MIN_INCLUDED,
    min_rest: int = 3,
    cost: float = config.ROUND_TRIP_COST,
    min_n: int = config.MIN_N,
    known: pd.Series | None = None,
    factors: dict[str, pd.DataFrame] | None = None,
    gap_notes_text: list[str] | None = None,
    missing_yahoo: list[str] | None = None,
    anchor_note: str = "",
    min_quintile: int = 15,
    ic_horizons: tuple[int, ...] = config.IC_HORIZONS,
    buckets: dict[str, tuple[float, ...]] | None = None,
) -> WalkForwardResult:
    if scores.empty or not years:
        return WalkForwardResult(ok=False, message="مفيش بيانات كافية للباك تست.")
    outer = max(years)
    origins = decision_origins(pd.DatetimeIndex(scores.index), horizon, outer)
    windows: list[Window] = []
    skipped = 0
    gaps = 0
    observations: list[_Observation] = []
    for origin in origins:
        if known is not None and not bool(known.iloc[origin]):
            gaps += 1
            continue
        window = _window_at(scores, returns, eligible, origin, horizon, top_n, min_names, min_rest)
        if window is None:
            skipped += 1
        else:
            windows.append(window)
        observations.append(_observe(scores, returns, eligible, factors, origin, horizon, min_names, min_quintile))
    slices = [
        _summarize(windows, scores.index[-1], span, cost, min_n)
        for span in years
    ]
    signals, quintiles = _summarize_signals(observations, min_n)
    curves = _factor_curves(scores, returns, eligible, factors, origins, known, horizon, min_names, min_n, buckets)
    horizon_ics = [
        _horizon_ic(scores, returns, eligible, factors, known, span, outer, min_names, min_n)
        for span in ic_horizons
    ]
    if not windows and all(item.hidden for item in signals):
        return WalkForwardResult(
            ok=False,
            message="مفيش نافذة 20 جلسة عدّت فحص الجودة والعدد.",
            skipped=skipped,
            gaps=gaps,
            gap_notes=list(gap_notes_text or []),
            missing_yahoo=list(missing_yahoo or []),
            anchor_note=anchor_note,
            slices=slices,
            signals=signals,
            quintiles=quintiles,
            curves=curves,
            horizon_ics=horizon_ics,
            horizon=horizon,
        )
    return WalkForwardResult(
        ok=True,
        message="الباك تست جاهز. الأوزان ما اتغيرتش.",
        windows=windows,
        slices=slices,
        skipped=skipped,
        gaps=gaps,
        gap_notes=list(gap_notes_text or []),
        missing_yahoo=list(missing_yahoo or []),
        anchor_note=anchor_note,
        signals=signals,
        quintiles=quintiles,
        curves=curves,
        horizon_ics=horizon_ics,
        horizon=horizon,
    )


def _window_at(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
    origin: int,
    horizon: int,
    top_n: int,
    min_names: int,
    min_rest: int,
) -> Window | None:
    row = scores.iloc[origin]
    allowed = eligible.iloc[origin].fillna(False).astype(bool) & row.notna()
    names = [name for name in scores.columns if bool(allowed.get(name, False))]
    if len(names) < max(min_names, top_n + min_rest):
        return None
    order = row[names].sort_values(ascending=False, kind="mergesort")
    top_names = list(order.index[:top_n])
    rest_names = list(order.index[top_n:])
    top_returns = _complete(returns, top_names, origin, horizon)
    rest_returns = _complete(returns, rest_names, origin, horizon)
    if len(top_returns) < top_n or len(rest_returns) < min_rest:
        return None
    top_mean = float(sum(top_returns) / len(top_returns))
    rest_mean = float(sum(rest_returns) / len(rest_returns))
    return Window(
        date=str(pd.Timestamp(scores.index[origin]).date()),
        top=top_names,
        top_return=top_mean,
        rest_return=rest_mean,
        excess=top_mean - rest_mean,
    )


def _complete(returns: pd.DataFrame, names: list[str], origin: int, horizon: int) -> list[float]:
    values: list[float] = []
    for name in names:
        result = holding_return(returns[name], origin, horizon)
        if result is not None:
            values.append(result)
    return values


@dataclass
class _Observation:
    score_ic: float | None
    quintiles: tuple[float, ...] | None
    factor_ic: dict[str, float | None]
    factor_spread: dict[str, float | None]


def _observe(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
    factors: dict[str, pd.DataFrame] | None,
    origin: int,
    horizon: int,
    min_names: int,
    min_quintile: int,
) -> _Observation:
    allowed = eligible.iloc[origin].fillna(False).astype(bool) & scores.iloc[origin].notna()
    names = [name for name in scores.columns if bool(allowed.get(name, False))]
    excess = _forward_excess(returns, names, origin, horizon)
    if excess is None or len(excess) < min_names:
        return _Observation(None, None, {}, {})
    excess_series = pd.Series(excess)
    score_ic = _spearman(scores.iloc[origin].reindex(excess_series.index), excess_series)
    quintiles = _quintile_means(scores.iloc[origin].reindex(excess_series.index), excess_series, min_quintile)
    factor_ic: dict[str, float | None] = {}
    factor_spread: dict[str, float | None] = {}
    for name, frame in (factors or {}).items():
        signal = frame.iloc[origin].reindex(excess_series.index)
        factor_ic[name] = _spearman(signal, excess_series)
        factor_spread[name] = _tail_spread(signal, excess_series)
    return _Observation(score_ic, quintiles, factor_ic, factor_spread)


def _forward_excess(
    returns: pd.DataFrame,
    names: list[str],
    origin: int,
    horizon: int,
) -> dict[str, float] | None:
    values = {}
    for name in names:
        result = holding_return(returns[name], origin, horizon)
        if result is not None:
            values[name] = result
    if len(values) < 2:
        return None
    baseline = sum(values.values()) / len(values)
    return {name: value - baseline for name, value in values.items()}


def _spearman(signal: pd.Series, outcome: pd.Series) -> float | None:
    paired = pd.concat([signal, outcome], axis=1).dropna()
    if len(paired) < 8 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return None
    ranked = paired.rank()
    value = ranked.iloc[:, 0].corr(ranked.iloc[:, 1])
    if pd.isna(value):
        return None
    return float(value)


def _quintile_means(signal: pd.Series, outcome: pd.Series, minimum: int) -> tuple[float, ...] | None:
    frame = pd.concat([signal.rename("signal"), outcome.rename("outcome")], axis=1).dropna()
    if len(frame) < minimum:
        return None
    frame = frame.sort_values("signal", ascending=False, kind="mergesort")
    buckets: list[list[float]] = [[] for _ in range(5)]
    for position, value in enumerate(frame["outcome"]):
        buckets[min(5 * position // len(frame), 4)].append(float(value))
    if any(len(bucket) < 3 for bucket in buckets):
        return None
    return tuple(sum(bucket) / len(bucket) for bucket in buckets)


def _tail_spread(signal: pd.Series, outcome: pd.Series) -> float | None:
    frame = pd.concat([signal.rename("signal"), outcome.rename("outcome")], axis=1).dropna()
    if len(frame) < 10:
        return None
    frame = frame.sort_values("signal", ascending=False, kind="mergesort")
    width = max(len(frame) // 5, 1)
    top = float(frame["outcome"].iloc[:width].mean())
    bottom = float(frame["outcome"].iloc[-width:].mean())
    return top - bottom


def _summarize_signals(
    observations: list[_Observation],
    min_n: int,
) -> tuple[list[SignalSummary], QuintileSummary]:
    signals = [_signal_summary("score", [item.score_ic for item in observations], _score_spreads(observations), min_n)]
    names = []
    for item in observations:
        for name in item.factor_ic:
            if name not in names:
                names.append(name)
    for name in names:
        signals.append(
            _signal_summary(
                name,
                [item.factor_ic.get(name) for item in observations],
                [item.factor_spread.get(name) for item in observations],
                min_n,
            )
        )
    quintile_rows = [item.quintiles for item in observations if item.quintiles is not None]
    if len(quintile_rows) < min_n:
        quintiles = QuintileSummary(len(quintile_rows), True, (None, None, None, None, None), 0)
    else:
        means = tuple(
            float(sum(row[bucket] for row in quintile_rows) / len(quintile_rows))
            for bucket in range(5)
        )
        monotonic = sum(1 for row in quintile_rows if all(row[pos] > row[pos + 1] for pos in range(4)))
        quintiles = QuintileSummary(len(quintile_rows), False, means, monotonic)
    return signals, quintiles


def _score_spreads(observations: list[_Observation]) -> list[float | None]:
    spreads: list[float | None] = []
    for item in observations:
        if item.quintiles is None:
            spreads.append(None)
        else:
            spreads.append(item.quintiles[0] - item.quintiles[-1])
    return spreads


def _signal_summary(name: str, values: list[float | None], spreads: list[float | None], min_n: int) -> SignalSummary:
    finite = [value for value in values if value is not None]
    finite_spreads = [value for value in spreads if value is not None]
    ic_hidden = len(finite) < min_n
    spread_hidden = len(finite_spreads) < min_n
    return SignalSummary(
        name=name,
        n=len(finite),
        hidden=ic_hidden,
        mean=None if ic_hidden else float(sum(finite) / len(finite)),
        median=None if ic_hidden else float(pd.Series(finite).median()),
        positive=None if ic_hidden else float(sum(value > 0 for value in finite) / len(finite)),
        spread_n=len(finite_spreads),
        spread_hidden=spread_hidden,
        spread=None if spread_hidden else float(sum(finite_spreads) / len(finite_spreads)),
    )


def _horizon_ic(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
    factors: dict[str, pd.DataFrame] | None,
    known: pd.Series | None,
    horizon: int,
    years: int,
    min_names: int,
    min_n: int,
) -> HorizonIc:
    origins = decision_origins(pd.DatetimeIndex(scores.index), horizon, years)
    observations: list[_Observation] = []
    for origin in origins:
        if known is not None and not bool(known.iloc[origin]):
            continue
        observations.append(_observe(scores, returns, eligible, factors, origin, horizon, min_names, 15))
    signals, _ = _summarize_signals(observations, min_n)
    score = next((item for item in signals if item.name == "score"), None)
    return HorizonIc(horizon=horizon, n=0 if score is None else score.n, signals=signals)


def _factor_curves(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
    factors: dict[str, pd.DataFrame] | None,
    origins: list[int],
    known: pd.Series | None,
    horizon: int,
    min_names: int,
    min_n: int,
    buckets: dict[str, tuple[float, ...]] | None,
) -> list[FactorCurve]:
    cuts = buckets if buckets is not None else config.FACTOR_BUCKETS
    pooled: dict[str, list[list[float]]] = {
        name: [[] for _ in range(len(edges) + 1)] for name, edges in cuts.items()
    }
    for origin in origins:
        if known is not None and not bool(known.iloc[origin]):
            continue
        allowed = eligible.iloc[origin].fillna(False).astype(bool) & scores.iloc[origin].notna()
        names = [name for name in scores.columns if bool(allowed.get(name, False))]
        excess = _forward_excess(returns, names, origin, horizon)
        if excess is None or len(excess) < min_names:
            continue
        for factor, edges in cuts.items():
            if factor == "score":
                signal = scores.iloc[origin]
            else:
                if factors is None or factor not in factors:
                    continue
                signal = factors[factor].iloc[origin]
            for symbol, value in excess.items():
                point = signal.get(symbol)
                if point is None or pd.isna(point):
                    continue
                pooled[factor][_bucket_index(factor, float(point), edges)].append(value)
    curves: list[FactorCurve] = []
    for factor, edges in cuts.items():
        labels = _bucket_labels(factor, edges)
        cells = []
        for label, values in zip(labels, pooled[factor]):
            count = len(values)
            hidden = count < min_n
            cells.append(
                BucketCell(
                    label=label,
                    n=count,
                    hidden=hidden,
                    mean=None if hidden else float(sum(values) / count),
                )
            )
        curves.append(FactorCurve(name=factor, buckets=tuple(cells)))
    return curves


def _bucket_index(factor: str, value: float, edges: tuple[float, ...]) -> int:
    if factor == "extension_penalty":
        if value <= 0:
            return 0
        if value < 5:
            return 1
        if value < 15:
            return 2
        if value < 25:
            return 3
        return 4
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def _bucket_labels(factor: str, edges: tuple[float, ...]) -> tuple[str, ...]:
    if factor.startswith("dist_"):
        labels = [f"أقل من {edges[0] * 100:+.0f}%"]
        for left, right in zip(edges, edges[1:]):
            labels.append(f"{left * 100:+.0f}% إلى {right * 100:+.0f}%")
        labels.append(f"أكثر من {edges[-1] * 100:+.0f}%")
        return tuple(labels)
    if factor == "extension_penalty":
        return ("0", "أكثر من 0 إلى 5", "5 إلى 15", "15 إلى 25", "25 فأعلى")
    labels = [f"أقل من {edges[0]:g}"]
    for left, right in zip(edges, edges[1:]):
        labels.append(f"{left:g} إلى {right:g}")
    labels.append(f"{edges[-1]:g} فأعلى")
    return tuple(labels)


def _summarize(
    windows: list[Window],
    end,
    years: int,
    cost: float,
    min_n: int,
) -> SliceSummary:
    cutoff = pd.Timestamp(end) - pd.DateOffset(years=years)
    chosen = [window for window in windows if pd.Timestamp(window.date) >= cutoff]
    count = len(chosen)
    if count < min_n:
        return SliceSummary(years, count, True, None, None, None, None, None, None)
    excess = pd.Series([window.excess for window in chosen], dtype=float)
    gross = float(excess.mean())
    return SliceSummary(
        years=years,
        n=count,
        hidden=False,
        top=float(sum(window.top_return for window in chosen) / count),
        rest=float(sum(window.rest_return for window in chosen) / count),
        gross=gross,
        net=gross - cost,
        beat=float((excess > 0).mean()),
        median=float(excess.median()),
    )
