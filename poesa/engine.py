"""تجميع الفحص والترتيب."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from poesa import config, data, quality
from poesa.data import naive_dates
from poesa.benchmark import select_benchmarks, trailing_compound
from poesa.scoring import band_name, score_panel
from poesa.stats import excess_table
from poesa.universe import UNIVERSE, Listing


@dataclass
class StockView:
    symbol: str
    yahoo: str
    name: str
    sector: str
    included: bool
    reasons: list[str]
    price: float | None = None
    as_of: str | None = None
    score: float | None = None
    band: str | None = None
    factors: dict[str, float] = field(default_factory=dict)
    atr_pct: float | None = None
    drawdown: float | None = None
    ann_vol: float | None = None
    dist_sma50: float | None = None
    dist_sma200: float | None = None
    extension_penalty: float | None = None
    stats: dict[int, dict] = field(default_factory=dict)
    chart: pd.DataFrame | None = None


@dataclass
class ScanResult:
    ok: bool
    message: str
    baseline: str
    baseline_note: str
    baseline_label: str
    case30_rows: int
    case30_close: float | None
    case30_date: str | None
    calendar_end: str | None
    basket_return_20d: float | None
    generated_at: str
    stocks: list[StockView]

    @property
    def ranked(self) -> list[StockView]:
        return [stock for stock in self.stocks if stock.included and stock.score is not None]

    @property
    def excluded(self) -> list[StockView]:
        return [stock for stock in self.stocks if not stock.included]


def run_scan(refresh: bool = False, progress=None) -> ScanResult:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    symbols = [config.INDEX_SYMBOL, *[item.yahoo for item in UNIVERSE]]

    def _progress(symbol: str, cached: bool) -> None:
        if progress:
            progress(symbol, cached)

    frames = data.load_many(symbols, refresh=refresh, progress=_progress)
    index_frame = frames.get(config.INDEX_SYMBOL, data.empty_frame())
    case_rows, case_close, case_date = _index_snapshot(index_frame)
    stock_frames = {item.symbol: frames.get(item.yahoo, data.empty_frame()) for item in UNIVERSE}
    calendar = quality.trading_calendar(stock_frames)
    healthy, calendar_message = quality.assess_calendar(calendar)
    calendar_end = None if len(calendar) == 0 else str(pd.Timestamp(calendar[-1]).date())

    if not healthy:
        return _stopped(
            calendar_message,
            case_rows,
            case_close,
            case_date,
            calendar_end,
            generated,
            _views_without_scores(stock_frames),
        )

    views: list[StockView] = []
    included_symbols: list[str] = []
    for listing in UNIVERSE:
        frame = stock_frames[listing.symbol]
        passed, reasons = quality.assess(frame, calendar)
        view = _base_view(listing, frame, passed, reasons)
        views.append(view)
        if passed:
            included_symbols.append(listing.symbol)

    if len(included_symbols) < config.MIN_INCLUDED:
        message = f"الأسهم اللي عدّت فحص الجودة {len(included_symbols)}، وأقل عدد للترتيب {config.MIN_INCLUDED}."
        return _stopped(message, case_rows, case_close, case_date, calendar_end, generated, views)

    prices = pd.DataFrame({symbol: _series(stock_frames[symbol], "adj", calendar) for symbol in included_symbols})
    returns = prices.pct_change()
    index_returns = None
    if case_rows >= config.MIN_CALENDAR_BARS:
        index_returns = _series(index_frame, "adj", calendar).pct_change()
    kind, benches = select_benchmarks(index_returns, returns)
    note, label = _baseline_copy(kind, case_rows, case_date, case_close)
    basket_20 = _basket_return(returns)

    ranked: list[StockView] = []
    by_symbol = {view.symbol: view for view in views}
    for symbol in included_symbols:
        frame = stock_frames[symbol]
        price = prices[symbol]
        bench = benches[symbol]
        panel = score_panel(
            price,
            _series(frame, "high", calendar),
            _series(frame, "low", calendar),
            _series(frame, "volume", calendar),
            bench,
        )
        view = by_symbol[symbol]
        finite_positions = panel["score"].notna().to_numpy().nonzero()[0]
        if len(finite_positions) == 0:
            view.included = False
            view.reasons = ["الدرجة مش محسوبة"]
            continue
        last_pos = int(finite_positions[-1])
        if len(panel) - 1 - last_pos > config.MAX_SCORE_STALE:
            view.included = False
            view.reasons = ["الدرجة مش محسوبة على آخر الجلسات"]
            continue
        current = float(panel["score"].iloc[last_pos])
        row = panel.iloc[last_pos]
        view.score = current
        view.band = band_name(current)
        view.factors = {name: float(row[name]) for name in config.WEIGHTS}
        view.atr_pct = _optional_float(row["atr_pct"])
        view.drawdown = _optional_float(row["drawdown"])
        view.ann_vol = _optional_float(row["ann_vol"])
        view.dist_sma50 = _optional_float(row["dist_sma50"])
        view.dist_sma200 = _optional_float(row["dist_sma200"])
        view.extension_penalty = _optional_float(row["extension_penalty"])
        view.stats = excess_table(panel["score"], returns[symbol], bench, current)
        view.chart = _chart(frame, panel, calendar)
        ranked.append(view)

    ranked.sort(key=lambda item: item.score or -1, reverse=True)
    excluded = [view for view in views if view.symbol not in {item.symbol for item in ranked}]
    if len(ranked) < config.MIN_INCLUDED:
        message = f"الأسهم اللي اتحسبت لها درجة {len(ranked)}، وأقل عدد للترتيب {config.MIN_INCLUDED}."
        return _stopped(message, case_rows, case_close, case_date, calendar_end, generated, ranked + excluded, kind, note, label, basket_20)

    return ScanResult(
        ok=True,
        message="الترتيب جاهز.",
        baseline=kind,
        baseline_note=note,
        baseline_label=label,
        case30_rows=case_rows,
        case30_close=case_close,
        case30_date=case_date,
        calendar_end=calendar_end,
        basket_return_20d=basket_20,
        generated_at=generated,
        stocks=ranked + excluded,
    )


def _baseline_copy(kind: str, rows: int, date: str | None, close: float | None) -> tuple[str, str]:
    quote = "من غير قراءة" if close is None or date is None else f"آخر قراءة {date} عند {close:,.0f}"
    checked = f"فحص ^CASE30 على ياهو: {rows} شمعة، {quote}."
    if kind == "official":
        return f"{checked} خط الأساس هو المؤشر نفسه.", "المؤشر"
    return (
        f"{checked} التاريخ ده مش كافي، فالعائد الزائد والقوة النسبية اتحسبوا ضد "
        "متوسط متساوي الوزن لباقي أسهم القائمة، من غير السهم نفسه. ده مش المؤشر الرسمي الموزون.",
        "باقي القائمة",
    )


def _basket_return(returns: pd.DataFrame) -> float | None:
    if returns.empty:
        return None
    peers = returns.notna().sum(axis=1)
    basket = returns.mean(axis=1).where(peers >= config.MIN_PEERS)
    trail = trailing_compound(basket, 20).dropna()
    if trail.empty:
        return None
    return float(trail.iloc[-1])


def _chart(frame: pd.DataFrame, panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    close = _series(frame, "close", calendar)
    volume = _series(frame, "volume", calendar)
    chart = pd.DataFrame(
        {
            "close": close,
            "volume": volume,
            "sma20": panel["sma20"],
            "sma50": panel["sma50"],
            "sma200": panel["sma200"],
            "rsi": panel["rsi"],
        }
    )
    return chart.dropna(subset=["close"]).tail(180)


def _views_without_scores(frames: dict[str, pd.DataFrame]) -> list[StockView]:
    views = []
    for listing in UNIVERSE:
        frame = frames.get(listing.symbol, data.empty_frame())
        reason = [] if frame is not None and not frame.empty else ["لا توجد بيانات من المصدر"]
        views.append(_base_view(listing, frame, False, reason or ["الترتيب واقف قبل حساب الدرجة"]))
    return views


def _base_view(listing: Listing, frame: pd.DataFrame, included: bool, reasons: list[str]) -> StockView:
    price = None
    as_of = None
    chart = None
    if frame is not None and not frame.empty:
        price = float(frame["close"].iloc[-1])
        as_of = str(pd.Timestamp(frame.index[-1]).date())
        chart = frame[["close", "volume"]].tail(180).copy()
    return StockView(
        symbol=listing.symbol,
        yahoo=listing.yahoo,
        name=listing.name,
        sector=listing.sector,
        included=included,
        reasons=list(reasons),
        price=price,
        as_of=as_of,
        chart=chart,
    )


def _series(frame: pd.DataFrame, column: str, calendar: pd.DatetimeIndex) -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series(index=calendar, dtype=float)
    values = frame[column].copy()
    values.index = naive_dates(values.index)
    values = values[~values.index.duplicated(keep="last")]
    return pd.to_numeric(values.reindex(calendar), errors="coerce")


def _index_snapshot(frame: pd.DataFrame) -> tuple[int, float | None, str | None]:
    if frame is None or frame.empty:
        return 0, None, None
    return len(frame), float(frame["close"].iloc[-1]), str(pd.Timestamp(frame.index[-1]).date())


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _stopped(
    message: str,
    case_rows: int,
    case_close: float | None,
    case_date: str | None,
    calendar_end: str | None,
    generated: str,
    stocks: list[StockView],
    kind: str = "none",
    note: str = "",
    label: str = "خط الأساس",
    basket_20: float | None = None,
) -> ScanResult:
    if not note:
        note, label = _baseline_copy("basket" if case_rows < config.MIN_CALENDAR_BARS else "official", case_rows, case_date, case_close)
        kind = "none"
    return ScanResult(
        ok=False,
        message=message,
        baseline=kind,
        baseline_note=note,
        baseline_label=label,
        case30_rows=case_rows,
        case30_close=case_close,
        case30_date=case_date,
        calendar_end=calendar_end,
        basket_return_20d=basket_20,
        generated_at=generated,
        stocks=stocks,
    )
