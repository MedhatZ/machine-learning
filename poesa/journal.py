"""دفتر توقعات: نحفظ أعلى 5 Setup، وبعد 20 جلسة نراجع هل سبقوا الباقي."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from poesa import config, data, quality
from poesa.engine import ScanResult, StockView, _series, run_scan
from poesa.stats import holding_return
from poesa.universe import UNIVERSE


JOURNAL_DIR = config.JOURNAL_DIR


@dataclass
class SaveResult:
    ok: bool
    path: Path | None
    message: str
    created: bool


@dataclass
class NameOutcome:
    symbol: str
    name: str
    score: float | None
    actual: float | None


@dataclass
class ReviewItem:
    session_date: str
    status: str
    top: list[NameOutcome]
    top_return: float | None
    rest_return: float | None
    excess: float | None
    correct: bool | None
    message: str


@dataclass
class ReviewReport:
    ok: bool
    message: str
    items: list[ReviewItem]
    pending: int
    graded: int
    correct: int
    wrong: int


def save_snapshot(
    scan: ScanResult,
    directory: Path | None = None,
    force: bool = False,
    weekly: bool = False,
    min_gap_sessions: int = 5,
) -> SaveResult:
    folder = directory or JOURNAL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    if not scan.ok or not scan.ranked or not scan.calendar_end:
        return SaveResult(False, None, "مفيش ترتيب جاهز يتحفظ.", False)
    path = folder / f"{scan.calendar_end}.json"
    if path.exists() and not force:
        return SaveResult(True, path, f"اللقطة موجودة لنفس الجلسة {scan.calendar_end}.", False)
    if weekly and not force:
        latest = latest_session(folder)
        if latest is not None:
            gap = _session_gap(latest, scan.calendar_end)
            if gap is not None and gap < min_gap_sessions:
                return SaveResult(
                    True,
                    folder / f"{latest}.json",
                    f"آخر لقطة {latest}، ولسه عدّى {gap} جلسات بس. الأسبوعي بيستنى {min_gap_sessions}.",
                    False,
                )
    payload = {
        "session_date": scan.calendar_end,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "horizon": config.DECISION_HORIZON,
        "top_n": config.TOP_N,
        "weights": dict(config.WEIGHTS),
        "baseline_label": scan.baseline_label,
        "basket_return_20d": scan.basket_return_20d,
        "top": [_stock_row(stock) for stock in scan.ranked[: config.TOP_N]],
        "ranked": [_stock_row(stock) for stock in scan.ranked],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return SaveResult(True, path, f"اتحفظت لقطة {scan.calendar_end}.", True)


def list_snapshots(directory: Path | None = None) -> list[dict]:
    folder = directory or JOURNAL_DIR
    if not folder.exists():
        return []
    rows = []
    for path in sorted(folder.glob("*.json")):
        payload = _read(path)
        if payload is None:
            continue
        rows.append(
            {
                "session_date": payload.get("session_date", path.stem),
                "saved_at": payload.get("saved_at"),
                "top": [item.get("symbol") for item in payload.get("top", [])],
                "path": str(path),
            }
        )
    return rows


def latest_session(directory: Path | None = None) -> str | None:
    rows = list_snapshots(directory)
    if not rows:
        return None
    return rows[-1]["session_date"]


def load_return_panel(
    refresh: bool = False,
    progress=None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex] | tuple[None, None]:
    symbols = sorted({item.symbol for item in UNIVERSE})
    frames = data.load_many([f"{symbol}.CA" for symbol in symbols], refresh=refresh, progress=progress)
    stock_frames = {symbol: frames.get(f"{symbol}.CA", data.empty_frame()) for symbol in symbols}
    calendar = quality.trading_calendar(stock_frames)
    if len(calendar) == 0:
        return None, None
    prices = pd.DataFrame({symbol: _series(stock_frames[symbol], "adj", calendar) for symbol in symbols})
    return prices.pct_change(), calendar


def review_snapshots(
    directory: Path | None = None,
    refresh: bool = False,
    progress=None,
    horizon: int = config.DECISION_HORIZON,
    month: str | None = None,
) -> ReviewReport:
    folder = directory or JOURNAL_DIR
    rows = list_snapshots(folder)
    if not rows:
        return ReviewReport(False, "مفيش لقطات محفوظة.", [], 0, 0, 0, 0)
    returns, calendar = load_return_panel(refresh=refresh, progress=progress)
    if returns is None or calendar is None:
        return ReviewReport(False, "مفيش تقويم أسعار نراجع عليه.", [], 0, 0, 0, 0)
    items: list[ReviewItem] = []
    for row in rows:
        payload = _read(Path(row["path"]))
        if payload is None:
            continue
        if month and not str(payload.get("session_date", "")).startswith(month):
            continue
        items.append(_grade(payload, returns, calendar, horizon))
    pending = sum(1 for item in items if item.status == "pending")
    graded = [item for item in items if item.status == "graded"]
    correct = sum(1 for item in graded if item.correct)
    wrong = len(graded) - correct
    if not items:
        return ReviewReport(False, "مفيش لقطات في الشهر المطلوب.", [], 0, 0, 0, 0)
    return ReviewReport(
        True,
        "المراجعة جاهزة.",
        items,
        pending,
        len(graded),
        correct,
        wrong,
    )


def save_from_scan(
    refresh: bool = False,
    force: bool = False,
    weekly: bool = False,
    progress=None,
) -> tuple[ScanResult, SaveResult]:
    scan = run_scan(refresh=refresh, progress=progress)
    return scan, save_snapshot(scan, force=force, weekly=weekly)


def _stock_row(stock: StockView) -> dict:
    return {
        "symbol": stock.symbol,
        "name": stock.name,
        "sector": stock.sector,
        "score": None if stock.score is None else round(float(stock.score), 2),
        "band": stock.band,
        "price": stock.price,
        "extension_penalty": stock.extension_penalty,
        "dist_sma50": stock.dist_sma50,
        "dist_sma200": stock.dist_sma200,
        "factors": {name: None if value is None else round(float(value), 2) for name, value in stock.factors.items()},
    }


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _session_gap(older: str, newer: str) -> int | None:
    try:
        left = pd.Timestamp(older)
        right = pd.Timestamp(newer)
    except (TypeError, ValueError):
        return None
    if right < left:
        return None
    sessions = pd.bdate_range(left, right)
    return max(len(sessions) - 1, 0)


def _grade(
    payload: dict,
    returns: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
    top_symbols: list[str] | None = None,
    rest_symbols: list[str] | None = None,
) -> ReviewItem:
    session = str(payload.get("session_date"))
    top_rows = payload.get("top") or []
    ranked_rows = payload.get("ranked") or top_rows
    if top_symbols is None:
        top_symbols = [item["symbol"] for item in top_rows if item.get("symbol")]
    if rest_symbols is None:
        rest_symbols = [item["symbol"] for item in ranked_rows if item.get("symbol") not in set(top_symbols)]
    names = {item.get("symbol"): item for item in ranked_rows if item.get("symbol")}
    stamp = pd.Timestamp(session)
    if stamp not in calendar:
        return ReviewItem(session, "missing", [], None, None, None, None, "تاريخ اللقطة مش موجود في تقويم الأسعار.")
    origin = int(calendar.get_loc(stamp))
    if origin + horizon >= len(calendar):
        left = len(calendar) - origin - 1
        return ReviewItem(
            session,
            "pending",
            [
                NameOutcome(symbol, names.get(symbol, {}).get("name", symbol), names.get(symbol, {}).get("score"), None)
                for symbol in top_symbols
            ],
            None,
            None,
            None,
            None,
            f"لسه باقي {horizon - left} جلسة قبل التقييم.",
        )
    top_actual = []
    for symbol in top_symbols:
        row = names.get(symbol, {})
        actual = None if symbol not in returns.columns else holding_return(returns[symbol], origin, horizon)
        top_actual.append(NameOutcome(symbol, row.get("name", symbol), row.get("score"), actual))
    top_values = [item.actual for item in top_actual if item.actual is not None]
    rest_values = []
    for symbol in rest_symbols:
        if symbol not in returns.columns:
            continue
        value = holding_return(returns[symbol], origin, horizon)
        if value is not None:
            rest_values.append(value)
    if len(top_values) < max(1, len(top_symbols) // 2) or len(rest_values) < 3:
        return ReviewItem(session, "incomplete", top_actual, None, None, None, None, "مسارات ناقصة بعد تاريخ اللقطة.")
    top_mean = float(sum(top_values) / len(top_values))
    rest_mean = float(sum(rest_values) / len(rest_values))
    excess = top_mean - rest_mean
    correct = excess > 0
    label = "صح" if correct else "غلط"
    return ReviewItem(
        session,
        "graded",
        top_actual,
        top_mean,
        rest_mean,
        excess,
        correct,
        f"{label}: أعلى {len(top_symbols)} {top_mean:+.1%} · الباقي {rest_mean:+.1%} · التفوق {excess:+.1%}",
    )
