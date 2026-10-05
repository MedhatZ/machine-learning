"""تنزيل أسعار ياهو مع كاش محلي."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from poesa import config

EMPTY_COLUMNS = ["open", "high", "low", "close", "adj", "volume"]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_COLUMNS)


def naive_dates(index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "_").replace("/", "_")
    return config.CACHE_DIR / f"{safe}.pkl"


def normalize(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_frame()
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=lambda column: str(column).strip().lower().replace(" ", "_"))
    if "close" not in frame.columns:
        return empty_frame()
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    for column in ("open", "high", "low", "volume"):
        if column not in frame.columns:
            frame[column] = pd.NA
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "adj": pd.to_numeric(frame["adj_close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce"),
        }
    )
    out.index = naive_dates(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["adj", "close"])
    out = out[out["adj"] > 0]
    out.index.name = "date"
    return out


def _fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = pickle.loads(path.read_bytes())
        fetched = datetime.fromisoformat(payload["fetched_at"])
    except (OSError, pickle.UnpicklingError, KeyError, ValueError, TypeError):
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fetched
    frame = payload.get("frame")
    return age < timedelta(hours=config.CACHE_HOURS) and isinstance(frame, pd.DataFrame) and not frame.empty


def _read_cache(path: Path) -> pd.DataFrame | None:
    try:
        payload = pickle.loads(path.read_bytes())
        frame = payload.get("frame")
    except (OSError, pickle.UnpicklingError, KeyError, TypeError):
        return None
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame
    return None


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "frame": frame}
    path.write_bytes(pickle.dumps(payload))


def download(symbol: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for _ in range(2):
        try:
            raw = yf.Ticker(symbol).history(start=config.HISTORY_START, auto_adjust=False)
            frame = normalize(raw)
            if not frame.empty:
                return frame
        except Exception as exc:  # شبكة أو شكل رد غير متوقع
            last_error = exc
    if last_error is not None:
        return empty_frame()
    return empty_frame()


def load_symbol(symbol: str, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(symbol)
    if not refresh and _fresh(path):
        cached = _read_cache(path)
        if cached is not None:
            return cached
    frame = download(symbol)
    if not frame.empty:
        _write_cache(path, frame)
    return frame


def load_many(symbols: list[str], refresh: bool = False, progress=None) -> dict[str, pd.DataFrame]:
    pending = []
    loaded: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _cache_path(symbol)
        if not refresh and _fresh(path):
            cached = _read_cache(path)
            if cached is not None:
                loaded[symbol] = cached
                if progress:
                    progress(symbol, True)
                continue
        pending.append(symbol)

    if not pending:
        return loaded

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, symbol): symbol for symbol in pending}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception:
                frame = empty_frame()
            if not frame.empty:
                _write_cache(_cache_path(symbol), frame)
            loaded[symbol] = frame
            if progress:
                progress(symbol, False)
    return loaded
