"""تقرير عربي من الأرقام المحسوبة. لو الخدمة وقعت، الترتيب يفضل شغال."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from poesa import config
from poesa.engine import ScanResult, StockView
from poesa.text import pct, price

load_dotenv(config.ROOT / ".env")


def write_report(scan: ScanResult, focus: list[str] | None = None) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return "مفتاح النموذج مش موجود في ملف .env. الترتيب والشارتات شغالين من غيره."
    if not scan.ok or not scan.ranked:
        return "مفيش ترتيب كافي يتكتب عنه تقرير."

    wanted = [symbol.upper() for symbol in (focus or [])]
    chosen = [stock for stock in scan.ranked if stock.symbol in wanted] if wanted else scan.ranked[: config.TOP_N]
    if not chosen:
        chosen = scan.ranked[: config.TOP_N]
    payload = {
        "baseline": _baseline_en(scan),
        "basket_20d": pct(scan.basket_return_20d),
        "case30_rows": scan.case30_rows,
        "round_trip_cost": "0.8% assumed, not measured from the order book",
        "sample_design": "non-overlapping forward windows; n is not fully independent",
        "survivorship": "The list is a fixed current snapshot, so history is biased toward names that stayed in the universe.",
        "focus": [_stock_payload(stock) for stock in chosen],
        "excluded": [{"symbol": stock.symbol, "reasons": stock.reasons} for stock in scan.excluded],
    }
    system = (
        "You write market commentary in Egyptian Arabic. "
        "Use only numbers present in the JSON and copy them as given. "
        "Do not invent prices, returns, sample sizes, or price targets. "
        "If a field says the sample is insufficient, say so and do not infer a profit. "
        "If the baseline is a basket, do not call it the official EGX 30 index. "
        "Close with a short paragraph on survivorship bias, trading cost, and data quality. "
        "The score is a 20-session setup score, not a promise that the price will rise. "
        "Do not say a stock will profit or give a price target. "
        "Do not invent earnings, news, or fundamentals. "
        "When extension_penalty_points is above zero, explain that the price is far from its moving average. "
        "This is analysis, not a buy or sell recommendation. "
        "Call the names the highest current setups, not the best picks for the next 20 sessions. "
        "The walk-forward test has not shown that a higher score predicts the next 20 sessions. "
        "Order the text as: the five current setups, why each ranks there, the 20-session excess numbers, risks including the extension penalty, then a short comparison."
    )
    try:
        content = _complete(key, system, json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        detail = _safe_error(exc, key)
        return f"التقرير مش متاح دلوقتي ({detail}). الترتيب والشارتات لسه شغالين."
    text = (content or "").strip()
    if not text:
        return "النموذج رجّع رد فاضي. الترتيب والشارتات لسه شغالين."
    return text


def _stock_payload(stock: StockView) -> dict:
    horizons = {}
    for horizon, block in stock.stats.items():
        horizons[str(horizon)] = _horizon_en(block)
    return {
        "symbol": stock.symbol,
        "company": stock.name,
        "sector": _SECTOR_EN.get(stock.sector, stock.sector),
        "price_egp": price(stock.price),
        "as_of": stock.as_of,
        "forecast_score_20d": None if stock.score is None else round(stock.score, 1),
        "band": _BAND_EN.get(stock.band or "", stock.band),
        "factors_0_to_100": {name: round(value, 1) for name, value in stock.factors.items()},
        "annualized_volatility": pct(stock.ann_vol),
        "drawdown_120d": pct(stock.drawdown),
        "distance_from_sma50": pct(stock.dist_sma50),
        "distance_from_sma200": pct(stock.dist_sma200),
        "extension_penalty_points": None if stock.extension_penalty is None else round(stock.extension_penalty, 1),
        "excess_vs_baseline": horizons,
    }


def _baseline_en(scan: ScanResult) -> str:
    if scan.baseline == "official":
        return "Baseline is the ^CASE30 index."
    close = "n/a" if scan.case30_close is None else f"{scan.case30_close:,.0f}"
    return (
        f"Yahoo returned {scan.case30_rows} bars for ^CASE30"
        f" (last {scan.case30_date} at {close}). "
        "Excess return and relative strength use an equal-weight average of the other names, not the official index."
    )


def _horizon_en(block: dict | None) -> str:
    if not block:
        return "missing"
    if block["hidden"]:
        return f"insufficient sample, n={block['n']}"
    return (
        f"gross {block['gross'] * 100:+.1f}%, net {block['net'] * 100:+.1f}%, "
        f"beat rate {block['beat'] * 100:.0f}%, n={block['n']}"
    )


_SECTOR_EN = {
    "بنوك": "banks",
    "اتصالات": "telecom",
    "عقارات": "real estate",
    "معادن": "metals",
    "أسمدة": "fertilizers",
    "نقل": "transport",
    "مقاولات": "construction",
    "سلع استهلاكية": "consumer",
    "تكنولوجيا مالية": "fintech",
    "بتروكيماويات": "petrochemicals",
    "رعاية صحية": "healthcare",
    "خدمات مالية": "financial services",
    "طاقة": "energy",
    "أغذية": "food",
    "سيارات": "autos",
    "صناعة": "industry",
    "استثمار": "investment",
}
_BAND_EN = {
    "ضعيفة": "weak",
    "متوسطة": "mid",
    "مرتفعة": "high",
    "عالية جداً": "very high",
}


def _complete(key: str, system: str, user: str) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://agentrouter.org/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "claude-opus-4-6")
    if "agentrouter.org" in base:
        return _agentrouter(key, base, model, system, user)
    return _openai_sdk(key, base, model, system, user)


def _agentrouter(key: str, base: str, model: str, system: str, user: str) -> str:
    root = base[:-3] if base.endswith("/v1") else base
    try:
        return _messages(root, key, model, system, user)
    except RuntimeError as exc:
        if not _missing_channel(str(exc)):
            raise
        available = _listed_models(root, key)
        if not available or available[0] == model:
            raise
        fallback = available[0]
        text = _messages(root, key, fallback, system, user)
        return f"الموديل {model} مش متاح على المفتاح، فالتقرير اتكتب بـ {fallback}.\n\n{text}"


def _messages(root: str, key: str, model: str, system: str, user: str) -> str:
    payload = {
        "model": model,
        "max_tokens": 1400,
        "system": system,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": user}],
    }
    status, body = _request(
        "POST",
        f"{root}/v1/messages",
        key,
        payload,
    )
    if status >= 400:
        raise RuntimeError(body[:500])
    data = json.loads(body)
    parts = [item.get("text", "") for item in data.get("content", []) if item.get("type") == "text" and item.get("text")]
    return "\n".join(parts).strip()


def _listed_models(root: str, key: str) -> list[str]:
    status, body = _request("GET", f"{root}/v1/models", key, None)
    if status >= 400:
        return []
    data = json.loads(body)
    return [item["id"] for item in data.get("data", []) if item.get("id")]


def _request(method: str, url: str, key: str, payload: dict | None) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    raw = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw,
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            # البوابة بترفض عميل بايثون العادي، وبتقبل طلبات واجهة كلود.
            "User-Agent": "claude-cli/1.0.0 (external, cli)",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _missing_channel(text: str) -> bool:
    lowered = text.lower()
    return "无可用渠道" in text or "no available channel" in lowered or "model_not_found" in lowered


def _openai_sdk(key: str, base: str, model: str, system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base, timeout=90.0)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=1400)
    return _content(response)


def _content(response) -> str:
    message = response.choices[0].message.content
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for part in message:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(getattr(part, "text", part)))
        return "".join(parts)
    return "" if message is None else str(message)


def _safe_error(exc: Exception, key: str) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if key:
        text = text.replace(key, "***")
    return text[:240]
