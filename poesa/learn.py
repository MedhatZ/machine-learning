"""تعلم من دفتر التوقعات: تشريح أخطاء + قواعد مجمّدة من غير تدوير الأوزان."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

from poesa import config
from poesa.journal import (
    ReviewItem,
    ReviewReport,
    _grade,
    _read,
    list_snapshots,
    load_return_panel,
    review_snapshots,
)


LEARN_DIR = config.LEARN_DIR


@dataclass
class LearningStatus:
    ok: bool
    message: str
    snapshots: int
    pending: int
    graded: int
    correct: int
    wrong: int
    ready: bool
    need_more: int


@dataclass
class SnapshotAutopsy:
    session_date: str
    correct: bool
    excess: float
    top_mean_extension: float | None
    top_mean_dist_sma50: float | None
    soft_pullback_share: float | None
    top_mean_momentum: float | None
    top_mean_rs: float | None
    rest_mean_momentum: float | None
    rest_mean_rs: float | None
    max_extension_symbol: str | None
    max_extension: float | None
    single_name_drag: bool
    high_extension: bool


@dataclass
class AutopsyReport:
    ok: bool
    message: str
    ready: bool
    graded: int
    correct: int
    wrong: int
    median_excess: float | None
    wrong_high_extension: int
    wrong_low_extension: int
    wrong_soft_pullback_mean: float | None
    wrong_extended_mean: float | None
    rows: list[SnapshotAutopsy] = field(default_factory=list)
    suggestion: str | None = None


@dataclass
class FrozenRule:
    rule_id: str
    created_at: str
    freeze_date: str
    kind: str
    params: dict
    rationale: str
    status: str


@dataclass
class RuleEvalItem:
    session_date: str
    baseline_correct: bool | None
    filtered_correct: bool | None
    baseline_excess: float | None
    filtered_excess: float | None
    removed: list[str]
    message: str


@dataclass
class RuleEvalReport:
    ok: bool
    message: str
    rule: FrozenRule | None
    oos_n: int
    baseline_correct: int
    filtered_correct: int
    items: list[RuleEvalItem] = field(default_factory=list)


def learning_status(
    directory: Path | None = None,
    refresh: bool = False,
    progress=None,
    report: ReviewReport | None = None,
) -> LearningStatus:
    rows = list_snapshots(directory)
    if not rows:
        return LearningStatus(
            False,
            "مفيش لقطات محفوظة. احفظ لقطة كل جلسة جديدة.",
            0,
            0,
            0,
            0,
            0,
            False,
            config.MIN_GRADED_FOR_LEARN,
        )
    reviewed = report or review_snapshots(directory=directory, refresh=refresh, progress=progress)
    need = max(config.MIN_GRADED_FOR_LEARN - reviewed.graded, 0)
    ready = reviewed.graded >= config.MIN_GRADED_FOR_LEARN
    if ready:
        message = f"جاهز للتشريح: {reviewed.graded} لقطة متقيّمة (صح {reviewed.correct} · غلط {reviewed.wrong})."
    else:
        message = (
            f"لسه مستنيين التقييم: متقيّم {reviewed.graded}/{config.MIN_GRADED_FOR_LEARN} · "
            f"معلّق {reviewed.pending}. كمّل الحفظ اليومي من غير تغيير أوزان."
        )
    return LearningStatus(
        True,
        message,
        len(rows),
        reviewed.pending,
        reviewed.graded,
        reviewed.correct,
        reviewed.wrong,
        ready,
        need,
    )


def run_autopsy(
    directory: Path | None = None,
    refresh: bool = False,
    progress=None,
    min_graded: int = config.MIN_GRADED_FOR_LEARN,
    report: ReviewReport | None = None,
) -> AutopsyReport:
    folder = directory or config.JOURNAL_DIR
    reviewed = report or review_snapshots(directory=folder, refresh=refresh, progress=progress)
    graded = [item for item in reviewed.items if item.status == "graded" and item.correct is not None]
    if not graded:
        return AutopsyReport(
            False,
            "مفيش لقطات متقيّمة لسه. استنى 20 جلسة بعد كل حفظ.",
            False,
            0,
            0,
            0,
            None,
            0,
            0,
            None,
            None,
        )
    payloads = {_payload_date(path): _read(Path(path)) for path in _paths(folder)}
    rows: list[SnapshotAutopsy] = []
    for item in graded:
        payload = payloads.get(item.session_date)
        if not payload:
            continue
        rows.append(_autopsy_row(item, payload))
    if not rows:
        return AutopsyReport(False, "اللقطات المتقيّمة من غير تفاصيل عوامل.", False, 0, 0, 0, None, 0, 0, None, None)
    wrong = [row for row in rows if not row.correct]
    wrong_high = sum(1 for row in wrong if row.high_extension)
    wrong_low = len(wrong) - wrong_high
    soft = [row.soft_pullback_share for row in wrong if row.soft_pullback_share is not None]
    extended = [row.top_mean_dist_sma50 for row in wrong if row.top_mean_dist_sma50 is not None]
    ready = len(rows) >= min_graded
    if ready and wrong and wrong_high >= max(2, (len(wrong) + 1) // 2):
        suggestion = (
            f"فرضية مجمّدة مقترحة: فلتر بعد MA50 فوق {config.MA50_EXTENDED_CAP:.0%} "
            f"(استبعاد من أعلى {config.TOP_N} وإحلال التالي في الترتيب). "
            "تتقاس على لقطات بعد تاريخ التجميد فقط، من غير تدوير WEIGHTS."
        )
    elif ready and not wrong:
        suggestion = "مفيش لقطات غلط في العيّنة الحالية. كمّل التخزين ومش تلمس المعادلة."
    elif ready:
        suggestion = "مفيش نمط امتداد واضح. كمّل التخزين شهر كمان ومش تغيّر الأوزان."
    else:
        suggestion = (
            f"لسه {min_graded - len(rows)} لقطة متقيّمة ناقصة قبل اعتماد فرضية. "
            "الحفظ اليومي يستمر والأوزان ثابتة."
        )
    excesses = [row.excess for row in rows]
    return AutopsyReport(
        True,
        "تشريح الأخطاء جاهز." if ready else "تشريح أولي — العيّنة لسه صغيرة.",
        ready,
        len(rows),
        sum(1 for row in rows if row.correct),
        len(wrong),
        float(median(excesses)) if excesses else None,
        wrong_high,
        wrong_low,
        float(sum(soft) / len(soft)) if soft else None,
        float(sum(extended) / len(extended)) if extended else None,
        rows,
        suggestion,
    )


def propose_rule(
    kind: str = "ma50_cap",
    freeze_date: str | None = None,
    rationale: str | None = None,
    params: dict | None = None,
    directory: Path | None = None,
) -> FrozenRule:
    if kind != "ma50_cap":
        raise ValueError(f"نوع قاعدة غير معروف: {kind}")
    folder = directory or LEARN_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    freeze = freeze_date or datetime.now().strftime("%Y-%m-%d")
    rule = FrozenRule(
        rule_id=f"ma50_cap_{freeze.replace('-', '')}",
        created_at=stamp,
        freeze_date=freeze,
        kind="ma50_cap",
        params=params or {"max_dist_sma50": config.MA50_EXTENDED_CAP, "top_n": config.TOP_N},
        rationale=rationale
        or (
            f"استبعاد الأسهم اللي بعد MA50 فوق {config.MA50_EXTENDED_CAP:.0%} من أعلى الترتيب، "
            "وقياس الأثر على لقطات بعد تاريخ التجميد فقط."
        ),
        status="proposed",
    )
    rules = [item for item in load_rules(folder) if item.rule_id != rule.rule_id]
    rules.append(rule)
    _write_rules(rules, folder)
    return rule


def load_rules(directory: Path | None = None) -> list[FrozenRule]:
    path = (directory or LEARN_DIR) / "rules.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload if isinstance(payload, list) else payload.get("rules", [])
    return [
        FrozenRule(
            rule_id=str(row.get("rule_id")),
            created_at=str(row.get("created_at", "")),
            freeze_date=str(row.get("freeze_date", "")),
            kind=str(row.get("kind", "")),
            params=dict(row.get("params") or {}),
            rationale=str(row.get("rationale", "")),
            status=str(row.get("status", "proposed")),
        )
        for row in rows
    ]


def evaluate_rule(
    rule_id: str,
    journal_dir: Path | None = None,
    learn_dir: Path | None = None,
    refresh: bool = False,
    progress=None,
    horizon: int = config.DECISION_HORIZON,
    returns=None,
    calendar=None,
) -> RuleEvalReport:
    rules = {rule.rule_id: rule for rule in load_rules(learn_dir)}
    rule = rules.get(rule_id)
    if rule is None:
        return RuleEvalReport(False, f"مفيش قاعدة بالرمز {rule_id}.", None, 0, 0, 0)
    if rule.kind != "ma50_cap":
        return RuleEvalReport(False, f"تقييم النوع {rule.kind} مش متاح.", rule, 0, 0, 0)

    folder = journal_dir or config.JOURNAL_DIR
    if returns is None or calendar is None:
        returns, calendar = load_return_panel(refresh=refresh, progress=progress)
    if returns is None or calendar is None:
        return RuleEvalReport(False, "مفيش تقويم أسعار نراجع عليه.", rule, 0, 0, 0)

    cap = float(rule.params.get("max_dist_sma50", config.MA50_EXTENDED_CAP))
    top_n = int(rule.params.get("top_n", config.TOP_N))
    items: list[RuleEvalItem] = []
    for row in list_snapshots(folder):
        payload = _read(Path(row["path"]))
        if payload is None:
            continue
        session = str(payload.get("session_date", ""))
        if session <= rule.freeze_date:
            continue
        baseline = _grade(payload, returns, calendar, horizon)
        if baseline.status != "graded":
            continue
        filtered = _apply_ma50_cap(payload, cap=cap, top_n=top_n)
        after = _grade(
            payload,
            returns,
            calendar,
            horizon,
            top_symbols=filtered["top_symbols"],
            rest_symbols=filtered["rest_symbols"],
        )
        items.append(
            RuleEvalItem(
                session_date=session,
                baseline_correct=baseline.correct,
                filtered_correct=after.correct if after.status == "graded" else None,
                baseline_excess=baseline.excess,
                filtered_excess=after.excess if after.status == "graded" else None,
                removed=filtered["removed"],
                message=after.message,
            )
        )
    if not items:
        return RuleEvalReport(
            True,
            f"مفيش لقطات متقيّمة بعد تاريخ التجميد {rule.freeze_date}. استنى عيّنة خارج العينة.",
            rule,
            0,
            0,
            0,
        )
    baseline_correct = sum(1 for row in items if row.baseline_correct)
    filtered_correct = sum(1 for row in items if row.filtered_correct)
    return RuleEvalReport(
        True,
        (
            f"تقييم خارج العيّنة للقاعدة {rule.rule_id}: "
            f"الأساس صح {baseline_correct}/{len(items)} · بعد الفلتر صح {filtered_correct}/{len(items)}. "
            "الأوزان لم تتغير."
        ),
        rule,
        len(items),
        baseline_correct,
        filtered_correct,
        items,
    )


def _write_rules(rules: list[FrozenRule], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "rule_id": rule.rule_id,
            "created_at": rule.created_at,
            "freeze_date": rule.freeze_date,
            "kind": rule.kind,
            "params": rule.params,
            "rationale": rule.rationale,
            "status": rule.status,
        }
        for rule in rules
    ]
    (directory / "rules.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _paths(folder: Path) -> list[str]:
    return [row["path"] for row in list_snapshots(folder)]


def _payload_date(path: str) -> str:
    payload = _read(Path(path))
    if payload and payload.get("session_date"):
        return str(payload["session_date"])
    return Path(path).stem


def _autopsy_row(item: ReviewItem, payload: dict) -> SnapshotAutopsy:
    top = list(payload.get("top") or [])
    ranked = list(payload.get("ranked") or top)
    top_symbols = {row.get("symbol") for row in top}
    rest = [row for row in ranked if row.get("symbol") not in top_symbols]
    extensions = [_num(row.get("extension_penalty")) for row in top]
    dist50 = [_num(row.get("dist_sma50")) for row in top]
    moms = [_factor(row, "momentum") for row in top]
    rss = [_factor(row, "relative_strength") for row in top]
    rest_moms = [_factor(row, "momentum") for row in rest]
    rest_rss = [_factor(row, "relative_strength") for row in rest]
    soft = [
        1.0
        for value in dist50
        if value is not None and config.MA50_SOFT_PULLBACK[0] <= value < config.MA50_SOFT_PULLBACK[1]
    ]
    max_ext = None
    max_sym = None
    for row, value in zip(top, extensions):
        if value is None:
            continue
        if max_ext is None or value > max_ext:
            max_ext = value
            max_sym = row.get("symbol")
    actuals = [outcome.actual for outcome in item.top if outcome.actual is not None]
    single_drag = False
    if len(actuals) >= 3:
        ordered = sorted(actuals)
        single_drag = ordered[0] < (sum(ordered[1:]) / len(ordered[1:])) - 0.05
    mean_ext = _mean(extensions)
    high = bool(mean_ext is not None and mean_ext >= config.HIGH_EXTENSION_PENALTY) or bool(
        max_ext is not None and max_ext >= config.HIGH_EXTENSION_PENALTY * 2
    )
    return SnapshotAutopsy(
        session_date=item.session_date,
        correct=bool(item.correct),
        excess=float(item.excess or 0.0),
        top_mean_extension=mean_ext,
        top_mean_dist_sma50=_mean(dist50),
        soft_pullback_share=(sum(soft) / len(top)) if top else None,
        top_mean_momentum=_mean(moms),
        top_mean_rs=_mean(rss),
        rest_mean_momentum=_mean(rest_moms),
        rest_mean_rs=_mean(rest_rss),
        max_extension_symbol=max_sym,
        max_extension=max_ext,
        single_name_drag=single_drag,
        high_extension=high,
    )


def _apply_ma50_cap(payload: dict, cap: float, top_n: int) -> dict:
    ranked = list(payload.get("ranked") or payload.get("top") or [])
    kept = []
    removed = []
    for row in ranked:
        dist = _num(row.get("dist_sma50"))
        if dist is not None and dist > cap:
            removed.append(str(row.get("symbol")))
            continue
        kept.append(row)
    top = kept[:top_n]
    top_symbols = [str(row.get("symbol")) for row in top if row.get("symbol")]
    rest_symbols = [str(row.get("symbol")) for row in kept[top_n:] if row.get("symbol")]
    rest_symbols.extend(removed)
    return {"top_symbols": top_symbols, "rest_symbols": rest_symbols, "removed": removed}


def _factor(row: dict, name: str) -> float | None:
    factors = row.get("factors") or {}
    return _num(factors.get(name))


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))
