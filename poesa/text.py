"""تنسيق الأرقام والنصوص المشتركة بين اللوحة وسطر الأوامر."""

from __future__ import annotations

from poesa import config
from poesa.backtest import WalkForwardResult
from poesa.engine import ScanResult, StockView
from poesa.journal import ReviewReport, SaveResult
from poesa.learn import AutopsyReport, FrozenRule, LearningStatus, RuleEvalReport


def share(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.{digits}f}%"


def num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}"


def stat_line(block: dict | None, baseline: str) -> str:
    if not block:
        return "—"
    if block["hidden"]:
        return f"عينة غير كافية (n={block['n']})"
    return (
        f"إجمالي {pct(block['gross'])} · صافي {pct(block['net'])} · "
        f"تفوق على {baseline} {share(block['beat'])} · n={block['n']}"
    )


def stock_block(stock: StockView, baseline: str) -> str:
    lines = [f"{stock.symbol}  {stock.name}  {stock.sector}"]
    if not stock.included:
        lines.append("مستبعد: " + "؛ ".join(stock.reasons))
        if stock.price is not None:
            lines.append(f"آخر سعر {price(stock.price)} في {stock.as_of}")
        return "\n".join(lines)
    factor_text = "  ".join(
        f"{config.FACTOR_LABELS[name]} {num(stock.factors.get(name))}" for name in config.WEIGHTS
    )
    lines.append(
        f"الدرجة {num(stock.score)} ({stock.band})  السعر {price(stock.price)}  حتى {stock.as_of}"
    )
    lines.append(factor_text)
    lines.append(
        f"بعد MA50 {pct(stock.dist_sma50)}  بعد MA200 {pct(stock.dist_sma200)}  "
        f"عقوبة الامتداد {num(stock.extension_penalty)} نقطة"
    )
    lines.append(f"تذبذب {pct(stock.ann_vol)}  تراجع {pct(stock.drawdown)}")
    for horizon in config.HORIZONS:
        lines.append(f"{horizon} جلسات: {stat_line(stock.stats.get(horizon), baseline)}")
    return "\n".join(lines)


def forecast_block(rank: int, stock: StockView, baseline: str) -> str:
    return "\n".join(
        [
            f"{rank}. {stock.symbol}  {stock.name}  درجة 20 جلسة {num(stock.score)} ({stock.band})",
            f"بعد MA50 {pct(stock.dist_sma50)}  بعد MA200 {pct(stock.dist_sma200)}  "
            f"عقوبة الامتداد {num(stock.extension_penalty)} نقطة",
            f"20 جلسة: {stat_line(stock.stats.get(config.DECISION_HORIZON), baseline)}",
        ]
    )


def walk_forward_text(result: WalkForwardResult) -> str:
    lines = [
        "ووك-فورورد: ارتباط الدرجة بعائد الـ20 جلسة التالية",
        "العضوية المعلنة في يوم الاختيار، والعوامل ببيانات حتى الإغلاق. الأوزان ما اتغيرتش.",
        *([result.anchor_note] if result.anchor_note else []),
        config.SAMPLE_NOTE,
        config.EXECUTION_NOTE,
    ]
    if result.gap_notes:
        lines.extend(result.gap_notes)
    if result.missing_yahoo:
        lines.append("مفيش تاريخ على ياهو: " + " ".join(result.missing_yahoo))
    lines.append(f"نوافذ بعضوية مجهولة: {result.gaps}")
    if not result.ok and not result.signals:
        lines.append(result.message)
        return "\n".join(lines)
    score = next((item for item in result.signals if item.name == "score"), None)
    if score is not None:
        lines.append("الدرجة: " + _signal_line(score))
    if result.quintiles is not None:
        lines.append(_quintile_line(result.quintiles))
    if result.horizon_ics:
        lines.append("")
        lines.append("Rank IC على آفاق مختلفة")
        lines.append(_horizon_table(result.horizon_ics))
    if result.curves:
        lines.append("")
        lines.append("شرائح العوامل وعائد 20 جلسة الزائد")
        for curve in result.curves:
            lines.append(_curve_line(curve))
    if any(item.name != "score" for item in result.signals):
        lines.append("")
        lines.append("كل عامل لوحده على 20 جلسة")
        for item in result.signals:
            if item.name == "score":
                continue
            lines.append(f"{_signal_label(item.name)}: {_signal_line(item)}")
    lines.append("")
    lines.append("مقارنة أعلى 5 بالباقي، سطر ثانوي")
    for block in result.slices:
        lines.append(_slice_line(block))
    lines.append(f"نوافذ اترفضت لأن العدد أو المسار كان ناقص: {result.skipped}")
    return "\n".join(lines)


def _signal_label(name: str) -> str:
    labels = {
        "score": "الدرجة",
        "dist_sma50": "بعد MA50",
        "dist_sma200": "بعد MA200",
        "extension_penalty": "عقوبة الامتداد",
    }
    return labels.get(name, config.FACTOR_LABELS.get(name, name))


def _signal_line(block) -> str:
    if block.hidden:
        ic = f"عينة غير كافية (n={block.n})"
    else:
        ic = f"Rank IC {block.mean:+.3f} · وسيط {block.median:+.3f} · موجب {share(block.positive)} · n={block.n}"
    if block.spread_hidden:
        spread = f"فرق الشرائح عينة غير كافية (n={block.spread_n})"
    else:
        spread = f"أعلى 20% ناقص أقل 20% {pct(block.spread)} · n={block.spread_n}"
    return f"{ic} · {spread}"


def _quintile_line(block) -> str:
    if block.hidden:
        return f"شرائح الدرجة: عينة غير كافية (n={block.n})"
    labels = ("أعلى 20%", "60–80%", "40–60%", "20–40%", "أقل 20%")
    parts = " · ".join(f"{label} {pct(value)}" for label, value in zip(labels, block.means))
    return f"شرائح الدرجة: {parts} · ترتيب نازل في {block.monotonic} من {block.n}"


def _horizon_table(blocks) -> str:
    names = ["score"]
    for block in blocks:
        for item in block.signals:
            if item.name not in names:
                names.append(item.name)
    header = "الأفق\tn\t" + "\t".join(_signal_label(name) for name in names)
    rows = [header]
    for block in blocks:
        by_name = {item.name: item for item in block.signals}
        cells = []
        for name in names:
            item = by_name.get(name)
            if item is None or item.hidden or item.mean is None:
                cells.append("—")
            else:
                cells.append(f"{item.mean:+.3f}")
        rows.append(f"{block.horizon}\t{block.n}\t" + "\t".join(cells))
    return "\n".join(rows)


def _curve_line(curve) -> str:
    parts = []
    for cell in curve.buckets:
        if cell.hidden:
            parts.append(f"{cell.label} مخفي (n={cell.n})")
        else:
            parts.append(f"{cell.label} {pct(cell.mean)} (n={cell.n})")
    return f"{_signal_label(curve.name)}: " + " · ".join(parts)


def _slice_line(block) -> str:
    title = f"آخر {block.years} سنوات"
    if block.hidden:
        return f"{title}: عينة غير كافية (n={block.n})"
    return (
        f"{title}: أفضل 5 {pct(block.top)} · الباقي {pct(block.rest)} · "
        f"تفوق إجمالي {pct(block.gross)} · صافي {pct(block.net)} · "
        f"النوافذ الإيجابية {share(block.beat)} · وسيط التفوق {pct(block.median)} · n={block.n}"
    )


def journal_save_text(result: SaveResult) -> str:
    if not result.ok:
        return result.message
    if result.created:
        return f"{result.message} الأوزان ثابتة — التقييم بعد 20 جلسة، والتشريح بعد {config.MIN_GRADED_FOR_LEARN} لقطات متقيّمة."
    return result.message


def journal_list_text(rows: list[dict]) -> str:
    if not rows:
        return "مفيش لقطات محفوظة."
    lines = ["دفتر التوقعات"]
    for row in rows:
        names = " ".join(row.get("top") or [])
        lines.append(f"{row['session_date']}  {names}  حفظ {row.get('saved_at') or '—'}")
    return "\n".join(lines)


def journal_review_text(report: ReviewReport) -> str:
    lines = [
        "مراجعة دفتر التوقعات",
        "اللقطة صح لو متوسط عائد أعلى 5 في الـ20 جلسة التالية أكبر من متوسط باقي الأسهم المرتبة وقت الحفظ.",
        config.EXECUTION_NOTE,
    ]
    if not report.ok and not report.items:
        lines.append(report.message)
        return "\n".join(lines)
    lines.append(
        f"اتقيّمت {report.graded} · صح {report.correct} · غلط {report.wrong} · لسه مستنية {report.pending}"
    )
    if report.graded < config.MIN_GRADED_FOR_LEARN:
        lines.append(
            f"للتعلم من الأخطاء لازم على الأقل {config.MIN_GRADED_FOR_LEARN} لقطات متقيّمة "
            f"(باقي {config.MIN_GRADED_FOR_LEARN - report.graded}). الأوزان ثابتة لحد ما نخلص التشريح."
        )
    else:
        lines.append("العيّنة المتقيّمة كافية لتشغيل تشريح الأخطاء (journal autopsy).")
    for item in report.items:
        lines.append("")
        lines.append(f"{item.session_date}  {item.message}")
        for name in item.top:
            actual = "—" if name.actual is None else pct(name.actual)
            lines.append(f"  {name.symbol}  درجة {num(name.score)}  عائد فعلي {actual}")
    return "\n".join(lines)


def learning_status_text(status: LearningStatus) -> str:
    lines = [
        "حالة التعلم من الدفتر",
        "الحفظ يومي · التقييم بعد 20 جلسة · التشريح بعد 8 لقطات متقيّمة · من غير تدوير أوزان.",
        status.message,
        f"لقطات محفوظة {status.snapshots} · متقيّم {status.graded} · صح {status.correct} · غلط {status.wrong} · معلّق {status.pending}",
    ]
    if not status.ready and status.need_more:
        lines.append(f"ناقص {status.need_more} لقطة متقيّمة قبل اعتماد فرضية قاعدة.")
    return "\n".join(lines)


def autopsy_text(report: AutopsyReport) -> str:
    lines = [
        "تشريح أخطاء دفتر التوقعات",
        report.message,
    ]
    if not report.ok and not report.rows:
        return "\n".join(lines)
    lines.append(
        f"متقيّم {report.graded} · صح {report.correct} · غلط {report.wrong} · وسيط التفوق {pct(report.median_excess)}"
    )
    lines.append(
        f"الغلط مع امتداد عالي {report.wrong_high_extension} · من غير امتداد عالي {report.wrong_low_extension}"
    )
    if report.wrong_soft_pullback_mean is not None:
        lines.append(f"متوسط نصيب شريحة السحب (−5%..0) في اللقطات الغلط: {share(report.wrong_soft_pullback_mean)}")
    if report.wrong_extended_mean is not None:
        lines.append(f"متوسط بعد MA50 في اللقطات الغلط: {pct(report.wrong_extended_mean)}")
    for row in report.rows:
        label = "صح" if row.correct else "غلط"
        drag = " · سحب من اسم واحد" if row.single_name_drag else ""
        ext = " · امتداد عالي" if row.high_extension else ""
        lines.append("")
        lines.append(
            f"{row.session_date}  {label}  تفوق {pct(row.excess)}  "
            f"امتداد {num(row.top_mean_extension)}  بعد MA50 {pct(row.top_mean_dist_sma50)}  "
            f"زخم {num(row.top_mean_momentum)}/{num(row.rest_mean_momentum)}  "
            f"قوة نسبية {num(row.top_mean_rs)}/{num(row.rest_mean_rs)}{ext}{drag}"
        )
        if row.max_extension_symbol:
            lines.append(f"  أعلى امتداد: {row.max_extension_symbol} ({num(row.max_extension)} نقطة)")
    if report.suggestion:
        lines.append("")
        lines.append(report.suggestion)
    return "\n".join(lines)


def frozen_rules_text(rules: list[FrozenRule]) -> str:
    if not rules:
        return "مفيش قواعد مجمّدة محفوظة."
    lines = ["القواعد المجمّدة (من غير تدوير WEIGHTS)"]
    for rule in rules:
        lines.append("")
        lines.append(f"{rule.rule_id}  {rule.status}  تجميد {rule.freeze_date}")
        lines.append(rule.rationale)
        lines.append(f"معاملات: {rule.params}")
    return "\n".join(lines)


def rule_eval_text(report: RuleEvalReport) -> str:
    lines = ["تقييم قاعدة مجمّدة خارج العيّنة", report.message]
    if report.rule:
        lines.append(f"القاعدة {report.rule.rule_id} · تجميد {report.rule.freeze_date}")
        lines.append(report.rule.rationale)
    for item in report.items:
        removed = " ".join(item.removed) if item.removed else "—"
        lines.append("")
        lines.append(
            f"{item.session_date}  أساس {item.baseline_correct} ({pct(item.baseline_excess)})  "
            f"فلتر {item.filtered_correct} ({pct(item.filtered_excess)})  شال {removed}"
        )
        if item.message:
            lines.append(f"  {item.message}")
    lines.append("")
    lines.append("ملاحظة: التقييم بيقارن الفلتر على اللقطات بعد تاريخ التجميد فقط. WEIGHTS ثابتة.")
    return "\n".join(lines)


def scan_header(scan: ScanResult) -> str:
    basket = pct(scan.basket_return_20d)
    lines = [
        "محلل البورصة المصرية",
        scan.baseline_note,
        config.SURVIVORSHIP_NOTE,
        config.COST_NOTE,
        config.SAMPLE_NOTE,
        config.ADVICE_NOTE,
        f"آخر جلسة في التقويم: {scan.calendar_end or '—'}  عائد السلة 20 جلسة: {basket}",
        f"وقت الحساب: {scan.generated_at}",
    ]
    if not scan.ok:
        lines.append(scan.message)
    return "\n".join(lines)
