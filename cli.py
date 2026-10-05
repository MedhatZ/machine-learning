"""أوامر: scan و analyze و compare و journal و backtest."""

from __future__ import annotations

import argparse
import sys

from poesa import config
from poesa.ai import write_report
from poesa.backtest import run_walk_forward
from poesa.engine import run_scan
from poesa.journal import list_snapshots, review_snapshots, save_snapshot
from poesa.learn import evaluate_rule, learning_status, load_rules, propose_rule, run_autopsy
from poesa.text import (
    autopsy_text,
    forecast_block,
    frozen_rules_text,
    journal_list_text,
    journal_review_text,
    journal_save_text,
    learning_status_text,
    rule_eval_text,
    scan_header,
    stock_block,
    walk_forward_text,
)
from poesa.universe import find


def main(argv: list[str] | None = None) -> int:
    _utf8()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--refresh", action="store_true", help="تجاهل الكاش ونزّل البيانات من جديد")
    common.add_argument("--report", action="store_true", help="اطلب تقريراً عربياً من النموذج")
    parser = argparse.ArgumentParser(description="محلل البورصة المصرية")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_parser = sub.add_parser("scan", parents=[common], help="رتّب الأسهم اللي عدّت الفحص")
    scan_parser.add_argument("--save", action="store_true", help="احفظ لقطة أعلى 5 لنفس جلسة التقويم")
    scan_parser.add_argument("--weekly", action="store_true", help="مع --save: متحفظش لو آخر لقطة أقرب من 5 جلسات")
    sub.add_parser("backtest", parents=[common], help="ووك-فورورد: Rank IC والشرائح التاريخية")
    analyze = sub.add_parser("analyze", parents=[common], help="تفاصيل سهم واحد")
    analyze.add_argument("symbol")
    compare = sub.add_parser("compare", parents=[common], help="قارن من سهمين لأربعة")
    compare.add_argument("symbols", nargs="+")
    journal = sub.add_parser("journal", help="دفتر التوقعات: حفظ ومراجعة وتعلم")
    journal_sub = journal.add_subparsers(dest="journal_command", required=True)
    save = journal_sub.add_parser("save", parents=[common], help="احفظ لقطة اليوم")
    save.add_argument("--force", action="store_true", help="استبدل لقطة نفس الجلسة لو موجودة")
    save.add_argument("--weekly", action="store_true", help="متحفظش لو آخر لقطة أقرب من 5 جلسات")
    journal_sub.add_parser("list", help="اعرض اللقطات المحفوظة")
    review = journal_sub.add_parser("review", parents=[common], help="راجع اللقطات اللي عدّى عليها 20 جلسة")
    review.add_argument("--month", help="شهر اللقطة بصيغة YYYY-MM")
    journal_sub.add_parser("status", parents=[common], help="هل العيّنة المتقيّمة كافية للتعلم؟")
    journal_sub.add_parser("autopsy", parents=[common], help="تشريح اللقطات الغلط بعد التقييم")
    rules = journal_sub.add_parser("rules", help="قواعد مجمّدة من غير تدوير أوزان")
    rules_sub = rules.add_subparsers(dest="rules_command", required=True)
    rules_sub.add_parser("list", help="اعرض القواعد المجمّدة")
    propose = rules_sub.add_parser("propose", help="سجّل فرضية فلتر MA50 مجمّدة")
    propose.add_argument("--freeze-date", help="YYYY-MM-DD؛ التقييم على اللقطات بعده فقط")
    propose.add_argument("--cap", type=float, default=config.MA50_EXTENDED_CAP, help="سقف بعد MA50")
    evaluate = rules_sub.add_parser("eval", parents=[common], help="قيّم قاعدة خارج العيّنة")
    evaluate.add_argument("rule_id", help="رمز القاعدة")

    args = parser.parse_args(argv)
    if args.command == "backtest":
        result = run_walk_forward(refresh=args.refresh, progress=_progress)
        print(walk_forward_text(result))
        return 0 if result.ok else 1
    if args.command == "journal":
        return _journal(args)

    scan = run_scan(refresh=args.refresh, progress=_progress)
    print(scan_header(scan))
    print()

    if args.command == "scan":
        if not scan.ranked:
            print(scan.message)
            return 1
        print("أعلى 5 Setup حاليًا")
        for rank, stock in enumerate(scan.ranked[: config.TOP_N], start=1):
            print(forecast_block(rank, stock, scan.baseline_label))
            print()
        print("الترتيب الكامل")
        for stock in scan.ranked:
            print(stock_block(stock, scan.baseline_label))
            print()
        if scan.excluded:
            print(f"مستبعدون ({len(scan.excluded)})")
            for stock in scan.excluded:
                print(f"- {stock.symbol} {stock.name}: {'؛ '.join(stock.reasons)}")
        if args.save:
            print()
            print(journal_save_text(save_snapshot(scan, weekly=args.weekly)))
        focus = [stock.symbol for stock in scan.ranked[:5]]
    elif args.command == "analyze":
        listing = find(args.symbol)
        if listing is None:
            print(f"{args.symbol} مش في القائمة الثابتة.")
            return 1
        stock = next(item for item in scan.stocks if item.symbol == listing.symbol)
        print(stock_block(stock, scan.baseline_label))
        focus = [stock.symbol]
    else:
        if len(args.symbols) < 2 or len(args.symbols) > 4:
            print("المقارنة من سهمين لأربعة.")
            return 1
        focus = []
        for token in args.symbols:
            listing = find(token)
            if listing is None:
                print(f"{token} مش في القائمة الثابتة.")
                return 1
            focus.append(listing.symbol)
        for symbol in focus:
            stock = next(item for item in scan.stocks if item.symbol == symbol)
            print(stock_block(stock, scan.baseline_label))
            print()

    if args.report:
        print()
        print(write_report(scan, focus))
    return 0 if scan.ok or args.command != "scan" else 1


def _journal(args) -> int:
    if args.journal_command == "list":
        print(journal_list_text(list_snapshots()))
        return 0
    if args.journal_command == "save":
        scan = run_scan(refresh=args.refresh, progress=_progress)
        print(scan_header(scan))
        print()
        result = save_snapshot(scan, force=args.force, weekly=args.weekly)
        print(journal_save_text(result))
        return 0 if result.ok else 1
    if args.journal_command == "status":
        print(learning_status_text(learning_status(refresh=args.refresh, progress=_progress)))
        return 0
    if args.journal_command == "autopsy":
        print(autopsy_text(run_autopsy(refresh=args.refresh, progress=_progress)))
        return 0
    if args.journal_command == "rules":
        return _rules(args)
    report = review_snapshots(refresh=args.refresh, progress=_progress, month=args.month)
    print(journal_review_text(report))
    return 0 if report.ok else 1


def _rules(args) -> int:
    if args.rules_command == "list":
        print(frozen_rules_text(load_rules()))
        return 0
    if args.rules_command == "propose":
        rule = propose_rule(
            freeze_date=args.freeze_date,
            params={"max_dist_sma50": args.cap, "top_n": config.TOP_N},
        )
        print(frozen_rules_text([rule]))
        print()
        print("القاعدة اتسجّلت مجمّدة. WEIGHTS لم تتغير. قيّمها لاحقًا بـ journal rules eval.")
        return 0
    result = evaluate_rule(args.rule_id, refresh=args.refresh, progress=_progress)
    print(rule_eval_text(result))
    return 0 if result.ok else 1


def _progress(symbol: str, cached: bool) -> None:
    state = "كاش" if cached else "تنزيل"
    print(f"{state}: {symbol}", file=sys.stderr)


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
