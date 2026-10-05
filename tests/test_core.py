"""اختبارات الحسابات من غير شبكة."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from poesa import config
from poesa.backtest import _bucket_index, decision_origins, eligible_mask, evaluate_walk_forward
from poesa.membership import Review, membership_intervals, membership_on
from poesa.benchmark import leave_one_out, select_benchmarks
from poesa.data import naive_dates
from poesa.engine import ScanResult, StockView
from poesa.journal import NameOutcome, ReviewItem, ReviewReport, _grade, list_snapshots, save_snapshot
from poesa.learn import (
    _apply_ma50_cap,
    evaluate_rule,
    learning_status,
    load_rules,
    propose_rule,
    run_autopsy,
)
from poesa.quality import assess, assess_calendar
from poesa.scoring import score_panel
from poesa.stats import excess_table, sample_positions


class CoreTests(unittest.TestCase):
    def test_weights_are_precommitted(self):
        self.assertEqual(
            config.WEIGHTS,
            {
                "momentum": 0.30,
                "relative_strength": 0.25,
                "trend": 0.20,
                "volume": 0.15,
                "risk": 0.10,
            },
        )
        self.assertAlmostEqual(sum(config.WEIGHTS.values()), 1.0)
        self.assertEqual(config.MIN_N, 8)
        self.assertEqual(config.MIN_GRADED_FOR_LEARN, 8)
        self.assertAlmostEqual(config.MA50_EXTENDED_CAP, 0.05)
        self.assertAlmostEqual(config.ROUND_TRIP_COST, 0.008)

    def test_sample_positions_do_not_overlap(self):
        positions = sample_positions(200, 0, 20)
        self.assertEqual(positions, list(range(0, 180, 20)))
        for left, right in zip(positions, positions[1:]):
            self.assertLessEqual(left + 20, right)

    def test_excess_is_zero_when_stock_matches_baseline(self):
        index = pd.bdate_range("2020-01-01", periods=250)
        daily = pd.Series(0.002, index=index)
        score = pd.Series(70.0, index=index)
        table = excess_table(score, daily, daily, 70.0)
        self.assertFalse(table[5]["hidden"])
        self.assertGreaterEqual(table[5]["n"], 8)
        self.assertAlmostEqual(table[5]["gross"], 0.0, places=9)
        self.assertAlmostEqual(table[5]["net"], -config.ROUND_TRIP_COST, places=9)

    def test_other_bands_are_excluded_and_small_samples_hide_the_mean(self):
        index = pd.bdate_range("2020-01-01", periods=200)
        score = pd.Series(85.0, index=index)
        positions = sample_positions(len(index), 0, 20)
        score.iloc[positions[:4]] = 50.0
        daily = pd.Series(0.01, index=index)
        bench = pd.Series(0.0, index=index)
        table = excess_table(score, daily, bench, 85.0)
        self.assertEqual(table[20]["n"], len(positions) - 4)
        self.assertTrue(table[20]["hidden"])
        self.assertIsNone(table[20]["gross"])
        self.assertIsNone(table[20]["net"])

    def test_leave_one_out_excludes_the_stock_itself(self):
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02],
                "B": [0.03, np.nan],
                "C": [0.05, 0.07],
            }
        )
        result = leave_one_out(returns, min_peers=1)
        self.assertAlmostEqual(result.iloc[0]["A"], 0.04)
        self.assertAlmostEqual(result.iloc[1]["A"], 0.07)
        self.assertTrue(pd.isna(result.iloc[1]["B"]))

    def test_official_index_is_used_only_when_history_is_long_enough(self):
        index = pd.Series(np.linspace(0.0, 0.01, config.MIN_CALENDAR_BARS))
        stocks = pd.DataFrame({"A": index, "B": index * 0.5})
        kind, bench = select_benchmarks(index, stocks)
        self.assertEqual(kind, "official")
        self.assertTrue(bench["A"].equals(index))
        short = index.iloc[:10]
        kind, bench = select_benchmarks(short, stocks.iloc[:10], min_peers=1)
        self.assertEqual(kind, "basket")
        self.assertAlmostEqual(bench.iloc[5]["A"], stocks.iloc[5]["B"])

    def test_extension_above_ma200_lowers_the_score(self):
        near = _path(early_return=0.0004, lift=1.0)
        extended = _path(early_return=0.0, lift=1.7)
        bench = pd.Series(0.0, index=near.index)
        near_panel = score_panel(near, near * 1.01, near * 0.99, pd.Series(1_000_000.0, index=near.index), bench)
        extended_bench = pd.Series(0.0, index=extended.index)
        extended_panel = score_panel(
            extended,
            extended * 1.01,
            extended * 0.99,
            pd.Series(1_000_000.0, index=extended.index),
            extended_bench,
        )
        self.assertLess(near_panel["dist_sma200"].iloc[-1], 0.15)
        self.assertGreater(extended_panel["dist_sma200"].iloc[-1], 0.45)
        self.assertGreater(extended_panel["extension_penalty"].iloc[-1], near_panel["extension_penalty"].iloc[-1])
        self.assertLess(extended_panel["score"].iloc[-1], near_panel["score"].iloc[-1] - 10)

    def test_walk_forward_pays_the_winners_and_skips_the_gap(self):
        index = pd.bdate_range("2022-01-03", periods=900)
        names = list("ABCDEFGH")
        origins = decision_origins(index, 20, 3)
        self.assertGreaterEqual(len(origins), 8)
        self.assertEqual(origins[1] - origins[0], 20)
        scores = pd.DataFrame(0.0, index=index, columns=names)
        eligible = pd.DataFrame(False, index=index, columns=names)
        returns = pd.DataFrame(0.0, index=index, columns=names)
        for origin in origins[:8]:
            eligible.iloc[origin] = True
            scores.iloc[origin, :5] = 90
            scores.iloc[origin, 5:] = 10
            returns.iloc[origin + 1 : origin + 21, :5] = 0.01
        result = evaluate_walk_forward(scores, returns, eligible, years=(3,), min_n=8)
        self.assertTrue(result.ok)
        self.assertEqual(result.slices[0].n, 8)
        self.assertFalse(result.slices[0].hidden)
        self.assertGreater(result.slices[0].gross, 0.05)
        self.assertAlmostEqual(result.slices[0].net, result.slices[0].gross - config.ROUND_TRIP_COST)
        self.assertEqual(result.windows[0].top, list("ABCDE"))
        self.assertNotIn(str(index[origins[0] + 5].date()), [window.date for window in result.windows])
        leaked = returns.copy()
        leaked.iloc[origins[0] + 21 :] = 0.5
        again = evaluate_walk_forward(scores, leaked, eligible, years=(3,), min_n=8)
        self.assertEqual(again.windows[0].top, result.windows[0].top)
        self.assertAlmostEqual(again.windows[0].excess, result.windows[0].excess)

    def test_walk_forward_hides_a_small_sample(self):
        index = pd.bdate_range("2024-01-02", periods=900)
        names = list("ABCDEFGH")
        origins = decision_origins(index, 20, 3)
        scores = pd.DataFrame(1.0, index=index, columns=names)
        eligible = pd.DataFrame(False, index=index, columns=names)
        eligible.iloc[origins[0]] = True
        returns = pd.DataFrame(0.0, index=index, columns=names)
        result = evaluate_walk_forward(scores, returns, eligible, years=(3,))
        self.assertEqual(result.slices[0].n, 1)
        self.assertTrue(result.slices[0].hidden)
        self.assertIsNone(result.slices[0].gross)

    def test_a_future_jump_does_not_disqualify_the_present(self):
        index = pd.bdate_range("2022-01-03", periods=400)
        price = pd.Series(100.0, index=index)
        price.iloc[350:] = 200.0
        volume = pd.Series(100_000.0, index=index)
        mask = eligible_mask(price, price, price * 1.01, price * 0.99, volume)
        self.assertTrue(bool(mask.iloc[340]))
        self.assertFalse(bool(mask.iloc[350]))

    def test_announced_membership_keeps_thirty_names_and_the_swdy_exit(self):
        intervals = membership_intervals()
        self.assertTrue(all(item.symbols is not None and len(item.symbols) == 30 for item in intervals))
        self.assertNotIn("SWDY", membership_on(pd.Timestamp("2025-02-02"), intervals))
        self.assertIn("SWDY", membership_on(pd.Timestamp("2025-01-30"), intervals))
        self.assertNotIn("MCQE", membership_on(pd.Timestamp("2025-02-02"), intervals))
        self.assertIn("MCQE", membership_on(pd.Timestamp("2025-08-03"), intervals))

    def test_a_name_returns_only_before_its_announced_deletion(self):
        reviews = (
            Review("2024-02-01", ("A",), ("B",), "src"),
            Review("2024-08-01", ("C",), ("A",), "src"),
        )
        intervals = membership_intervals("2024-08-01", ("C", "D"), reviews)
        self.assertEqual(membership_on(pd.Timestamp("2024-08-01"), intervals), frozenset({"C", "D"}))
        self.assertEqual(membership_on(pd.Timestamp("2024-02-01"), intervals), frozenset({"A", "D"}))
        earlier = membership_on(pd.Timestamp("2024-01-15"), intervals)
        self.assertIn("B", earlier)
        self.assertNotIn("A", earlier)

    def test_incomplete_review_hides_the_following_period(self):
        reviews = (
            Review("2024-02-01", ("A",), ("B",), "src", complete=False),
            Review("2024-08-01", ("C",), ("A",), "src"),
        )
        intervals = membership_intervals("2024-08-01", ("C", "D"), reviews)
        self.assertEqual(membership_on(pd.Timestamp("2024-08-01"), intervals), frozenset({"C", "D"}))
        self.assertIsNone(membership_on(pd.Timestamp("2024-02-01"), intervals))
        self.assertIsNone(membership_on(pd.Timestamp("2024-01-15"), intervals))

    def test_rank_ic_follows_the_forward_return_and_ignores_later_prices(self):
        index, names, origins = _ic_market()
        scores, eligible, returns = _matched_signal(index, names, origins[:8], sign=1)
        known = pd.Series(True, index=index)
        known.iloc[origins[8]] = False
        result = evaluate_walk_forward(scores, returns, eligible, known=known, years=(3,), min_n=8)
        score = result.signals[0]
        self.assertEqual(result.gaps, 1)
        self.assertEqual(score.n, 8)
        self.assertGreater(score.mean, 0.95)
        self.assertFalse(result.quintiles.hidden)
        self.assertEqual(result.quintiles.monotonic, result.quintiles.n)
        self.assertGreater(result.quintiles.means[0], result.quintiles.means[-1])
        leaked = returns.copy()
        leaked.iloc[origins[7] + 21 :] = 0.5
        again = evaluate_walk_forward(scores, leaked, eligible, known=known, years=(3,), min_n=8)
        self.assertAlmostEqual(again.signals[0].mean, score.mean, places=8)

    def test_fixed_distance_buckets_follow_the_pullback_shape(self):
        index = pd.bdate_range("2022-01-03", periods=900)
        names = [f"N{i:02d}" for i in range(10)]
        origins = decision_origins(index, 20, 3)[:8]
        scores = pd.DataFrame(50.0, index=index, columns=names)
        eligible = pd.DataFrame(False, index=index, columns=names)
        returns = pd.DataFrame(0.0, index=index, columns=names)
        dist = pd.DataFrame(0.0, index=index, columns=names)
        cuts = (-0.05, 0.0, 0.05, 0.15)
        centers = (-0.08, -0.02, 0.02, 0.08, 0.20)
        payoffs = (0.01, 0.03, 0.02, 0.0, -0.02)
        for origin in origins:
            eligible.iloc[origin] = True
            for position, name in enumerate(names):
                bucket = position % 5
                dist.iloc[origin, position] = centers[bucket]
                returns.iloc[origin + 1 : origin + 21, position] = payoffs[bucket]
        result = evaluate_walk_forward(
            scores,
            returns,
            eligible,
            factors={"dist_sma50": dist},
            years=(3,),
            min_n=8,
            ic_horizons=(20,),
            buckets={"dist_sma50": cuts},
        )
        curve = next(item for item in result.curves if item.name == "dist_sma50")
        means = [cell.mean for cell in curve.buckets]
        self.assertEqual([cell.n for cell in curve.buckets], [16, 16, 16, 16, 16])
        self.assertGreater(means[1], means[0])
        self.assertGreater(means[1], means[4])
        self.assertEqual(_bucket_index("extension_penalty", 0.0, (0.0, 5.0, 15.0, 25.0)), 0)
        self.assertEqual(_bucket_index("extension_penalty", 25.0, (0.0, 5.0, 15.0, 25.0)), 4)

    def test_rank_ic_is_computed_on_each_requested_horizon(self):
        index = pd.bdate_range("2022-01-03", periods=1100)
        names = [f"N{i:02d}" for i in range(15)]
        scores = pd.DataFrame(0.0, index=index, columns=names)
        eligible = pd.DataFrame(False, index=index, columns=names)
        returns = pd.DataFrame(0.0, index=index, columns=names)
        for horizon in (5, 20, 60):
            for origin in decision_origins(index, horizon, 3)[:12]:
                _fill_signal(scores, eligible, returns, names, origin, 1, horizon)
        result = evaluate_walk_forward(
            scores,
            returns,
            eligible,
            years=(3,),
            min_n=8,
            ic_horizons=(5, 20, 60),
        )
        horizons = [block.horizon for block in result.horizon_ics]
        self.assertEqual(horizons, [5, 20, 60])
        for block in result.horizon_ics:
            score = next(item for item in block.signals if item.name == "score")
            self.assertGreaterEqual(score.n, 8)
            self.assertFalse(score.hidden)
            self.assertGreater(score.mean, 0.9)

    def test_rank_ic_cancels_when_the_sign_alternates(self):
        index, names, origins = _ic_market()
        scores = pd.DataFrame(0.0, index=index, columns=names)
        eligible = pd.DataFrame(False, index=index, columns=names)
        returns = pd.DataFrame(0.0, index=index, columns=names)
        for step, origin in enumerate(origins[:8]):
            sign = 1 if step % 2 == 0 else -1
            _fill_signal(scores, eligible, returns, names, origin, sign)
        result = evaluate_walk_forward(scores, returns, eligible, years=(3,), min_n=8)
        score = result.signals[0]
        self.assertAlmostEqual(score.mean, 0.0, places=2)
        self.assertAlmostEqual(score.positive, 0.5)

    def test_journal_saves_once_per_session_and_grades_after_horizon(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            scan = _fake_scan("2024-01-02")
            first = save_snapshot(scan, directory=folder)
            second = save_snapshot(scan, directory=folder)
            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(len(list_snapshots(folder)), 1)
            weekly = save_snapshot(_fake_scan("2024-01-03"), directory=folder, weekly=True)
            self.assertFalse(weekly.created)
            index = pd.bdate_range("2024-01-02", periods=30)
            returns = pd.DataFrame(0.0, index=index, columns=list("ABCDEFGH"))
            returns.iloc[1:21, :5] = 0.01
            returns.iloc[1:21, 5:] = -0.01
            payload = {
                "session_date": "2024-01-02",
                "top": [{"symbol": name, "name": name, "score": 80 - i} for i, name in enumerate("ABCDE")],
                "ranked": [{"symbol": name, "name": name, "score": 80 - i} for i, name in enumerate("ABCDEFGH")],
            }
            graded = _grade(payload, returns, index, 20)
            self.assertEqual(graded.status, "graded")
            self.assertTrue(graded.correct)
            pending = _grade(payload, returns.iloc[:10], index[:10], 20)
            self.assertEqual(pending.status, "pending")

    def test_learning_autopsy_and_frozen_rule_stay_offline_weights(self):
        weights_before = dict(config.WEIGHTS)
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / "journal"
            learn = Path(temp) / "learn"
            journal.mkdir()
            payloads = []
            items = []
            for offset in range(8):
                session = f"2024-02-{offset + 1:02d}"
                extended = offset < 5
                top = []
                for i, symbol in enumerate("ABCDE"):
                    top.append(
                        {
                            "symbol": symbol,
                            "name": symbol,
                            "score": 80 - i,
                            "extension_penalty": 12.0 if extended and i < 2 else 0.0,
                            "dist_sma50": 0.08 if extended and i < 2 else -0.02,
                            "dist_sma200": 0.1,
                            "factors": {
                                "momentum": 70.0,
                                "relative_strength": 65.0,
                                "trend": 60.0,
                                "volume": 50.0,
                                "risk": 40.0,
                            },
                        }
                    )
                ranked = list(top)
                for i, symbol in enumerate("FGH"):
                    ranked.append(
                        {
                            "symbol": symbol,
                            "name": symbol,
                            "score": 40 - i,
                            "extension_penalty": 0.0,
                            "dist_sma50": -0.03,
                            "dist_sma200": 0.0,
                            "factors": {
                                "momentum": 40.0,
                                "relative_strength": 35.0,
                                "trend": 40.0,
                                "volume": 40.0,
                                "risk": 50.0,
                            },
                        }
                    )
                payload = {"session_date": session, "top": top, "ranked": ranked}
                (journal / f"{session}.json").write_text(
                    __import__("json").dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                payloads.append(payload)
                excess = -0.03 if extended else 0.02
                items.append(
                    ReviewItem(
                        session,
                        "graded",
                        [
                            NameOutcome(symbol, symbol, 80 - i, -0.04 if extended and i == 0 else excess)
                            for i, symbol in enumerate("ABCDE")
                        ],
                        -0.02 if extended else 0.03,
                        0.01,
                        excess,
                        excess > 0,
                        "graded",
                    )
                )
            report = ReviewReport(True, "ok", items, pending=0, graded=8, correct=3, wrong=5)
            status = learning_status(directory=journal, report=report)
            self.assertTrue(status.ready)
            self.assertEqual(status.need_more, 0)
            autopsy = run_autopsy(directory=journal, report=report, min_graded=8)
            self.assertTrue(autopsy.ok)
            self.assertTrue(autopsy.ready)
            self.assertGreaterEqual(autopsy.wrong_high_extension, 2)
            self.assertIn("MA50", autopsy.suggestion or "")
            filtered = _apply_ma50_cap(payloads[0], cap=0.05, top_n=5)
            self.assertIn("A", filtered["removed"])
            self.assertEqual(len(filtered["top_symbols"]), 5)
            rule = propose_rule(freeze_date="2024-02-04", directory=learn)
            self.assertEqual(rule.kind, "ma50_cap")
            self.assertEqual(config.WEIGHTS, weights_before)
            self.assertEqual(load_rules(learn)[0].rule_id, rule.rule_id)

            index = pd.bdate_range("2024-02-05", periods=40)
            returns = pd.DataFrame(0.0, index=index, columns=list("ABCDEFGH"))
            # بعد التجميد: الممتدون يضعفوا، والسحب يفوز — الفلتر المفروض يحسّن.
            for session_offset, session in enumerate(["2024-02-05", "2024-02-06", "2024-02-07"]):
                origin = session_offset
                returns.iloc[origin + 1 : origin + 21, :2] = -0.01
                returns.iloc[origin + 1 : origin + 21, 2:5] = 0.005
                returns.iloc[origin + 1 : origin + 21, 5:] = 0.002
                top = []
                for i, symbol in enumerate("ABCDE"):
                    top.append(
                        {
                            "symbol": symbol,
                            "name": symbol,
                            "score": 80 - i,
                            "extension_penalty": 15.0 if i < 2 else 0.0,
                            "dist_sma50": 0.12 if i < 2 else -0.02,
                            "factors": {"momentum": 60.0, "relative_strength": 60.0},
                        }
                    )
                ranked = list(top) + [
                    {
                        "symbol": symbol,
                        "name": symbol,
                        "score": 30,
                        "extension_penalty": 0.0,
                        "dist_sma50": -0.03,
                        "factors": {"momentum": 30.0, "relative_strength": 30.0},
                    }
                    for symbol in "FGH"
                ]
                payload = {"session_date": session, "top": top, "ranked": ranked}
                (journal / f"{session}.json").write_text(
                    __import__("json").dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            result = evaluate_rule(
                rule.rule_id,
                journal_dir=journal,
                learn_dir=learn,
                returns=returns,
                calendar=index,
            )
            self.assertTrue(result.ok)
            self.assertGreaterEqual(result.oos_n, 1)
            self.assertGreaterEqual(result.filtered_correct, result.baseline_correct)
            self.assertEqual(config.WEIGHTS, weights_before)

    def test_score_ignores_future_prices(self):
        price, high, low, volume, bench = _rising_market()
        before = score_panel(price, high, low, volume, bench)
        crashed = price.copy()
        crashed.iloc[300:] = crashed.iloc[300:] * 0.5
        high.iloc[300:] = high.iloc[300:] * 0.5
        low.iloc[300:] = low.iloc[300:] * 0.5
        after = score_panel(crashed, high, low, volume, bench)
        self.assertAlmostEqual(before["score"].iloc[250], after["score"].iloc[250], places=8)

    def test_clean_series_passes_and_a_jump_is_excluded(self):
        frame = _ohlcv()
        passed, reasons = assess(frame, frame.index)
        self.assertTrue(passed, reasons)
        broken = frame.copy()
        broken.loc[broken.index[100] :, ["open", "high", "low", "close", "adj"]] *= 1.6
        passed, reasons = assess(broken, broken.index)
        self.assertFalse(passed)
        self.assertTrue(any("قفزة" in reason for reason in reasons))

    def test_cairo_midnight_keeps_the_session_date(self):
        index = pd.DatetimeIndex([pd.Timestamp("2026-10-01 00:00:00", tz="Africa/Cairo")])
        self.assertEqual(str(naive_dates(index)[0].date()), "2026-10-01")

    def test_calendar_rejects_a_long_gap_and_stale_end(self):
        fresh = pd.bdate_range(end="2026-10-02", periods=450)
        passed, _ = assess_calendar(fresh, today=pd.Timestamp("2026-10-04"))
        self.assertTrue(passed)
        stale = pd.bdate_range(end="2026-08-01", periods=450)
        passed, message = assess_calendar(stale, today=pd.Timestamp("2026-10-04"))
        self.assertFalse(passed)
        self.assertIn("يوم", message)
        gapped = fresh.delete(np.arange(250, 290))
        passed, message = assess_calendar(gapped, today=pd.Timestamp("2026-10-04"))
        self.assertFalse(passed)
        self.assertIn("فجوة", message)


def _fake_scan(session: str) -> ScanResult:
    ranked = []
    for index, symbol in enumerate("ABCDEFGH"):
        ranked.append(
            StockView(
                symbol=symbol,
                yahoo=f"{symbol}.CA",
                name=symbol,
                sector="اختبار",
                included=True,
                reasons=[],
                score=90 - index,
                band="مرتفعة",
                factors={"momentum": 50.0},
                dist_sma50=0.01,
                dist_sma200=0.02,
                extension_penalty=0.0,
            )
        )
    return ScanResult(
        ok=True,
        message="ok",
        baseline="basket",
        baseline_note="note",
        baseline_label="باقي القائمة",
        case30_rows=1,
        case30_close=1.0,
        case30_date=session,
        calendar_end=session,
        basket_return_20d=-0.01,
        generated_at="now",
        stocks=ranked,
    )


def _ic_market():
    index = pd.bdate_range("2022-01-03", periods=900)
    names = [f"N{i:02d}" for i in range(15)]
    return index, names, decision_origins(index, 20, 3)


def _matched_signal(index, names, origins, sign: int):
    scores = pd.DataFrame(0.0, index=index, columns=names)
    eligible = pd.DataFrame(False, index=index, columns=names)
    returns = pd.DataFrame(0.0, index=index, columns=names)
    for origin in origins:
        _fill_signal(scores, eligible, returns, names, origin, sign)
    return scores, eligible, returns


def _fill_signal(scores, eligible, returns, names, origin: int, sign: int, horizon: int = 20) -> None:
    eligible.iloc[origin] = True
    for position, name in enumerate(names):
        level = len(names) - position
        scores.iloc[origin, position] = level
        returns.iloc[origin + 1 : origin + horizon + 1, position] = sign * level / 1000.0


def _path(early_return: float, lift: float, days: int = 260, quiet: int = 25) -> pd.Series:
    returns = np.full(days, early_return)
    lift_days = 25
    start = days - quiet - lift_days
    if lift > 1:
        returns[start : start + lift_days] = lift ** (1 / lift_days) - 1
    returns[-quiet:] = 0.0004
    index = pd.bdate_range("2022-01-03", periods=days)
    return pd.Series(100 * np.cumprod(1 + returns), index=index)


def _rising_market(days: int = 400):
    index = pd.bdate_range("2022-01-03", periods=days)
    price = pd.Series(100 * np.cumprod(np.repeat(1.001, days)), index=index)
    return price, price * 1.01, price * 0.99, pd.Series(1_000_000.0, index=index), pd.Series(0.0, index=index)


def _ohlcv(days: int = 500) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-10-02", periods=days)
    price = pd.Series(np.linspace(100, 160, days), index=index)
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "adj": price,
            "volume": 200_000.0,
        }
    )


if __name__ == "__main__":
    unittest.main()
