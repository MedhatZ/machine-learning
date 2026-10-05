"""لوحة ترتيب ومقارنة أسهم البورصة المصرية."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from poesa import config
from poesa.ai import write_report
from poesa.backtest import WalkForwardResult, run_walk_forward
from poesa.engine import ScanResult, StockView, run_scan
from poesa.journal import list_snapshots, review_snapshots, save_snapshot
from poesa.learn import evaluate_rule, learning_status, load_rules, propose_rule, run_autopsy
from poesa.text import (
    _quintile_line,
    _signal_label,
    _signal_line,
    autopsy_text,
    frozen_rules_text,
    journal_review_text,
    journal_save_text,
    learning_status_text,
    num,
    pct,
    price,
    rule_eval_text,
    share,
    stat_line,
)

st.set_page_config(page_title="محلل البورصة المصرية", layout="wide")
st.markdown(
    """
    <style>
      .stApp { direction: rtl; }
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { direction: rtl; }
      .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("محلل البورصة المصرية")
    st.caption(config.ADVICE_NOTE)
    refresh = st.button("تحديث البيانات")
    if refresh:
        st.session_state.pop("scan", None)
        st.session_state.pop("report", None)
        st.session_state["do_refresh"] = True
    if "scan" not in st.session_state:
        do_refresh = st.session_state.pop("do_refresh", False)
        with st.spinner("بيحمّل أسعار EGX ويحسب الدرجة. أول مرة ممكن تاخد دقايق."):
            st.session_state.scan = run_scan(refresh=do_refresh)
    scan: ScanResult = st.session_state.scan
    _banner(scan)
    if not scan.ok:
        st.error(scan.message)
    _metrics(scan)
    sectors = ["الكل", *sorted({stock.sector for stock in scan.ranked})]
    sector = st.selectbox("القطاع", sectors)
    shown = scan.ranked if sector == "الكل" else [stock for stock in scan.ranked if stock.sector == sector]
    st.subheader("أعلى 5 Setup حاليًا")
    st.caption("الدرجة وصف للـ setup الحالي. الباك تست لسه ما أثبتش إنها بتختار اللي هيتفوق في الـ20 جلسة الجاية.")
    st.dataframe(_forecast_table(scan.ranked[: config.TOP_N], scan.baseline_label), hide_index=True, width="stretch")
    st.subheader("دفتر التوقعات")
    st.caption(
        "نحفظ أعلى 5 مع تاريخ الجلسة، وبعد 20 جلسة نراجع هل سبقوا باقي الأسهم المرتبة وقت الحفظ. "
        f"التعلم من الأخطاء بعد {config.MIN_GRADED_FOR_LEARN} لقطات متقيّمة، من غير تدوير أوزان."
    )
    journal_cols = st.columns(4)
    if journal_cols[0].button("احفظ لقطة اليوم"):
        st.session_state.journal_save = save_snapshot(scan)
    if journal_cols[1].button("احفظ أسبوعيًا"):
        st.session_state.journal_save = save_snapshot(scan, weekly=True)
    if journal_cols[2].button("راجع اللقطات"):
        with st.spinner("بيحسب العوائد الفعلية بعد 20 جلسة..."):
            st.session_state.journal_review = review_snapshots()
    if journal_cols[3].button("حالة التعلم"):
        with st.spinner("بيراجع اللقطات المتقيّمة..."):
            st.session_state.journal_status = learning_status()
    if st.session_state.get("journal_save"):
        st.info(journal_save_text(st.session_state.journal_save))
    rows = list_snapshots()
    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "الجلسة": row["session_date"],
                        "أعلى 5": " ".join(row.get("top") or []),
                        "وقت الحفظ": row.get("saved_at") or "—",
                    }
                    for row in rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("مفيش لقطات محفوظة لسه.")
    if st.session_state.get("journal_status"):
        st.markdown(learning_status_text(st.session_state.journal_status).replace("\n", "  \n"))
    if st.session_state.get("journal_review"):
        st.markdown(journal_review_text(st.session_state.journal_review).replace("\n", "  \n"))
    learn_cols = st.columns(3)
    if learn_cols[0].button("تشريح الأخطاء"):
        with st.spinner("بيشرّح اللقطات الغلط..."):
            st.session_state.journal_autopsy = run_autopsy()
    if learn_cols[1].button("اقترح قاعدة MA50"):
        st.session_state.journal_rule = propose_rule()
    if learn_cols[2].button("اعرض القواعد"):
        st.session_state.journal_rules = load_rules()
    if st.session_state.get("journal_autopsy"):
        st.markdown(autopsy_text(st.session_state.journal_autopsy).replace("\n", "  \n"))
    if st.session_state.get("journal_rule"):
        st.info("اتسجّلت قاعدة مجمّدة. WEIGHTS لم تتغير.")
        st.markdown(frozen_rules_text([st.session_state.journal_rule]).replace("\n", "  \n"))
    if st.session_state.get("journal_rules") is not None:
        rules = st.session_state.journal_rules
        st.markdown(frozen_rules_text(rules).replace("\n", "  \n"))
        if rules:
            choice = st.selectbox("قيّم قاعدة خارج العيّنة", [rule.rule_id for rule in rules])
            if st.button("شغّل تقييم القاعدة") and choice:
                with st.spinner("بيقيّم الفلتر على اللقطات بعد التجميد..."):
                    st.session_state.journal_rule_eval = evaluate_rule(choice)
    if st.session_state.get("journal_rule_eval"):
        st.markdown(rule_eval_text(st.session_state.journal_rule_eval).replace("\n", "  \n"))
    st.subheader("الترتيب الكامل")
    st.dataframe(_rank_table(shown, scan.baseline_label), hide_index=True, width="stretch")
    st.subheader("ارتباط الدرجة بالـ20 جلسة التالية")
    st.caption("عضوية معلنة في يوم الاختيار، وعوامل حتى الإغلاق. الأوزان ثابتة والنتيجة مش بتغيرها.")
    if st.button("احسب الباك تست"):
        with st.spinner("بيحسب Rank IC والشرائح على نوافذ 20 جلسة..."):
            st.session_state.backtest = run_walk_forward()
    if st.session_state.get("backtest"):
        _backtest(st.session_state.backtest)

    with st.expander(f"مستبعدون ({len(scan.excluded)})"):
        if scan.excluded:
            st.dataframe(_excluded_table(scan), hide_index=True, width="stretch")
        else:
            st.write("مفيش أسهم مستبعدة.")

    labels = {f"{stock.symbol} — {stock.name}": stock for stock in scan.stocks if stock.chart is not None}
    if labels:
        st.subheader("السهم")
        picked = st.selectbox("اختار سهم", list(labels), key="detail")
        _detail(labels[picked], scan.baseline_label)

    ranked_labels = {f"{stock.symbol} — {stock.name}": stock.symbol for stock in scan.ranked}
    chosen = st.multiselect("مقارنة (حد أقصى 4)", list(ranked_labels), max_selections=4)
    if len(chosen) >= 2:
        _compare([next(stock for stock in scan.ranked if stock.symbol == ranked_labels[label]) for label in chosen], scan)

    if st.button("اكتب التقرير"):
        focus = [ranked_labels[label] for label in chosen] or [stock.symbol for stock in scan.ranked[: config.TOP_N]]
        with st.spinner("بيكتب التقرير من الأرقام المحسوبة..."):
            st.session_state.report = write_report(scan, focus)
    if st.session_state.get("report"):
        st.subheader("التقرير")
        st.markdown(st.session_state.report)

    with st.expander("المنهج"):
        for line in config.methodology_lines():
            st.write(line)


def _backtest(result: WalkForwardResult) -> None:
    if result.anchor_note:
        st.caption(result.anchor_note)
    for note in result.gap_notes:
        st.warning(note)
    if result.missing_yahoo:
        st.caption("مفيش تاريخ على ياهو: " + " ".join(result.missing_yahoo))
    st.caption(f"نوافذ بعضوية مجهولة: {result.gaps}")
    if not result.ok and not result.signals:
        st.error(result.message)
        return
    score = next((item for item in result.signals if item.name == "score"), None)
    if score is not None:
        st.write("الدرجة: " + _signal_line(score))
    if result.quintiles is not None:
        st.write(_quintile_line(result.quintiles))
    if result.horizon_ics:
        st.subheader("Rank IC على آفاق مختلفة")
        st.dataframe(_horizon_frame(result.horizon_ics), hide_index=True, width="stretch")
    if result.curves:
        st.subheader("شرائح العوامل وعائد 20 جلسة الزائد")
        st.caption("شرائح اقتصادية ثابتة قبل النتيجة، مش Quintiles. المتوسط يتخبى لو n أقل من 8.")
        for curve in result.curves:
            st.write(_signal_label(curve.name))
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "الشريحة": cell.label,
                            "العائد الزائد": "مخفي" if cell.hidden else pct(cell.mean),
                            "n": cell.n,
                        }
                        for cell in curve.buckets
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    factor_rows = [item for item in result.signals if item.name != "score"]
    if factor_rows:
        st.subheader("كل عامل لوحده على 20 جلسة")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "العامل": _signal_label(item.name),
                        "Rank IC": "مخفي" if item.hidden else f"{item.mean:+.3f}",
                        "وسيط IC": "مخفي" if item.hidden else f"{item.median:+.3f}",
                        "موجب": "مخفي" if item.hidden else share(item.positive),
                        "n": item.n,
                        "أعلى ناقص أقل": "مخفي" if item.spread_hidden else pct(item.spread),
                    }
                    for item in factor_rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.subheader("مقارنة أعلى 5 بالباقي")
    columns = st.columns(len(result.slices) or 1)
    for column, block in zip(columns, result.slices):
        column.metric(
            f"صافي آخر {block.years} سنوات",
            "عينة غير كافية" if block.hidden else pct(block.net),
        )
    if result.slices:
        summary = []
        for block in result.slices:
            summary.append(
                {
                    "الفترة": f"آخر {block.years} سنوات",
                    "النوافذ": block.n,
                    "أفضل 5": "مخفي" if block.hidden else pct(block.top),
                    "الباقي": "مخفي" if block.hidden else pct(block.rest),
                    "التفوق": "مخفي" if block.hidden else pct(block.gross),
                    "الصافي": "مخفي" if block.hidden else pct(block.net),
                    "نوافذ إيجابية": "مخفي" if block.hidden else share(block.beat),
                    "وسيط التفوق": "مخفي" if block.hidden else pct(block.median),
                }
            )
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    st.caption(
        f"نوافذ اترفضت: {result.skipped}. الصافي يخصم 0.8% مرة كل نافذة. "
        "المقارنة دي سطر ثانوي. النوافذ مش مستقلة تماماً، وأي فترة عضويتها مجهولة متشالة."
    )
    if result.windows:
        rows = [
            {
                "التاريخ": window.date,
                "أفضل 5": pct(window.top_return),
                "الباقي": pct(window.rest_return),
                "التفوق": pct(window.excess),
                "الأسماء": " ".join(window.top),
            }
            for window in result.windows
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _horizon_frame(blocks) -> pd.DataFrame:
    names = ["score"]
    for block in blocks:
        for item in block.signals:
            if item.name not in names:
                names.append(item.name)
    rows = []
    for block in blocks:
        by_name = {item.name: item for item in block.signals}
        row = {"الأفق": block.horizon, "n": block.n}
        for name in names:
            item = by_name.get(name)
            row[_signal_label(name)] = "—" if item is None or item.hidden or item.mean is None else f"{item.mean:+.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _banner(scan: ScanResult) -> None:
    st.info(scan.baseline_note)
    st.warning(config.SURVIVORSHIP_NOTE)
    st.caption(config.COST_NOTE)
    st.caption(config.SAMPLE_NOTE)


def _metrics(scan: ScanResult) -> None:
    columns = st.columns(4)
    columns[0].metric("أسهم في الترتيب", len(scan.ranked))
    columns[1].metric("مستبعدون", len(scan.excluded))
    columns[2].metric("عائد السلة 20 جلسة", pct(scan.basket_return_20d))
    columns[3].metric("شمعات ^CASE30", scan.case30_rows)
    st.caption(f"آخر جلسة في التقويم: {scan.calendar_end or '—'} · الحساب: {scan.generated_at}")


def _forecast_table(stocks: list[StockView], baseline: str) -> pd.DataFrame:
    rows = []
    for rank, stock in enumerate(stocks, start=1):
        rows.append(
            {
                "#": rank,
                "الرمز": stock.symbol,
                "الشركة": stock.name,
                "درجة 20 جلسة": num(stock.score),
                "عقوبة": num(stock.extension_penalty),
                "بعد MA50": pct(stock.dist_sma50),
                "بعد MA200": pct(stock.dist_sma200),
                "20 جلسة": stat_line(stock.stats.get(config.DECISION_HORIZON), baseline),
                "زخم قصير": num(stock.factors.get("momentum")),
                "قوة نسبية": num(stock.factors.get("relative_strength")),
                "اتجاه": num(stock.factors.get("trend")),
                "حجم واختراق": num(stock.factors.get("volume")),
                "مخاطر": num(stock.factors.get("risk")),
            }
        )
    return pd.DataFrame(rows)


def _rank_table(stocks: list[StockView], baseline: str) -> pd.DataFrame:
    rows = []
    for stock in stocks:
        block = stock.stats.get(20, {})
        rows.append(
            {
                "الرمز": stock.symbol,
                "الشركة": stock.name,
                "القطاع": stock.sector,
                "السعر": price(stock.price),
                "درجة 20 جلسة": num(stock.score),
                "الشريحة": stock.band,
                "زخم قصير": num(stock.factors.get("momentum")),
                "قوة نسبية": num(stock.factors.get("relative_strength")),
                "اتجاه": num(stock.factors.get("trend")),
                "حجم واختراق": num(stock.factors.get("volume")),
                "مخاطر": num(stock.factors.get("risk")),
                "بعد MA200": pct(stock.dist_sma200),
                "عقوبة": num(stock.extension_penalty),
                "20 جلسة": stat_line(block, baseline),
            }
        )
    return pd.DataFrame(rows)


def _excluded_table(scan: ScanResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "الرمز": [stock.symbol for stock in scan.excluded],
            "الشركة": [stock.name for stock in scan.excluded],
            "آخر سعر": [price(stock.price) for stock in scan.excluded],
            "السبب": ["؛ ".join(stock.reasons) for stock in scan.excluded],
        }
    )


def _detail(stock: StockView, baseline: str) -> None:
    if not stock.included:
        st.error("مستبعد: " + "؛ ".join(stock.reasons))
    else:
        columns = st.columns(4)
        columns[0].metric("درجة 20 جلسة", f"{num(stock.score)} · {stock.band}")
        columns[1].metric("عقوبة الامتداد", num(stock.extension_penalty))
        columns[2].metric("البعد عن متوسط 50", pct(stock.dist_sma50))
        columns[3].metric("البعد عن متوسط 200", pct(stock.dist_sma200))
        horizon_rows = []
        for horizon in config.HORIZONS:
            block = stock.stats.get(horizon, {})
            horizon_rows.append(
                {
                    "الأفق": f"{horizon} جلسات",
                    "الحالات": block.get("n", 0),
                    "الإجمالي": "مخفي" if block.get("hidden") else pct(block.get("gross")),
                    "الصافي": "مخفي" if block.get("hidden") else pct(block.get("net")),
                    f"تفوق على {baseline}": "مخفي" if block.get("hidden") else share(block.get("beat")),
                }
            )
        st.dataframe(pd.DataFrame(horizon_rows), hide_index=True, width="stretch")
    if stock.chart is not None and not stock.chart.empty:
        st.plotly_chart(_chart(stock), width="stretch")


def _compare(stocks: list[StockView], scan: ScanResult) -> None:
    st.subheader("المقارنة")
    rows = []
    for stock in stocks:
        rows.append(
            {
                "السهم": f"{stock.symbol} {stock.name}",
                "درجة 20 جلسة": num(stock.score),
                **{config.FACTOR_LABELS[name]: num(stock.factors.get(name)) for name in config.WEIGHTS},
                "20 جلسة": stat_line(stock.stats.get(20), scan.baseline_label),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    figure = go.Figure()
    for stock in stocks:
        figure.add_bar(name=stock.symbol, x=list(config.FACTOR_LABELS.values()), y=[stock.factors.get(name) for name in config.WEIGHTS])
    figure.update_layout(template="plotly_dark", barmode="group", height=420, paper_bgcolor="#121417", plot_bgcolor="#1c2128")
    st.plotly_chart(figure, width="stretch")


def _chart(stock: StockView) -> go.Figure:
    frame = stock.chart
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.04)
    figure.add_trace(go.Scatter(x=frame.index, y=frame["close"], name="الإغلاق"), row=1, col=1)
    for column, label in (("sma20", "متوسط 20"), ("sma50", "متوسط 50"), ("sma200", "متوسط 200")):
        if column in frame:
            figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=label), row=1, col=1)
    figure.add_trace(go.Bar(x=frame.index, y=frame["volume"], name="الحجم"), row=2, col=1)
    if "rsi" in frame:
        figure.add_trace(go.Scatter(x=frame.index, y=frame["rsi"], name="RSI"), row=3, col=1)
        figure.add_hline(y=70, row=3, col=1, line_dash="dot", line_color="#8a8478")
        figure.add_hline(y=30, row=3, col=1, line_dash="dot", line_color="#8a8478")
    figure.update_layout(
        template="plotly_dark",
        height=720,
        paper_bgcolor="#121417",
        plot_bgcolor="#1c2128",
        legend_orientation="h",
        title=f"{stock.symbol} — {stock.name}",
    )
    return figure


main()
