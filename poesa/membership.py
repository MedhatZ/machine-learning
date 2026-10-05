"""عضوية EGX30 من المراجعات المعلنة. الفترة اللي ملهاش إضافة وحذف بتفضل مجهولة."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from poesa.universe import UNIVERSE


@dataclass(frozen=True)
class Review:
    effective: str
    additions: tuple[str, ...]
    deletions: tuple[str, ...]
    source: str
    complete: bool = True


@dataclass(frozen=True)
class Interval:
    start: pd.Timestamp
    end: pd.Timestamp | None
    symbols: frozenset[str] | None


# لقطة المشروع بعد مراجعة 2026-09-01، متظبطة على المراجعات المعلنة.
# SWDY في اللقطة ومراجعة 2025-02-02 حذفاه من غير رجوع.
# MCQE دخل في 2025-08-03 ومن غير إعلان خروج، واللقطة كانت ساقطاه.
ANCHOR_DATE = "2026-09-01"
ANCHOR_NOTE = (
    "المرساة لقطة المشروع في 2026-09-01 بعد تصحيحين من المراجعات المعلنة، مش قائمة رسمية كاملة. "
    "SWDY اتشال لأن حذفه في 2025-02-02 من غير إعلان رجوع. "
    "MCQE اتضاف لأن دخوله في 2025-08-03 من غير إعلان خروج."
)
ANCHOR: tuple[str, ...] = tuple(sorted((*[item.symbol for item in UNIVERSE if item.symbol != "SWDY"], "MCQE")))

REVIEWS: tuple[Review, ...] = (
    Review(
        "2023-02-01",
        ("ADIB", "EFID", "TALM"),
        ("ALCN", "HDBK", "QNBA"),
        "https://www.egypttoday.com/Article/3/122238/3-companies-enter-EGX30-after-periodic-review",
    ),
    Review(
        "2023-08-01",
        ("BTFH", "BINV", "ALCN", "ORHD"),
        ("TALM", "ISPH", "CLHO", "RMDA"),
        "https://eltaameer.com/archives/131446",
    ),
    Review(
        "2024-02-01",
        ("PHAR", "ISPH", "SUGR"),
        ("CIRA", "CIEB", "BINV"),
        "https://www.dailynewsegypt.com/2024/02/04/egx-announces-changes-in-market-indices-after-semi-annual-review/",
    ),
    Review(
        "2024-08-01",
        ("EMFD", "BINV", "CIEB", "CLHO", "FAIT"),
        ("CCAP", "ISPH", "ORHD", "SUGR", "PHAR"),
        "https://enterpriseam.com/egypt/2024/07/31/emaar-misr-b-investments-credit-agricole-and-more-join-egx30-after-index-review/",
    ),
    Review(
        "2025-02-02",
        ("PHAR", "EGAL", "ISPH", "ORHD", "RMDA", "CCAP"),
        ("ESRS", "BINV", "FAIT", "CLHO", "SWDY", "HELI"),
        "https://mondovisione.com/media-and-resources/news/the-egyptian-exchange-egx-announces-the-semi-annual-review-of-its-market-indic-2025130/",
    ),
    Review(
        "2025-08-03",
        ("RAYA", "MCQE", "ARCC"),
        ("ALCN", "PHAR", "EFID"),
        "https://mondovisione.com/media-and-resources/news/the-egyptian-exchange-egx-announces-the-semi-annual-review-of-its-market-indic-2025730/",
    ),
    Review(
        "2026-02-01",
        ("EFID", "EGCH", "HELI", "OIH"),
        ("SKPC", "MFPC", "MASR", "CIEB"),
        "https://mondovisione.com/_assets/files/press_release_indices_feb_2026_english_english.pdf",
    ),
    Review(
        "2026-09-01",
        ("MFPC", "ALCN", "CLHO", "SKPC"),
        ("ARCC", "EGCH", "ORWE", "OIH"),
        "https://mondovisione.com/media-and-resources/news/the-egyptian-exchange-egx-announces-the-semi-annual-review-of-its-market-indic-2026816/",
    ),
)

EXTRA_NAMES = {
    "BINV": "بي إنفستمنتس",
    "CIEB": "كريدي أجريكول",
    "FAIT": "فيصل الإسلامي",
    "SUGR": "الدلتا للسكر",
    "PHAR": "إيبيكو",
    "CIRA": "سيرا للتعليم",
    "TALM": "تعليم",
    "HDBK": "التعمير والإسكان",
    "QNBA": "قطر الوطني الأهلي",
    "ESRS": "عز الدخيلة",
    "MCQE": "أسمنت قنا",
    "MASR": "مدينة مصر",
    "EGCH": "كيما",
    "OIH": "أوراسكوم للاستثمار",
    "ORWE": "النساجون الشرقيون",
    "ARCC": "العربية للأسمنت",
}


def research_symbols(
    anchor: tuple[str, ...] = ANCHOR,
    reviews: tuple[Review, ...] = REVIEWS,
) -> tuple[str, ...]:
    names = set(anchor)
    for review in reviews:
        names.update(review.additions)
        names.update(review.deletions)
    return tuple(sorted(names))


def membership_intervals(
    anchor_date: str = ANCHOR_DATE,
    anchor: tuple[str, ...] = ANCHOR,
    reviews: tuple[Review, ...] = REVIEWS,
) -> tuple[Interval, ...]:
    members = set(anchor)
    anchor_ts = pd.Timestamp(anchor_date)
    ordered = [
        review
        for review in sorted(reviews, key=lambda item: pd.Timestamp(item.effective))
        if pd.Timestamp(review.effective) <= anchor_ts
    ]
    intervals = [Interval(anchor_ts, None, frozenset(members))]
    start = anchor_ts
    for review in reversed(ordered):
        review_ts = pd.Timestamp(review.effective)
        if review_ts > start:
            continue
        if not review.complete or not _consistent(members, review):
            intervals.append(Interval(review_ts, start, None))
            intervals.append(Interval(pd.Timestamp("1990-01-01"), review_ts, None))
            break
        if review_ts < start:
            intervals.append(Interval(review_ts, start, frozenset(members)))
        members = _reverse(members, review)
        start = review_ts
    else:
        intervals.append(Interval(pd.Timestamp("1990-01-01"), start, frozenset(members)))
    return tuple(sorted(intervals, key=lambda item: item.start))


def membership_on(day: pd.Timestamp, intervals: tuple[Interval, ...]) -> frozenset[str] | None:
    stamp = pd.Timestamp(day)
    for interval in intervals:
        end_ok = interval.end is None or stamp < interval.end
        if interval.start <= stamp and end_ok:
            return interval.symbols
    return None


def membership_frame(
    index: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    intervals: tuple[Interval, ...] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    resolved = membership_intervals() if intervals is None else intervals
    member = pd.DataFrame(False, index=index, columns=list(symbols))
    known = pd.Series(False, index=index)
    for stamp in index:
        symbols_on = membership_on(stamp, resolved)
        if symbols_on is None:
            continue
        known.loc[stamp] = True
        present = [symbol for symbol in symbols_on if symbol in member.columns]
        if present:
            member.loc[stamp, present] = True
    return member, known


def gap_notes(intervals: tuple[Interval, ...]) -> list[str]:
    notes = []
    for interval in intervals:
        if interval.symbols is not None:
            continue
        start = "قبل أول مراجعة" if interval.start <= pd.Timestamp("1990-01-02") else str(interval.start.date())
        end = "مفتوحة" if interval.end is None else str(interval.end.date())
        notes.append(f"عضوية مجهولة من {start} إلى {end}")
    return notes


def _consistent(members: set[str], review: Review) -> bool:
    return set(review.additions) <= members and set(review.deletions).isdisjoint(members)


def _reverse(members: set[str], review: Review) -> set[str]:
    return (members - set(review.additions)) | set(review.deletions)
