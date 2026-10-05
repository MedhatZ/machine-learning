"""لقطة ثابتة لأسهم القاهرة. مش بتتسحب لحظياً من البورصة."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Listing:
    symbol: str
    name: str
    sector: str

    @property
    def yahoo(self) -> str:
        return f"{self.symbol}.CA"


# مراجعة سبتمبر 2026 المعلنة أضافت MFPC وALCN وCLHO وSKPC،
# وشالت ARCC وEGCH وORWE وOIH. فالمور VLMR مش متاح على ياهو.
# SWDY مضاف كسهم كبير سائل، والعضوية الرسمية ممكن تختلف عن اللقطة دي.
UNIVERSE: tuple[Listing, ...] = (
    Listing("COMI", "التجاري الدولي", "بنوك"),
    Listing("ETEL", "المصرية للاتصالات", "اتصالات"),
    Listing("TMGH", "طلعت مصطفى", "عقارات"),
    Listing("EGAL", "مصر للألومنيوم", "معادن"),
    Listing("MFPC", "موبكو", "أسمدة"),
    Listing("ABUK", "أبو قير للأسمدة", "أسمدة"),
    Listing("ALCN", "الإسكندرية للحاويات", "نقل"),
    Listing("ORAS", "أوراسكوم كونستراكشون", "مقاولات"),
    Listing("EAST", "الشرقية للدخان", "سلع استهلاكية"),
    Listing("EFIH", "إي فاينانس", "تكنولوجيا مالية"),
    Listing("SKPC", "سيدي كرير للبتروكيماويات", "بتروكيماويات"),
    Listing("CLHO", "كليوباترا", "رعاية صحية"),
    Listing("FWRY", "فوري", "تكنولوجيا مالية"),
    Listing("HRHO", "إي إف جي القابضة", "خدمات مالية"),
    Listing("ADIB", "أبوظبي الإسلامي", "بنوك"),
    Listing("AMOC", "أموك", "طاقة"),
    Listing("BTFH", "بلتون", "خدمات مالية"),
    Listing("EFID", "إيديتا", "أغذية"),
    Listing("EMFD", "إعمار مصر", "عقارات"),
    Listing("GBCO", "جي بي أوتو", "سيارات"),
    Listing("HELI", "مصر الجديدة للإسكان", "عقارات"),
    Listing("ISPH", "ابن سينا فارما", "رعاية صحية"),
    Listing("JUFO", "جهينة", "أغذية"),
    Listing("ORHD", "أوراسكوم للتنمية", "عقارات"),
    Listing("PHDC", "بالم هيلز", "عقارات"),
    Listing("CCAP", "القلعة", "استثمار"),
    Listing("RAYA", "راية", "خدمات مالية"),
    Listing("RMDA", "راميدا", "رعاية صحية"),
    Listing("OCDI", "سوديك", "عقارات"),
    Listing("SWDY", "السويدي إليكتريك", "صناعة"),
)


def find(token: str) -> Listing | None:
    key = token.strip().upper().removesuffix(".CA")
    for listing in UNIVERSE:
        if listing.symbol == key:
            return listing
    return None
