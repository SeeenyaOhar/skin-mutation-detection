
from __future__ import annotations

RAW11 = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCCKA",
         "INF", "BEN_OTH", "MAL_OTH"]

HEAD9 = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCCKA", "OTH_BEN"]

TO_HEAD9 = {
    "MEL": "MEL", "NV": "NV", "BCC": "BCC", "AKIEC": "AKIEC", "BKL": "BKL",
    "DF": "DF", "VASC": "VASC", "SCCKA": "SCCKA",
    "INF": "OTH_BEN", "BEN_OTH": "OTH_BEN",
    "MAL_OTH": None,
}

MILK_MALIGNANT = {"BCC", "SCCKA", "MEL", "AKIEC", "MAL_OTH"}
PAD_MALIGNANT = {"BCC", "SCC", "MEL", "ACK"}

PAD_TO_HEAD9 = {"BCC": "BCC", "SCC": "SCCKA", "MEL": "MEL",
                "ACK": "AKIEC", "NEV": "NV", "SEK": "BKL"}

UA = {
    "MEL": "меланома",
    "NV": "меланоцитарний невус",
    "BCC": "базальноклітинна карцинома",
    "AKIEC": "актинічний кератоз / хвороба Боуена",
    "BKL": "доброякісний кератоз",
    "DF": "дерматофіброма",
    "VASC": "судинне ураження",
    "SCCKA": "плоскоклітинна карцинома / кератоакантома",
    "INF": "запальне ураження",
    "BEN_OTH": "інше доброякісне",
    "MAL_OTH": "інше злоякісне",
    "OTH_BEN": "інше доброякісне (INF + BEN_OTH)",
}
