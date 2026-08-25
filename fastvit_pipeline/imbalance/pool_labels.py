
from __future__ import annotations

ISIC2019_TO_RAW11 = {
    "MEL": "MEL", "NV": "NV", "BCC": "BCC", "AK": "AKIEC",
    "BKL": "BKL", "DF": "DF", "VASC": "VASC", "SCC": "SCCKA",
}

DERM12345_TO_RAW11 = {
    "Nevus": "NV",
    "Lentigo simplex": "NV",
    "Ink-spot lentigo": "NV",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Lichen planus like keratosis": "BKL",
    "Basal cell carcinoma": "BCC",
    "Squamous cell carcinoma, NOS": "SCCKA",
    "Squamous cell carcinoma in situ": "AKIEC",
    "Solar or actinic keratosis": "AKIEC",
    "Melanoma, NOS": "MEL",
    "Melanoma Invasive": "MEL",
    "Melanoma in situ": "MEL",
    "Dermatofibroma": "DF",
    "Hemangioma": "VASC",
    "Lymphangioma": "VASC",
    "Angiokeratoma": "VASC",
    "Vascular spider": "VASC",
    "Pyogenic granuloma": "VASC",
    "Kaposi sarcoma": "MAL_OTH",
    "Dermatofibrosarcoma protuberans": "MAL_OTH",
    "Paget disease": "MAL_OTH",
}

DERM7PT_TO_RAW11 = {
    "clark nevus": "NV",
    "reed or spitz nevus": "NV",
    "dermal nevus": "NV",
    "blue nevus": "NV",
    "congenital nevus": "NV",
    "combined nevus": "NV",
    "recurrent nevus": "NV",
    "melanoma": "MEL",
    "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "melanoma metastasis": "MEL",
    "basal cell carcinoma": "BCC",
    "seborrheic keratosis": "BKL",
    "lentigo": "BKL",
    "melanosis": "BKL",
    "dermatofibroma": "DF",
    "vascular lesion": "VASC",
}

HIBA_TO_RAW11 = {
    "nevus": "NV",
    "basal cell carcinoma": "BCC",
    "melanoma": "MEL",
    "squamous cell carcinoma": "SCCKA",
    "seborrheic keratosis": "BKL",
    "actinic keratosis": "AKIEC",
    "dermatofibroma": "DF",
    "vascular lesion": "VASC",
    "solar lentigo": "BKL",
    "lichenoid keratosis": "BKL",
}

MAPPINGS = {
    "isic2019": ISIC2019_TO_RAW11,
    "derm12345": DERM12345_TO_RAW11,
    "derm7pt": DERM7PT_TO_RAW11,
    "hiba": HIBA_TO_RAW11,
}
