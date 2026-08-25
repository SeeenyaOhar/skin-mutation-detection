
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from captions import taxonomy as CT


RAW11_TO_CAPTION_CLASS = {
    "MEL": "MEL", "NV": "NV", "BCC": "BCC", "AKIEC": "AK", "BKL": "BKL",
    "DF": "DF", "VASC": "VASC", "SCCKA": "SCC",
}

SITE_PHRASE = {
    "head_neck_face": "face", "trunk": "trunk", "lower_extremity": "leg",
    "upper_extremity": "arm", "hand": "hand", "foot": "foot", "genital": "groin",
}

MONET_PHRASE = {
    "MONET_ulceration_crust": "an ulcerated, crusted surface",
    "MONET_hair": "surrounding coarse hair",
    "MONET_vasculature_vessels": "visible small blood vessels",
    "MONET_erythema": "surrounding redness",
    "MONET_pigmented": "a pigmented, brownish colour",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid": "a moist, glistening surface",
    "MONET_skin_markings_pen_ink_purple_pen": "nearby skin marking",
}
MONET_THRESHOLD = 0.5
FALLBACK_DISEASE_NAME = {"INF": "inflammatory", "BEN_OTH": "benign", "MAL_OTH": "malignant"}
FALLBACK_MALIGNANT = {"MAL_OTH"}

BODY_MARKER = "Here's a description of the mole:"

VERDICT_RE = re.compile(
    r"\b(overall|consistent with|benign|malignan|cancer|diagnos|suggestive of|"
    r"concerning|red flag|melanoma|nevus|nevi|carcinoma|keratosis|"
    r"dermatofibroma|angioma|granuloma|lesion type)", re.I)


LOCATION_RE = re.compile(
    r"\b(located|situated|located on|aspect of|on the (face|cheek|nose|scalp|ear|"
    r"lip|neck|chest|back|abdomen|arm|forearm|hand|thigh|leg|knee|foot|toe|trunk|"
    r"shoulder|extremity))\b", re.I)
SURFACE_RE = re.compile(
    r"\b(bleed|scab|crust|ulcerat|erythema|redness|smooth surface|hair)", re.I)


def strip_verdicts(body: str, drop_surface: bool = False) -> str:
    parts = re.split(r"(?<=[.!?])\s+", body.replace("\n", " "))
    kept = []
    for p in parts:
        p = p.strip()
        if not p or VERDICT_RE.search(p) or LOCATION_RE.search(p):
            continue
        if drop_surface and SURFACE_RE.search(p):
            continue
        kept.append(p)
    return " ".join(kept)

TEMPLATE = (
    "The image depicts a lesion, which is classified as {malignancy}. "
    "It appears to have the {disease} disease. "
    "This is a clinical photograph: a close-up photograph of skin taken with "
    "an ordinary camera in room lighting, showing the lesion together with "
    "the surrounding skin and its natural texture. It is not a dermoscopic "
    "image — there is no dermatoscope, no polarised light, no immersion "
    "fluid, and no circular dark border. The lesion is on the {site}. "
    "{body_marker} {body}"
)

STRENGTHS = [0.5, 0.6, 0.7]


def stable_rng(key: str) -> random.Random:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def load_pools(pools_dir: Path) -> dict[str, list[dict]]:
    pools = {}
    for f in pools_dir.glob("*.json"):
        if f.stem == "summary":
            continue
        pools[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return pools

TARGET_TOTAL = {
    "BKL": 600, "SCCKA": 600, "AKIEC": 600, "DF": 600, "VASC": 600,
    "INF": 324, "BEN_OTH": 288,
}
MAX_SYNTH_MULT = 8


def build_text(prompt_id: str, cls: str, site: str, monet: dict[str, float],
               pools: dict[str, list[dict]], body_mode: str = "stripped") -> tuple[str, str]:
    rng = stable_rng(prompt_id)
    site_phrase = SITE_PHRASE.get(site, "skin")

    monet_phrases = [p for col, p in MONET_PHRASE.items() if monet.get(col, 0.0) > MONET_THRESHOLD]
    monet_sentence = (
        " This specific photograph additionally shows " + ", ".join(monet_phrases) + "."
        if monet_phrases else ""
    )

    caption_cls = RAW11_TO_CAPTION_CLASS.get(cls)
    pool = pools.get(caption_cls) if caption_cls else None
    if pool:
        slugs = CT.CLASS_TO_SLUGS[caption_cls]
        slug = slugs[rng.randrange(len(slugs))]
        malignancy = CT.CLASS_MALIGNANCY[caption_cls]
        if body_mode == "monet_only":
            body = (", ".join(monet_phrases) + ".") if monet_phrases else "no distinctive features noted."
            body_source = "monet_only"
        else:
            raw = pool[rng.randrange(len(pool))]["body"]
            mined = (raw if body_mode == "raw"
                     else strip_verdicts(raw, drop_surface=bool(monet_phrases)))
            body = (mined + monet_sentence).strip()
            body_source = f"dermaflux_caption_pool_{body_mode}+monet"
    else:
        slug = FALLBACK_DISEASE_NAME[cls]
        malignancy = "malignant" if cls in FALLBACK_MALIGNANT else "benign"
        body = (", ".join(monet_phrases) + ".") if monet_phrases else "no distinctive features noted."
        body_source = "monet_fallback"

    text = TEMPLATE.format(malignancy=malignancy, disease=slug, site=site_phrase,
                           body_marker=BODY_MARKER, body=body)
    return text, body_source


def load_metadata(path: Path) -> dict[str, dict]:
    monet_cols = None
    out = {}
    with path.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        monet_cols = [c for c in rd.fieldnames or [] if c.startswith("MONET")]
        for r in rd:
            out[r["isic_id"]] = {
                "site": r.get("site", ""),
                "monet": {c: float(r[c]) for c in monet_cols if r.get(c) not in (None, "")},
            }
    return out


def load_real_clinical_parents(pool_path: Path, classes: list[str]) -> dict[str, list[dict]]:
    by_class: dict[str, list[dict]] = {c: [] for c in classes}
    with pool_path.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            if r["split"] != "train" or r["dataset"] != "milk10k":
                continue
            if not r["image_type"].startswith("clinical"):
                continue
            if r["raw11"] not in by_class:
                continue
            by_class[r["raw11"]].append(r)
    return by_class


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("data/pool_raw11_rebuilt.csv"))
    ap.add_argument("--milk-metadata", type=Path, required=True)
    ap.add_argument("--pools", type=Path, default=Path("pools"),
                    help="real caption-body pools from captions.build_class_pools")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", nargs="*", default=list(TARGET_TOTAL))
    ap.add_argument("--max-parents-per-class", type=int, default=None,
                    help="pilot knob: cap how many real parents are used")
    ap.add_argument("--max-children-per-parent", type=int, default=None,
                    help="pilot knob: cap children/parent below the 8x rule")
    ap.add_argument("--body-mode", choices=["stripped", "monet_only", "raw"],
                    default="stripped",
                    help="stripped: mined body minus verdict/location/"
                         "MONET-contradicting sentences. monet_only: drop the "
                         "mined body entirely. raw: original unmodified body.")
    ap.add_argument("--strengths", nargs="*", type=float, default=STRENGTHS,
                    help="img2img strengths cycled across a parent's children. "
                         "Lower keeps more of the parent's diagnostic texture. "
                         "Ignored if --strength-range is given.")
    ap.add_argument("--strength-range", nargs=2, type=float, default=None,
                    metavar=("LOW", "HIGH"),
                    help="draw each child's strength uniformly from [LOW, HIGH] "
                         "instead of cycling --strengths; still deterministic "
                         "(seeded from class+parent+child index).")
    args = ap.parse_args()
    strengths = args.strengths

    md = load_metadata(args.milk_metadata)
    parents_by_class = load_real_clinical_parents(args.pool, args.classes)
    pools = load_pools(args.pools)

    rows = []
    for cls in args.classes:
        parents = parents_by_class[cls]
        if args.max_parents_per_class:
            parents = parents[:args.max_parents_per_class]
        if not parents:
            print(f"[warn] no real clinical parents found for {cls}")
            continue

        n_real = len(parents_by_class[cls])
        target_synth = min(TARGET_TOTAL[cls] - n_real, MAX_SYNTH_MULT * n_real)
        target_synth = max(target_synth, 0)
        children_per_parent = math.ceil(target_synth / len(parents)) if parents else 0
        if args.max_children_per_parent:
            children_per_parent = min(children_per_parent, args.max_children_per_parent)

        made = 0
        for parent in parents:
            info = md.get(parent["isic_id"], {"site": "", "monet": {}})
            for k in range(children_per_parent):
                if made >= target_synth and not args.max_children_per_parent:
                    break
                if args.strength_range:
                    lo, hi = args.strength_range
                    strength_rng = stable_rng(f"{cls}-{parent['isic_id']}-strength-{k}")
                    strength = round(strength_rng.uniform(lo, hi), 2)
                else:
                    strength = strengths[k % len(strengths)]
                seed = k
                prompt_id = f"synth-{cls}-{parent['isic_id']}-s{int(strength*100):03d}-{seed}"
                text, body_source = build_text(prompt_id, cls, info["site"], info["monet"],
                                               pools, args.body_mode)
                rows.append({
                    "prompt_id": prompt_id,
                    "class": cls,
                    "variant": "img2img_v1",
                    "parent_isic_id": parent["isic_id"],
                    "parent_path": parent["path"],
                    "strength": strength,
                    "seed": seed,
                    "body_source": body_source,
                    "text": text,
                })
                made += 1

        print(f"{cls}: {n_real} real parents -> {made} synth "
              f"({children_per_parent}/parent, target was {target_synth})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
