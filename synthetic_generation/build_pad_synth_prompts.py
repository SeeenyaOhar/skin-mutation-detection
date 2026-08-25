
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from train.build_synth_prompts import (
    BODY_MARKER, STRENGTHS, TEMPLATE, load_pools, stable_rng,
)
from captions import taxonomy as CT

PAD_TO_CAPTION_CLASS = {
    "ACK": "AK", "BCC": "BCC", "MEL": "MEL", "NEV": "NV", "SCC": "SCC", "SEK": "BKL",
}

PAD_REGION_PHRASE = {
    "FACE": "face", "FOREARM": "forearm", "CHEST": "chest", "BACK": "back",
    "ARM": "arm", "NOSE": "nose", "HAND": "hand", "NECK": "neck",
    "THIGH": "thigh", "EAR": "ear", "ABDOMEN": "abdomen", "LIP": "lip",
    "SCALP": "scalp", "FOOT": "foot",
}

SYMPTOM_PHRASE = {
    "bleed": "the lesion has been reported to bleed",
    "itch": "the lesion has been reported to itch",
    "changed": "the lesion has recently changed in appearance",
    "grew": "the lesion has recently grown in size",
    "hurt": "the lesion has been reported to be painful",
}

def build_pad_text(prompt_id: str, pad_class: str, region: str, row: dict, pools: dict,
                   body_mode: str = "raw") -> tuple[str, str]:
    rng = stable_rng(prompt_id)
    site_phrase = PAD_REGION_PHRASE.get(region, "skin")
    caption_cls = PAD_TO_CAPTION_CLASS[pad_class]

    extra = []
    d1, d2 = row.get("diameter_1"), row.get("diameter_2")
    try:
        if d1 and d2:
            extra.append(f"approximately {float(d1):.0f} by {float(d2):.0f} millimetres in size")
    except ValueError:
        pass
    if row.get("elevation", "").strip().upper() == "TRUE":
        extra.append("palpably raised above the surrounding skin")
    for col, phrase in SYMPTOM_PHRASE.items():
        if row.get(col, "").strip().upper() == "TRUE":
            extra.append(phrase)
    extra_sentence = (" This specific photograph additionally shows: " + ", ".join(extra) + ".") if extra else ""

    pool = pools.get(caption_cls)
    if pool:
        slugs = CT.CLASS_TO_SLUGS[caption_cls]
        slug = slugs[rng.randrange(len(slugs))]
        malignancy = CT.CLASS_MALIGNANCY[caption_cls]
        body = pool[rng.randrange(len(pool))]["body"] + extra_sentence
        body_source = "dermaflux_caption_pool+pad_meta"
    else:
        slug = caption_cls.lower()
        malignancy = CT.CLASS_MALIGNANCY.get(caption_cls, "benign")
        body = (", ".join(extra) + ".") if extra else "no distinctive features noted."
        body_source = "pad_meta_only"

    text = TEMPLATE.format(malignancy=malignancy, disease=slug, site=site_phrase,
                           body_marker=BODY_MARKER, body=body)
    return text, body_source


def load_pad_parents(metadata_path: Path, images_root: Path) -> list[dict]:
    by_name = {p.name: p for p in images_root.glob("imgs_part_*/**/*.png")}

    rows = []
    with metadata_path.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            if r["diagnostic"] not in PAD_TO_CAPTION_CLASS:
                continue
            found = by_name.get(r["img_id"])
            if found is None:
                continue
            r["_path"] = str(found)
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pad-metadata", type=Path, required=True)
    ap.add_argument("--pad-images-root", type=Path, required=True,
                    help="dir containing imgs_part_1/2/3")
    ap.add_argument("--pools", type=Path, default=Path("pools"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", nargs="*", default=list(PAD_TO_CAPTION_CLASS))
    ap.add_argument("--max-parents-per-class", type=int, default=None)
    ap.add_argument("--children-per-parent", type=int, default=1)
    ap.add_argument("--strength-range", nargs=2, type=float, default=None,
                    metavar=("LOW", "HIGH"),
                    help="draw each child's strength uniformly from [LOW, HIGH] "
                         "instead of cycling STRENGTHS; deterministic per parent+child.")
    ap.add_argument("--txt2img-frac", type=float, default=0.0,
                    help="fraction of each class's parents assigned pure txt2img "
                         "(strength=1.0, same prompt, no init latent) instead of "
                         "img2img — total image count per class stays capped at "
                         "--max-parents-per-class either way. 0.5 = half/half.")
    args = ap.parse_args()

    pools = load_pools(args.pools)
    parents = load_pad_parents(args.pad_metadata, args.pad_images_root)

    by_class: dict[str, list[dict]] = {c: [] for c in args.classes}
    for r in parents:
        if r["diagnostic"] in by_class:
            by_class[r["diagnostic"]].append(r)

    rows = []
    for cls in args.classes:
        plist = by_class[cls]
        if args.max_parents_per_class:
            plist = plist[:args.max_parents_per_class]
        made = 0
        n_txt2img = round(len(plist) * args.txt2img_frac)
        split_order = sorted(plist, key=lambda p: stable_rng(f"{cls}-split-{p['img_id']}").random())
        txt2img_ids = {p["img_id"] for p in split_order[:n_txt2img]}

        for p in plist:
            img_id = p["img_id"]
            parent_stem = Path(img_id).stem

            if img_id in txt2img_ids:
                t2i_prompt_id = f"{parent_stem}_{cls}_txt2img"
                text, body_source = build_pad_text(t2i_prompt_id, cls, p.get("region", ""), p, pools)
                rows.append({
                    "prompt_id": t2i_prompt_id,
                    "class": cls,
                    "variant": "txt2img_pad_v1",
                    "parent_img_id": img_id,
                    "parent_path": p["_path"],
                    "strength": 1.0,
                    "seed": 0,
                    "body_source": body_source,
                    "text": text,
                })
                made += 1
                continue

            for k in range(args.children_per_parent):
                if args.strength_range:
                    lo, hi = args.strength_range
                    strength_rng = stable_rng(f"{cls}-{img_id}-strength-{k}")
                    strength = round(strength_rng.uniform(lo, hi), 2)
                else:
                    strength = STRENGTHS[k % len(STRENGTHS)]
                seed = k
                prompt_id = f"{parent_stem}_{cls}_s{int(strength*100):03d}"
                if args.children_per_parent > 1:
                    prompt_id += f"-{seed}"
                text, body_source = build_pad_text(prompt_id, cls, p.get("region", ""), p, pools)
                rows.append({
                    "prompt_id": prompt_id,
                    "class": cls,
                    "variant": "img2img_pad_v1",
                    "parent_img_id": img_id,
                    "parent_path": p["_path"],
                    "strength": strength,
                    "seed": seed,
                    "body_source": body_source,
                    "text": text,
                })
                made += 1
        print(f"{cls}: {len(plist)} PAD parents -> {made} synth")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
