"""Stage every sample of the discslabel data JSONs into a flat QC folder.

For each entry (TRAINING / VALIDATION / TESTING) of every ``--data-json``:
  * copy the IMAGE to  ``<qc_dir>/images/<dataset>__<basename>.nii.gz``
  * copy the LABEL to  ``<qc_dir>/labels/<dataset>__<basename>.nii.gz``  (ground truth)

``<dataset>`` is the first segment of the BIDS-relative path, the same naming as
``prepare_evaluation.sh``, so ``discslabel_evaluate`` pairs GT / prediction directly.

An image listed in several JSONs / splits is staged once; ``<qc_dir>/qc_index.json``
records every place it is used, so the review / apply steps know which JSON
entries to drop and which BIDS label file to replace.

Usage:
    python qc_stage.py --bids-root $DISCSLABEL_DATA/bids --qc-dir <qc_dir> \
        --data-json <data.json> [<evaluation.json> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

SPLITS = ("TRAINING", "VALIDATION", "TESTING")


def stem_of(image_rel: str) -> str:
    base = os.path.basename(image_rel)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return f"{image_rel.split('/')[0]}__{base}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids-root", required=True, help="Folder holding the BIDS datasets ($DISCSLABEL_DATA/bids).")
    parser.add_argument("--qc-dir", required=True, help="QC output folder.")
    parser.add_argument("--data-json", nargs="+", required=True, help="discslabel data JSON(s) to QC.")
    args = parser.parse_args()

    images_dir = os.path.join(args.qc_dir, "images")
    labels_dir = os.path.join(args.qc_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    index_path = os.path.join(args.qc_dir, "qc_index.json")
    samples: dict = {}
    n_copied = n_skipped = n_missing = 0
    for data_json in args.data_json:
        data_json = os.path.abspath(data_json)
        with open(data_json) as f:
            data = json.load(f)
        for split in SPLITS:
            for entry in data.get(split, []):
                image_rel, label_rel = entry.get("IMAGE"), entry.get("LABEL")
                if not image_rel or not label_rel:
                    continue
                stem = stem_of(image_rel)
                sample = samples.setdefault(
                    stem, {"image": image_rel, "label": label_rel, "uses": []}
                )
                if sample["image"] != image_rel:
                    print(f"[stage] WARN: stem collision {stem}: {sample['image']} vs {image_rel}")
                    continue
                if sample["label"] != label_rel:
                    print(f"[stage] WARN: {stem} has two different labels: {sample['label']} vs {label_rel} "
                          f"(QC uses the first one)")
                sample["uses"].append({"json": data_json, "split": split})

    for stem, sample in samples.items():
        src_img = os.path.join(args.bids_root, sample["image"])
        src_lbl = os.path.join(args.bids_root, sample["label"])
        if not os.path.isfile(src_img) or not os.path.isfile(src_lbl):
            if n_missing < 10:
                print(f"[stage] missing in BIDS: {src_img if not os.path.isfile(src_img) else src_lbl}")
            elif n_missing == 10:
                print("[stage] ... (more missing files, see 'missing' in qc_index.json)")
            sample["missing"] = True
            n_missing += 1
            continue
        dst_img = os.path.join(images_dir, stem + ".nii.gz")
        dst_lbl = os.path.join(labels_dir, stem + ".nii.gz")
        if os.path.isfile(dst_img) and os.path.isfile(dst_lbl):
            n_skipped += 1
            continue
        shutil.copyfile(src_img, dst_img)
        shutil.copyfile(src_lbl, dst_lbl)
        n_copied += 1

    index = {
        "bids_root": os.path.abspath(args.bids_root),
        "data_jsons": [os.path.abspath(p) for p in args.data_json],
        "samples": samples,
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[stage] {len(samples)} unique samples: copied={n_copied} skipped={n_skipped} "
          f"missing_bids={n_missing} -> {args.qc_dir}")


if __name__ == "__main__":
    main()
