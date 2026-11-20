"""Paired statistics between TotalSpineSeg and the new method.

Reads OUT/metrics_tss/metrics.json and OUT/metrics_new/metrics.json (from
run_eval.py), keeps the images scored by both methods, and runs a two-sided
Wilcoxon signed-rank test per metric:

  * per image: F1@tol, recall, precision, mean distance (mm)
  * per disc : distance (mm) of discs matched by BOTH methods
    (discs of a same image are not independent: read this one as secondary)

Usage:
    python stats.py --out-dir OUT [--tolerance-mm 10]

Writes OUT/per_image.csv and OUT/stats.csv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", required=True)
parser.add_argument("--tolerance-mm", type=float, default=10.0)
args = parser.parse_args()
tol = args.tolerance_mm
METHODS = ("tss", "new")


def load(run):
    with open(os.path.join(args.out_dir, f"metrics_{run}", "metrics.json")) as f:
        return {p["image"]: p for p in json.load(f)["per_image"]}


def image_metrics(p):
    dists = [m["distance_mm"] for m in p["matched"]]
    hits = sum(d <= tol for d in dists)
    return {
        "f1": 2 * hits / (p["n_gt"] + p["n_pred"]) if p["n_gt"] + p["n_pred"] else np.nan,
        "recall": p["n_matched"] / p["n_gt"] if p["n_gt"] else np.nan,
        "precision": p["n_matched"] / p["n_pred"] if p["n_pred"] else np.nan,
        "dist_mean": np.mean(dists) if dists else np.nan,
    }


runs = {m: load(m) for m in METHODS}
common = sorted(set(runs["tss"]) & set(runs["new"]))
for m in METHODS:
    print(f"{m}: {len(runs[m])} images, {len(set(runs[m]) - set(common))} not in the other method")
print(f"{len(common)} common images\n")

# Per-image table: one row per image, columns <metric>_<method>
rows = []
for img in common:
    row = {"image": img}
    for m in METHODS:
        row.update({f"{k}_{m}": v for k, v in image_metrics(runs[m][img]).items()})
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out_dir, "per_image.csv"), index=False)

# Per-disc paired distances (discs matched by both methods)
a, b = [], []
for img in common:
    dt = {m["label"]: m["distance_mm"] for m in runs["tss"][img]["matched"]}
    dn = {m["label"]: m["distance_mm"] for m in runs["new"][img]["matched"]}
    for lbl in sorted(set(dt) & set(dn)):
        a.append(dt[lbl])
        b.append(dn[lbl])
pairs = {f"{k} (per image)": (df[f"{k}_tss"], df[f"{k}_new"]) for k in ("f1", "recall", "precision", "dist_mean")}
pairs["distance (per disc)"] = (pd.Series(a), pd.Series(b))

stats = []
for name, (x, y) in pairs.items():
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    diff = y - x
    p = wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
    stats.append({
        "metric": name,
        "n": len(x),
        "tss_mean": x.mean(), "tss_std": x.std(), "tss_median": np.median(x),
        "new_mean": y.mean(), "new_std": y.std(), "new_median": np.median(y),
        "new_better": int(np.sum(diff < 0) if "dist" in name else np.sum(diff > 0)),
        "tss_better": int(np.sum(diff > 0) if "dist" in name else np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "p_wilcoxon": p,
    })
stats = pd.DataFrame(stats)
stats.to_csv(os.path.join(args.out_dir, "stats.csv"), index=False)

pd.set_option("display.width", 200)
print(f"F1 at {tol:g} mm; distances in mm (lower is better)")
print(stats.round(4).to_string(index=False))
