"""Evaluate TotalSpineSeg and the new discs labeling method against manual GT.

GT files are named <stem><gt_suffix>.nii.gz; they are symlinked as <stem>.nii.gz
into <out_dir>/gt so that discslabel's evaluate() can pair them with predictions
named <stem>.nii.gz.

Usage:
    python run_eval.py --gt-dir GT --gt-suffix _label-discs_dlabel \
        --tss-dir TSS --new-dir NEW --out-dir OUT [--image-dir IMAGES]

Writes OUT/metrics_tss/metrics.json and OUT/metrics_new/metrics.json.
Needs the discslabel package (discs-labeling-multicontrast) installed.
"""

import argparse
import os

from discslabel.evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--gt-dir", required=True)
parser.add_argument("--gt-suffix", required=True, help="e.g. _label-discs_dlabel")
parser.add_argument("--tss-dir", required=True)
parser.add_argument("--new-dir", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--image-dir", default="", help="images for error previews (optional)")
parser.add_argument("--tolerance-mm", type=float, default=10.0)
args = parser.parse_args()

# GT with the suffix removed, so stems match the prediction names
gt_link_dir = os.path.join(args.out_dir, "gt")
os.makedirs(gt_link_dir, exist_ok=True)
ext = ".nii.gz"
for name in sorted(os.listdir(args.gt_dir)):
    if not name.endswith(args.gt_suffix + ext):
        print(f"skip GT (suffix mismatch): {name}")
        continue
    link = os.path.join(gt_link_dir, name[: -len(args.gt_suffix + ext)] + ext)
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(args.gt_dir, name)), link)

for run_name, pred_dir in (("tss", args.tss_dir), ("new", args.new_dir)):
    print(f"\n===== {run_name}: {pred_dir}")
    evaluate(
        gt_dir=gt_link_dir,
        pred_dir=pred_dir,
        out_dir=os.path.join(args.out_dir, f"metrics_{run_name}"),
        run_name=run_name,
        num_workers=0,  # inline: easier to debug
        image_dir=args.image_dir,
        previews="errors" if args.image_dir else "none",
        tolerance_mm=args.tolerance_mm,
    )
