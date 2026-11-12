"""
Find the margin of sc_crop.py (--margin RL AP SI, mm) needed so that the crop contains the whole segmentation,
for every image of a flat folder, and the margin that works for all (or most) images.

For each image of IMAGES_DIR (*.nii.gz):
    1. Run the sc_crop prediction once (the YOLO model is loaded once for all images) and get the box without
       margin, in voxels of the resampled LAS image (1 x 1 x 10 mm, RL AP SI), like sc_crop.find_box.
    2. Load the segmentation with the same name in SEGS_DIR (all non-zero voxels, or only --labels), and bring
       the center of each segmented voxel to the voxels of the resampled image (through the world coordinates).
       sc_crop keeps the voxels whose center is inside the box, so the margin needed on each side is the distance
       (mm) between the box and the farthest segmented voxel center outside it (0 if all are inside).
    3. Save one row per image in OUTPUT_DIR/margins.csv (written as it goes: when the script is run again, the
       images already in the csv are skipped, so it can be resumed after a crash or a killed job).

At the end, the summary (OUTPUT_DIR/summary.txt) gives, per axis (RL AP SI), the max and some percentiles of the
needed margin over the images, and the number of images fully covered with each candidate margin. sc_crop adds
the same margin on both sides of an axis, so the needed margin of an axis is the max of its two sides (the csv
also has each side, if the box is off-center for some structures).

Usage (on the server):
    python find_margins.py --images-dir images/ --segs-dir segs/ --model best.pt \
        --sc-crop-dir ~/code/sc-crop --output-dir margins_out
    # only some structures of the segmentation, e.g. labels 1 2 and 100 to 200:
    python find_margins.py ... --labels 1 2 100-200

Author: Nathan Molinier
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

AXES = ("RL", "AP", "SI")
SIDES = (("R", "L"), ("A", "P"), ("I", "S"))   # (low, high) side of each axis of the LAS grid
PERCENTILES = (50, 90, 95, 99, 100)


def get_parser():
    parser = argparse.ArgumentParser(description="Find the sc_crop margin needed to keep the whole segmentation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images-dir", required=True, type=Path, help="Flat folder of images (nii.gz).")
    parser.add_argument("--segs-dir", required=True, type=Path,
                        help="Flat folder of segmentations, same names as the images (see --seg-suffix).")
    parser.add_argument("--model", required=True, type=Path, help="YOLO weights (best.pt of a training).")
    parser.add_argument("--sc-crop-dir", type=Path, default=Path(__file__).resolve().parents[2] / "sc-crop",
                        help="Folder of the sc-crop repository (with sc_crop.py).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for margins.csv and summary.txt.")
    parser.add_argument("--seg-suffix", default="",
                        help="Suffix added to the image name to get the segmentation name "
                             "(e.g. '_label-spine_dseg' for sub-01_T2w.nii.gz -> sub-01_T2w_label-spine_dseg.nii.gz).")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Values of the segmentation to keep in the crop, e.g. '1 2 100-200' (default: all "
                             "non-zero values).")
    parser.add_argument("--structure", choices=("sc", "canal"), default="canal", help="Structure to crop around.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (model trained on RGB slices).")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25],
                        help="Confidence threshold: one value for all classes, or one value per class (SC canal).")
    parser.add_argument("--margins", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 40, 50],
                        help="Candidate margins (mm) for the coverage table of the summary.")
    return parser


def parse_labels(labels):
    """Return the sorted array of the values given as 'v' or 'a-b' (inclusive), or None for all non-zero."""
    if not labels:
        return None
    values = set()
    for item in labels:
        if "-" in item.strip("-"):
            a, b = item.split("-", 1)
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(item))
    return np.array(sorted(values))


def name_of(path):
    return path.name.removesuffix(".gz").removesuffix(".nii")


def needed_margins(box, resampled_affine, seg_path, labels, voxel_size):
    """Return (needed margin of each side (6 values, mm, order R L A P I S), number of segmented voxels) so that
    all the segmented voxel centers are inside the box (lower, upper) of the resampled grid."""
    seg = nib.load(seg_path)
    data = np.asarray(seg.dataobj)
    if data.ndim > 3:
        data = data[..., 0]
    mask = np.isin(data, labels) if labels is not None else data != 0
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return None, 0
    # Segmented voxel centers -> world -> voxels of the resampled grid. The resampled grid keeps the directions of
    # the image (permutation/flip + scaling), so the extremes along each axis are reached at voxel centers.
    to_resampled = np.linalg.inv(resampled_affine) @ seg.affine
    pts = nib.affines.apply_affine(to_resampled, ijk.astype(np.float64))
    lower, upper = box
    low_side = np.maximum(lower - pts.min(axis=0), 0) * voxel_size    # R, A, I sides (mm)
    high_side = np.maximum(pts.max(axis=0) - upper, 0) * voxel_size   # L, P, S sides (mm)
    return np.stack([low_side, high_side], axis=1).ravel(), len(ijk)


def summarize(rows, candidates):
    """Return the summary text of the per-image needed margins."""
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    lines = [f"{len(rows)} images: {len(ok)} ok, {len(failed)} failed (no detection, no segmentation, error...)"]
    for r in failed:
        lines.append(f"    {r['image']}: {r['status']}")
    if not ok:
        return "\n".join(lines)
    need = np.array([[float(r[f"margin_{a}"]) for a in AXES] for r in ok])

    lines += ["", "Needed margin (mm) per axis (max of the 2 sides), over the images:",
              f"{'':>10}" + "".join(f"{a:>10}" for a in AXES)]
    for p in PERCENTILES:
        label = "max" if p == 100 else f"p{p}"
        lines.append(f"{label:>10}" + "".join(f"{v:>10.1f}" for v in np.percentile(need, p, axis=0)))

    # Margin that works for all the images (ceil to the mm), and to keep 95% / 99% of the images on each axis
    for p in (100, 99, 95):
        m = np.ceil(np.percentile(need, p, axis=0) - 1e-6)
        covered = np.all(need <= m + 1e-6, axis=1).sum()
        label = "all images" if p == 100 else f"{p}% of the images on each axis"
        lines.append(f"\nMargin for {label}: --margin {' '.join(f'{v:g}' for v in m)}"
                     f"  ({covered}/{len(ok)} images fully covered)")

    lines += ["", "Images fully covered with one margin for all axes (--margin M):"]
    for m in candidates:
        covered = np.all(need <= m + 1e-6, axis=1).sum()
        lines.append(f"    {m:>6g} mm: {covered}/{len(ok)} ({100 * covered / len(ok):.1f}%)")

    lines += ["", "Images needing the largest margin (check the detection / the segmentation):"]
    for i in np.argsort(-need.max(axis=1))[:10]:
        lines.append(f"    {ok[i]['image']}: " + ", ".join(f"{a} {v:.1f}" for a, v in zip(AXES, need[i]))
                     + f" (detected on {ok[i]['n_detected']}/{ok[i]['n_slices']} slices)")
    return "\n".join(lines)


def main():
    args = get_parser().parse_args()
    sys.path.insert(0, str(args.sc_crop_dir.expanduser().resolve()))
    sys.path.insert(0, str(args.sc_crop_dir.expanduser().resolve() / "inference"))
    from predict_image import get_thresholds, predict_slices
    from preprocess_image import VOXEL_SIZE, cut_slices, resample_image
    from sc_crop import STRUCTURES, find_box
    from ultralytics import YOLO

    labels = parse_labels(args.labels)
    voxel_size = np.array(VOXEL_SIZE, dtype=float)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "margins.csv"
    fields = (["image", "status", "n_slices", "n_detected", "n_seg_voxels"]
              + [f"margin_{s}" for pair in SIDES for s in pair] + [f"margin_{a}" for a in AXES])

    # Resume: keep the rows already computed
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    done = {r["image"] for r in rows}

    images = sorted(args.images_dir.glob("*.nii.gz"))
    todo = [p for p in images if p.name not in done]
    print(f"{len(images)} images, {len(done)} already in {csv_path}, {len(todo)} to process")

    model = YOLO(args.model)
    thresholds = get_thresholds(args.conf, model.names)
    cls = STRUCTURES[args.structure]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for i, image in enumerate(todo, 1):
            start = time.time()
            row = {"image": image.name}
            seg_path = args.segs_dir / f"{name_of(image)}{args.seg_suffix}.nii.gz"
            try:
                if not seg_path.exists():
                    raise FileNotFoundError("no segmentation")
                resampled = resample_image(image)
                preds = predict_slices(model, cut_slices(np.asarray(resampled.dataobj), args.rgb), thresholds)
                row["n_slices"] = len(preds)
                row["n_detected"] = sum(cls in p for p in preds)
                box = find_box(preds, cls, resampled.shape, 0)
                if box is None:
                    raise RuntimeError(f"no {args.structure} detected")
                sides, n_vox = needed_margins(box, resampled.affine, seg_path, labels, voxel_size)
                row["n_seg_voxels"] = n_vox
                if sides is None:
                    raise RuntimeError("empty segmentation")
                for (low, high), (m_low, m_high), axis in zip(SIDES, sides.reshape(3, 2), AXES):
                    row[f"margin_{low}"], row[f"margin_{high}"] = round(m_low, 2), round(m_high, 2)
                    row[f"margin_{axis}"] = round(max(m_low, m_high), 2)
                row["status"] = "ok"
                msg = ", ".join(f"{a} {row[f'margin_{a}']}" for a in AXES)
            except Exception as e:   # one bad image must not stop the whole run
                row["status"] = f"failed: {e}".replace("\n", " ")
                msg = row["status"]
                if not isinstance(e, (FileNotFoundError, RuntimeError)):
                    traceback.print_exc()
            writer.writerow(row)
            f.flush()
            rows.append({k: str(row.get(k, "")) for k in fields})
            print(f"[{i}/{len(todo)}] {image.name}: {msg} ({time.time() - start:.1f} s)", flush=True)

    summary = summarize(rows, args.margins)
    (args.output_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved {csv_path} and {args.output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
