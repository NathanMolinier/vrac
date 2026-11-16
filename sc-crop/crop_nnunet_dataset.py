"""
Crop a whole nnUNet (v2) raw dataset around the spinal canal (default) or the spinal cord (SC), to make a new
dataset with the same case and file names (e.g. Dataset101_X -> Dataset102_X).

For each case:
    1. Detect the structure on the reference channel (default _0000) and compute the crop box (see sc_crop.py)
    2. Apply the SAME crop to every channel of the case and to its label, so they keep the same grid
    3. If nothing is detected: copy the case uncropped (default) or skip it (--on-fail skip)
    4. Measure how much of the label is outside the crop box, in mm on each side (R L A P S I)

Outputs (in OUTPUT dataset folder):
    imagesTr/, labelsTr/, imagesTs/, labelsTs/ (when present in the input), same file names
    dataset.json  : copied from the input (numTraining = number of training cases written)
    crop_log.csv  : box, shapes, label cut off (voxels, %, mm3) and label beyond the box on each side (mm)
    crop.log      : everything printed during the run (command, parameters, warnings, errors, summary)
    qc/ (--qc-dir): <case>.png for the cases whose label is cut off (--qc all: every case), and summary.png

Usage:
    python crop_nnunet_dataset.py --input $nnUNet_raw/Dataset101_X --output $nnUNet_raw/Dataset102_X \
        --model runs/detect/train/weights/best.pt --margin 5 5 20
"""
import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed (cluster)
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# sc_crop.py (same folder) also puts inference/ on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_crop import STRUCTURES, find_box, get_margins, to_image_voxels  # noqa: E402
from predict_image import get_thresholds, predict_slices  # noqa: E402
from preprocess_image import cut_slices, resample_image  # noqa: E402

SPLITS = [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]
# RAS voxel axes: (side of the low indices, side of the high indices)
AXIS_SIDES = [("L", "R"), ("P", "A"), ("I", "S")]
SIDES = ["R", "L", "A", "P", "S", "I"]   # same order as --margin (RL AP SI)
CSV_FIELDS = (["split", "case", "status", "detected_slices", "box_xyz", "original_shape", "cropped_shape",
               "lost_label_voxels", "lost_label_percent", "lost_label_mm3"]
              + [f"overflow_{s}_mm" for s in SIDES])
logger = logging.getLogger("crop_nnunet_dataset")


def get_parser():
    parser = argparse.ArgumentParser(description="Crop a whole nnUNet dataset around the spinal canal or the SC.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Input dataset folder (e.g. Dataset101_X).")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset folder (e.g. Dataset102_X).")
    parser.add_argument("--model", required=True, type=Path, help="YOLO weights (best.pt of a training).")
    parser.add_argument("--structure", choices=STRUCTURES, default="sc", help="Structure to crop around.")
    parser.add_argument("--channel", default="0000", help="Channel used for the detection.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (model trained on RGB slices).")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.26],
                        help="Confidence threshold: one value for all classes, or one per class (SC canal).")
    parser.add_argument("--margin", type=float, nargs="+", default=[0],
                        help="Margin on both sides of the box (mm): one value, or one per axis (RL AP SI).")
    parser.add_argument("--on-fail", choices=["copy", "skip"], default="copy",
                        help="Case with no detection: copy it uncropped, or leave it out of the new dataset.")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Log file (default: OUTPUT/crop.log). Appended to if it exists.")
    parser.add_argument("--qc", choices=["lost", "all", "none"], default="lost",
                        help="Per-case figures: cases whose label is cut off, every case, or none "
                             "(summary.png is always made).")
    parser.add_argument("--qc-dir", type=Path, default=None, help="Folder of the figures (default: OUTPUT/qc).")
    return parser


def setup_logging(log_file):
    """Log to the console and to `log_file` (appended), with the time and level of each message in the file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(log_file, mode="a")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console)
    logger.addHandler(file)


def fail(msg):
    """Log the error, then stop."""
    logger.error(msg)
    raise SystemExit(msg)


def group_cases(folder, ending):
    """Return {case: {channel: path}} for the images of an nnUNet images folder (<case>_<channel><ending>)."""
    cases = {}
    for f in sorted(folder.glob(f"*{ending}")):
        case, _, channel = f.name[:-len(ending)].rpartition("_")
        cases.setdefault(case, {})[channel] = f
    return cases


def detect_crop(ref, model, cls, rgb, thresholds, margin):
    """Return (crop slices on the grid of `ref`, number of slices with a detection, number of slices), with crop
    None if the structure is not detected or the box is outside the image."""
    resampled = resample_image(ref)
    preds = predict_slices(model, cut_slices(np.asarray(resampled.dataobj), rgb), thresholds)
    n_detected = sum(cls in p for p in preds)
    box = find_box(preds, cls, resampled.shape, margin)
    if box is None:
        return None, n_detected, len(preds)
    crop = to_image_voxels(*box, resampled.affine, nib.load(ref))
    if any(s.stop <= s.start for s in crop):
        return None, n_detected, len(preds)
    return crop, n_detected, len(preds)


# ---------------------------------------------------------------------------------------------------------------
# Label cut off by the crop, and figures
# ---------------------------------------------------------------------------------------------------------------
def to_ras(affine):
    """Return (function reorienting a 3D array of this grid to RAS voxel axes, voxel size (mm) along R, A, S).
    Only flips and permutations of the axes: the crop box stays a box."""
    ornt = nib.orientations.io_orientation(affine)
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    ras_zooms = np.empty(3)
    ras_zooms[ornt[:, 0].astype(int)] = zooms
    return (lambda a: nib.orientations.apply_orientation(a, ornt)), ras_zooms


def extent(mask, axis):
    """(first, last) index of `mask` along `axis`, or None if the mask is empty."""
    idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
    return (int(idx[0]), int(idx[-1])) if idx.size else None


def label_overflow_mm(label_ras, box_ras, zooms):
    """{side: mm of the label beyond the crop box on that side} for the sides R L A P S I (RAS arrays). It is the
    extra margin that side would need to keep the whole label."""
    out = dict.fromkeys(SIDES, 0.0)
    for axis, (low, high) in enumerate(AXIS_SIDES):
        lab, box = extent(label_ras, axis), extent(box_ras, axis)
        if lab is None:
            return out
        out[low] = round(max(0, box[0] - lab[0]) * zooms[axis], 1)
        out[high] = round(max(0, lab[1] - box[1]) * zooms[axis], 1)
    return out


def save_case_png(path, img_ras, label_ras, box_ras, zooms, title):
    """Sagittal and coronal views (label projected over all slices) and the axial slice that loses the most label:
    image, label kept (green), label cut off (red), crop box (yellow)."""
    kept, lost = label_ras & box_ras, label_ras & ~box_ras
    vmin, vmax = np.percentile(img_ras, [1, 99])
    center = [int(np.median(i)) for i in np.nonzero(label_ras)] if label_ras.any() else \
        [n // 2 for n in img_ras.shape]
    z_axial = int(np.argmax(lost.sum(axis=(0, 1)))) if lost.any() else center[2]
    # (name, axis removed, index of the background slice, label projected over all slices?)
    views = [("Sagittal", 0, center[0], True), ("Coronal", 1, center[1], True),
             (f"Axial (slice {z_axial})", 2, z_axial, False)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, (name, axis, index, project) in zip(axes, views):
        h, v = [a for a in range(3) if a != axis]   # horizontal and vertical axes of the view
        aspect = zooms[v] / zooms[h]
        show = dict(origin="lower", aspect=aspect, interpolation="nearest")
        ax.imshow(np.take(img_ras, index, axis=axis).T, cmap="gray", vmin=vmin, vmax=vmax, **show)
        for mask, color in ((kept, (0, 1, 0)), (lost, (1, 0, 0))):
            proj = (mask.any(axis=axis) if project else np.take(mask, index, axis=axis)).T
            rgba = np.zeros(proj.shape + (4,))
            rgba[proj] = (*color, 0.55)
            ax.imshow(rgba, **show)
        (h0, h1), (v0, v1) = extent(box_ras, h), extent(box_ras, v)
        ax.add_patch(Rectangle((h0 - 0.5, v0 - 0.5), h1 - h0 + 1, v1 - v0 + 1, fill=False, edgecolor="yellow",
                               linewidth=1.5))
        ax.set_xlabel(f"{AXIS_SIDES[h][0]}  ←  →  {AXIS_SIDES[h][1]}")
        ax.set_ylabel(f"{AXIS_SIDES[v][0]}  ←  →  {AXIS_SIDES[v][1]}")
        ax.set_title(name + (" — label: all slices" if project else ""))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title + "\ngreen: label kept   red: label cut off   yellow: crop box", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_summary_png(path, rows, margin):
    """Where the label is cut off (mm beyond the box, per side) and whether it comes from missed detections."""
    rows = [r for r in rows if "overflow_R_mm" in r]
    lost_rows = [r for r in rows if r["lost_label_voxels"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    if lost_rows:
        data = [[r[f"overflow_{s}_mm"] for r in lost_rows] for s in SIDES]
        ax1.boxplot(data, showfliers=False)
        rng = np.random.default_rng(0)
        for i, values in enumerate(data, 1):
            ax1.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=6, alpha=0.4)
        ax1.set_xticks(range(1, 7), SIDES)
    ax1.set_xlabel("Side of the crop box")
    ax1.set_ylabel("Label beyond the box (mm)")
    ax1.set_title(f"Where the label is cut off ({len(lost_rows)}/{len(rows)} cases with a label)\n"
                  f"margin used: {' '.join(f'{m:g}' for m in get_margins(margin))} mm (RL AP SI)")

    detected = [int(r["detected_slices"].split("/")[0]) / max(1, int(r["detected_slices"].split("/")[1]))
                for r in rows]
    percent = [r["lost_label_percent"] for r in rows]
    ax2.scatter(detected, percent, s=8, alpha=0.5)
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Fraction of the axial slices where the structure is detected")
    ax2.set_ylabel("% of the label cut off")
    ax2.set_title("Label cut off vs detection: losses at low fractions = missed slices")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def suggest_margin(rows, margin, q=95):
    """Margin (mm, RL AP SI) that would have kept the whole label in q% of the cases (approximate: the box also
    snaps to voxel centers)."""
    rows = [r for r in rows if "overflow_R_mm" in r]
    if not rows:
        return None
    current = get_margins(margin)
    return [round(float(current[i] + np.percentile([max(r[f"overflow_{a}_mm"], r[f"overflow_{b}_mm"])
                                                    for r in rows], q)), 1)
            for i, (a, b) in enumerate([("R", "L"), ("A", "P"), ("S", "I")])]


# ---------------------------------------------------------------------------------------------------------------
def crop_case(case, channels, args, images_dir, labels_dir, ending, model, cls, thresholds, prefix, qc_dir):
    """Crop all the channels and the label of one case, save them and return the row of crop_log.csv."""
    ref = nib.load(channels[args.channel])
    shape = tuple(int(n) for n in ref.shape[:3])
    label = args.input / labels_dir / f"{case}{ending}"
    files = list(channels.values()) + ([label] if label.exists() else [])
    for f in files:
        if nib.load(f).shape[:3] != shape:
            raise ValueError(f"{f.name} does not have the shape of the _{args.channel} image {shape}")

    crop, n_detected, n_slices = detect_crop(channels[args.channel], model, cls, args.rgb, thresholds, args.margin)
    row = {"split": images_dir, "case": case, "status": "cropped", "detected_slices": f"{n_detected}/{n_slices}",
           "original_shape": shape}
    if crop is None:
        row["status"] = f"not detected ({args.on_fail})"
        logger.warning(f"{prefix}: {args.structure} not detected, case "
                       + ("skipped" if args.on_fail == "skip" else "copied uncropped"))
        if args.on_fail == "skip":
            return row
        crop = tuple(slice(0, n) for n in shape)

    for f in channels.values():
        nib.save(nib.load(f).slicer[crop], args.output / images_dir / f.name)

    box = np.zeros(shape, dtype=bool)
    box[crop] = True
    if label.exists():
        lab = nib.load(label)
        data = np.asanyarray(lab.dataobj)
        lost = int(np.count_nonzero(data[~box]))
        total = int(np.count_nonzero(data))
        ras, zooms = to_ras(ref.affine)
        label_ras, box_ras = ras(data != 0), ras(box)
        row.update(lost_label_voxels=lost,
                   lost_label_percent=round(100 * lost / total, 2) if total else 0,
                   lost_label_mm3=round(lost * float(np.prod(lab.header.get_zooms()[:3])), 1))
        row.update({f"overflow_{s}_mm": mm for s, mm in label_overflow_mm(label_ras, box_ras, zooms).items()})
        sides = ", ".join(f"{s} {row[f'overflow_{s}_mm']:g}" for s in SIDES if row[f"overflow_{s}_mm"])
        if lost:
            logger.warning(f"{prefix}: {lost} label voxels ({row['lost_label_percent']}% of the label) outside "
                           f"the crop, label beyond the box (mm): {sides}")
        nib.save(lab.slicer[crop], args.output / labels_dir / label.name)

        if args.qc == "all" or (args.qc == "lost" and lost):
            img = np.asanyarray(ref.dataobj)
            img = img if img.ndim == 3 else img.reshape(img.shape[:3] + (-1,))[..., 0]
            save_case_png(qc_dir / f"{case}.png", ras(img.astype(np.float32)), label_ras, box_ras, zooms,
                          f"{case} — detected on {n_detected}/{n_slices} slices — "
                          f"{lost} voxels cut off ({row['lost_label_percent']}%)"
                          + (f" — beyond the box (mm): {sides}" if sides else ""))

    row["cropped_shape"] = tuple(s.stop - s.start for s in crop)
    row["box_xyz"] = " ".join(f"{s.start}-{s.stop - 1}" for s in crop)
    logger.info(f"{prefix}: {row['status']}, detected on {n_detected}/{n_slices} slices, "
                f"{shape} -> {row['cropped_shape']}")
    return row


def main():
    args = get_parser().parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--output must be different from --input")
    setup_logging(args.log_file or args.output / "crop.log")
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Command: " + " ".join(sys.argv))
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    try:
        get_margins(args.margin)   # check the margin before loading anything
    except SystemExit as e:
        fail(str(e))
    if not (args.input / "dataset.json").is_file():
        fail(f"No dataset.json in {args.input}")
    dataset_json = json.loads((args.input / "dataset.json").read_text())
    ending = dataset_json.get("file_ending", ".nii.gz")

    model = YOLO(args.model)
    thresholds = get_thresholds(args.conf, model.names)
    cls = STRUCTURES[args.structure]
    args.output.mkdir(parents=True, exist_ok=True)
    qc_dir = args.qc_dir or args.output / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    log, n_train = [], 0
    for images_dir, labels_dir in SPLITS:
        if not (args.input / images_dir).is_dir():
            continue
        cases = group_cases(args.input / images_dir, ending)
        logger.info(f"{images_dir}: {len(cases)} cases")
        (args.output / images_dir).mkdir(exist_ok=True)
        if (args.input / labels_dir).is_dir():
            (args.output / labels_dir).mkdir(exist_ok=True)

        for i, (case, channels) in enumerate(cases.items(), 1):
            prefix = f"[{images_dir} {i}/{len(cases)}] {case}"
            if args.channel not in channels:
                logger.warning(f"{prefix}: no _{args.channel} image, case skipped")
                log.append({"split": images_dir, "case": case, "status": f"error: no _{args.channel} image"})
                continue
            try:
                row = crop_case(case, channels, args, images_dir, labels_dir, ending, model, cls, thresholds,
                                prefix, qc_dir)
            except Exception as e:   # unreadable file, shape mismatch...: log it and go on with the next case
                logger.error(f"{prefix}: {type(e).__name__}: {e} -> case NOT in the output dataset")
                for f in list(channels.values()) + [args.input / labels_dir / f"{case}{ending}"]:
                    (args.output / f.parent.name / f.name).unlink(missing_ok=True)   # no half-written case
                log.append({"split": images_dir, "case": case, "status": f"error: {type(e).__name__}: {e}"})
                continue
            log.append(row)
            if images_dir == "imagesTr" and row["status"] != "not detected (skip)":
                n_train += 1

    dataset_json["numTraining"] = n_train
    (args.output / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    for extra in args.input.glob("*.json"):   # e.g. splits or other metadata
        if extra.name != "dataset.json":
            shutil.copy2(extra, args.output / extra.name)

    with open(args.output / "crop_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(log)
    save_summary_png(qc_dir / "summary.png", log, args.margin)

    error_cases = [r["case"] for r in log if r["status"].startswith("error")]
    failed_cases = [r["case"] for r in log if r["status"].startswith("not detected")]
    lost_cases = [r["case"] for r in log if r.get("lost_label_voxels")]
    logger.info(f"Done in {(time.time() - start_time) / 60:.1f} min: {len(log)} cases, {len(error_cases)} errors "
                f"(not in the output), {len(failed_cases)} not detected, {len(lost_cases)} with label voxels cut "
                f"off. Details in {args.output / 'crop_log.csv'}, figures in {qc_dir}")
    if error_cases:
        logger.info("  Errors: " + ", ".join(error_cases))
    if failed_cases:
        logger.info("  Not detected: " + ", ".join(failed_cases))
    if lost_cases:
        logger.info("  Label cut off: " + ", ".join(lost_cases))
    suggestion = suggest_margin(log, args.margin)
    if suggestion and lost_cases:
        logger.info(f"  Margin that keeps the whole label in 95% of the cases: --margin "
                    + " ".join(f"{m:g}" for m in suggestion))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("Crash")   # full traceback in the log file
        raise