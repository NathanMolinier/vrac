"""
Generate QC preview images (mid-sagittal) for every file listed in a dataset
config JSON, organized by split (TRAINING / VALIDATION / TESTING).

For every entry in a split, the IMAGE is loaded, reoriented to RSP, and the
middle sagittal slice is normalized using its 5th/95th percentile. One figure
is produced per entry, with one subplot per overlay key. Each overlay key is
either a segmentation (colored discrete labels drawn semi-transparently, with
each value annotated at its centroid) or a point-wise label volume (each
labeled voxel is projected onto the sagittal plane and drawn as a Gaussian
blob with its value annotated next to it).

Example
-------
    vrac_qc_config \\
        --config /path/to/whole-spine.json \\
        --output /path/to/qc_folder \\
        --keys LABEL:label SEG_SPINE:seg SEG_CORD:seg SEG_CANAL:seg
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from vrac.data_management.image import Image


SPLITS_DEFAULT = ("TRAINING", "VALIDATION", "TESTING")


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2D mid-sagittal QC preview images for every entry of a "
            "dataset config, with one subplot per overlay key."
        )
    )
    parser.add_argument("--config", "-c", required=True, type=str,
                        help="Path to the dataset config JSON (with TRAINING / VALIDATION / TESTING).")
    parser.add_argument("--output", "-o", required=True, type=str,
                        help="Output folder where QC images will be written.")
    parser.add_argument("--keys", "-k", nargs="+", required=True,
                        help=("Overlay keys with their type as KEY:TYPE pairs. "
                              "TYPE must be 'seg' (segmentation) or 'label' (point-wise labels). "
                              "Example: LABEL:label SEG_SPINE:seg SEG_CORD:seg SEG_CANAL:seg"))
    parser.add_argument("--splits", "-s", nargs="+", default=list(SPLITS_DEFAULT),
                        choices=list(SPLITS_DEFAULT),
                        help="Which splits to process. Default: all three.")
    parser.add_argument("--format", "-f", default="png", choices=["png", "jpg"],
                        help="Output image format. Default: png.")
    parser.add_argument("--jobs", "-j", default=0, type=int,
                        help="Number of worker processes. 0 (default) uses all available CPUs.")
    parser.add_argument("--sigma", default=2.5, type=float,
                        help="Standard deviation (in voxels) of the Gaussian blob used to draw point labels.")
    return parser


def _parse_keys(raw_keys):
    parsed = []
    for item in raw_keys:
        if ":" not in item:
            raise SystemExit(f"Invalid --keys entry '{item}'. Expected KEY:TYPE (e.g. SEG_SPINE:seg).")
        name, kind = item.split(":", 1)
        name = name.strip()
        kind = kind.strip().lower()
        if kind not in ("seg", "label"):
            raise SystemExit(f"Invalid overlay type '{kind}' for key '{name}'. Must be 'seg' or 'label'.")
        parsed.append((name, kind))
    return parsed


def _mid_sagittal_rsp(nii_path):
    img = Image(nii_path).change_orientation("RSP")
    data = img.data
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Unsupported ndim={data.ndim} for {nii_path}")
    mid = data.shape[0] // 2
    return np.asarray(data[mid, :, :]), data


def _normalize_image_slice(sl):
    sl = sl.astype(np.float32)
    lo = np.percentile(sl, 5.0)
    hi = np.percentile(sl, 95.0)
    if hi <= lo:
        hi = lo + 1.0
    sl = np.clip(sl, lo, hi)
    sl = (sl - lo) / (hi - lo)
    return sl


def _display_slice(sl):
    # Flip so Superior is up (RSP: axis S is axis 0 of the sagittal slice).
    return np.flipud(sl)


def _discrete_cmap(values):
    base = plt.get_cmap("tab20")
    colors = [base(i % 20) for i in range(len(values))]
    return ListedColormap(colors)


def _overlay_segmentation(ax, seg_slice_display, bg_shape):
    seg_slice_display = np.asarray(seg_slice_display)
    uniq = np.unique(seg_slice_display)
    uniq = uniq[uniq != 0]
    if uniq.size == 0:
        return

    cmap = _discrete_cmap(uniq)
    value_to_idx = {v: i for i, v in enumerate(uniq)}
    idx_slice = np.zeros_like(seg_slice_display, dtype=np.float32)
    for v, i in value_to_idx.items():
        idx_slice[seg_slice_display == v] = i + 1

    masked = np.ma.masked_where(idx_slice == 0, idx_slice - 1)
    ax.imshow(masked, cmap=cmap, alpha=0.45, interpolation="nearest",
              vmin=0, vmax=max(len(uniq) - 1, 1))

    for v in uniq:
        ys, xs = np.where(seg_slice_display == v)
        if ys.size == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        ax.text(cx + 2, cy, f"{int(v)}", color="white", fontsize=6,
                ha="left", va="center",
                bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"))


def _gaussian_blob(shape, cy, cx, sigma):
    ys, xs = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2)))


def _overlay_labels(ax, label_volume_rsp, sagittal_shape, sigma):
    """Project all non-zero voxels onto the sagittal plane and draw a Gaussian
    blob (+ value) at each (y, z) projection. ``label_volume_rsp`` is the full
    3D volume (RSP). The sagittal plane spans axes 1 (S) and 2 (P) of RSP.
    """
    nonzero = np.argwhere(label_volume_rsp > 0)
    if nonzero.size == 0:
        return

    # Group by (y, z) in volume coordinates, keep one representative value per
    # position (max, which is useful when a label occupies several voxels).
    yz_to_value = {}
    for x, y, z in nonzero:
        key = (int(y), int(z))
        v = label_volume_rsp[x, y, z]
        prev = yz_to_value.get(key)
        if prev is None or v > prev:
            yz_to_value[key] = v

    H, W = sagittal_shape  # display coords
    accum = np.zeros((H, W), dtype=np.float32)
    label_map = np.zeros((H, W), dtype=np.float32)

    for (y, z), value in yz_to_value.items():
        # Display slice is flipud of (S, P) -> row index = (H - 1 - y), col = z
        cy_disp = (H - 1) - y
        cx_disp = z
        blob = _gaussian_blob((H, W), cy_disp, cx_disp, sigma)
        accum = np.maximum(accum, blob)
        label_map[blob > 0.5] = value

        ax.text(cx_disp + sigma + 1, cy_disp, f"{int(value)}",
                color="yellow", fontsize=6, ha="left", va="center",
                bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"))

    masked = np.ma.masked_where(accum < 0.05, accum)
    ax.imshow(masked, cmap="autumn", alpha=0.7, interpolation="bilinear",
              vmin=0.0, vmax=1.0)


def _resolve(path, datasets_root):
    if os.path.isabs(path):
        return path
    return os.path.join(datasets_root, path)


def _process_entry(entry, keys, datasets_root, split_out_dir, ext, sigma):
    image_rel = entry.get("IMAGE")
    if not image_rel:
        return None, "missing IMAGE"
    image_path = _resolve(image_rel, datasets_root)

    try:
        img_slice, _ = _mid_sagittal_rsp(image_path)
    except Exception as exc:
        return image_rel, f"image load failed: {exc!r}"

    bg = _display_slice(_normalize_image_slice(img_slice))
    H, W = bg.shape

    n = len(keys)
    fig_width = max(4.0 * n, 4.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 6.0), squeeze=False)
    axes = axes[0]

    base_name = os.path.splitext(os.path.splitext(os.path.basename(image_rel))[0])[0]
    fig.suptitle(base_name, fontsize=10)

    for ax, (key, kind) in zip(axes, keys):
        ax.imshow(bg, cmap="gray", interpolation="bilinear", vmin=0.0, vmax=1.0)
        ax.set_title(f"{key} ({kind})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        overlay_rel = entry.get(key)
        if not overlay_rel:
            ax.text(0.5, 0.5, f"no {key}", color="red", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            continue
        overlay_path = _resolve(overlay_rel, datasets_root)

        try:
            if kind == "seg":
                seg_slice, _ = _mid_sagittal_rsp(overlay_path)
                _overlay_segmentation(ax, _display_slice(seg_slice), bg.shape)
            else:  # 'label'
                full = Image(overlay_path).change_orientation("RSP").data
                if full.ndim == 4:
                    full = full[..., 0]
                _overlay_labels(ax, full, bg.shape, sigma)
        except Exception as exc:
            ax.text(0.5, 0.05, f"{key} failed: {exc!r}", color="red",
                    ha="center", va="bottom", transform=ax.transAxes, fontsize=6)

    out_name = base_name + "." + ext
    out_path = os.path.join(split_out_dir, out_name)

    # Avoid name collisions by prefixing with a hash of the relative image path.
    if os.path.exists(out_path):
        suffix = str(abs(hash(image_rel)) % 10_000_000)
        out_path = os.path.join(split_out_dir, f"{base_name}_{suffix}.{ext}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return image_rel, None


def main():
    args = get_parser().parse_args()

    config_path = os.path.abspath(args.config)
    output_root = os.path.abspath(args.output)
    ext = args.format
    sigma = args.sigma
    keys = _parse_keys(args.keys)

    with open(config_path, "r") as f:
        config = json.load(f)

    datasets_root = config.get("DATASETS_PATH", "")
    if datasets_root and not os.path.isabs(datasets_root):
        datasets_root = os.path.abspath(datasets_root)

    os.makedirs(output_root, exist_ok=True)

    workers = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    jobs = []
    for split in args.splits:
        entries = config.get(split, [])
        if not entries:
            print(f"[{split}] no entries, skipping.")
            continue
        split_out_dir = os.path.join(output_root, split)
        os.makedirs(split_out_dir, exist_ok=True)
        print(f"[{split}] {len(entries)} entries -> {split_out_dir}")
        for entry in entries:
            jobs.append((entry, split, split_out_dir))

    if not jobs:
        print("Nothing to do.")
        return

    print(f"Running on {workers} worker(s).")
    done = 0
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_entry, entry, keys, datasets_root,
                        split_out_dir, ext, sigma): (entry, split)
            for entry, split, split_out_dir in jobs
        }
        for fut in as_completed(futures):
            entry, split = futures[fut]
            try:
                path, err = fut.result()
            except Exception as exc:
                path, err = entry.get("IMAGE", "<unknown>"), repr(exc)
            done += 1
            if err is not None:
                failed.append((split, path, err))
            if done % 25 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] processed")

    if failed:
        print(f"{len(failed)} entry/entries failed:")
        for split, path, err in failed[:20]:
            print(f"  [{split}] {path}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
