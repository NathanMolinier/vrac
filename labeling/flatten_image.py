"""Sagittal and coronal "flattened" previews of a spine MRI.

The image is preprocessed like the discslabel training inputs (reoriented to RPI,
SCT "from" convention, then resampled to 1 mm isotropic with linear interpolation),
then four sagittal views (top row) and four coronal views (bottom row) are shown:

  1. mean over a slab of ``--slab-mm`` (default 10 mm) centred on the middle slice
  2. the middle slice (sagittal: ``shape[0] // 2``, same as discslabel_inference)
  3. the standard deviation over the same ``--slab-mm`` slab
  4. the standard deviation along the whole projection axis

Sagittal views project along R-L, coronal views along P-A.
Display: superior up; sagittal anterior left; coronal patient right on the left
(radiological convention).

Usage:
    python flatten_image.py <image.nii.gz> [-o preview.png] [--slab-mm 10] [--coronal-slab-mm 10]

Needs the ``discslabel`` package (discs-labeling-multicontrast) to be installed.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from discslabel.utils.image import Image, resample_nib


def preprocess(path: str, orientation: str = "RPI", resolution_mm: float = 1.0) -> np.ndarray:
    """Load, reorient and resample; return the (R-L, P-A, I-S) float volume."""
    image = Image(path)
    image.change_orientation(orientation)
    image = resample_nib(
        image,
        new_size=[resolution_mm] * 3,
        new_size_type="mm",
        interpolation="linear",
        verbose=False,
    )
    data = np.asarray(image.data, dtype=np.float32)
    while data.ndim > 3:  # 4D: keep the first volume
        data = data[..., 0]
    return data


def flattened_views(data: np.ndarray, axis: int, slab_mm: float, resolution_mm: float = 1.0) -> dict:
    """Return the four 2D views (mean slab, middle slice, std slab, full std) projected along ``axis``."""
    vol = np.moveaxis(data, axis, 0)
    n = vol.shape[0]
    mid = n // 2
    half = max(1, int(round(slab_mm / resolution_mm))) / 2.0
    lo = max(0, int(np.floor(mid - half + 0.5)))
    hi = min(n, int(np.floor(mid + half + 0.5)))
    slab = vol[lo:hi]
    name = {0: "R-L", 1: "P-A", 2: "I-S"}[axis]
    return {
        f"mean over {hi - lo} mm (slices {lo}-{hi - 1})": slab.mean(axis=0),
        f"middle slice ({mid}/{n})": vol[mid],
        f"std over {hi - lo} mm (slices {lo}-{hi - 1})": slab.std(axis=0),
        f"std along {name} ({n} slices)": vol.std(axis=0),
    }


def sagittal_views(data: np.ndarray, slab_mm: float, resolution_mm: float = 1.0) -> dict:
    """(P-A, I-S) views projected along R-L."""
    return flattened_views(data, 0, slab_mm, resolution_mm)


def coronal_views(data: np.ndarray, slab_mm: float, resolution_mm: float = 1.0) -> dict:
    """(R-L, I-S) views projected along P-A."""
    return flattened_views(data, 1, slab_mm, resolution_mm)


def to_display(view: np.ndarray) -> np.ndarray:
    """Sagittal (P-A, I-S) -> rows = superior to inferior, cols = anterior to posterior."""
    return np.ascontiguousarray(view.T[::-1, ::-1])


def to_display_coronal(view: np.ndarray) -> np.ndarray:
    """Coronal (R-L, I-S) -> rows = superior to inferior, cols = patient right to left."""
    return np.ascontiguousarray(view.T[::-1, :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Input NIfTI image.")
    parser.add_argument("-o", "--out", default=None, help="Save the figure to this PNG instead of showing it.")
    parser.add_argument("--slab-mm", type=float, default=10.0, help="Thickness of the averaged slab (default: 10).")
    parser.add_argument(
        "--coronal-slab-mm", type=float, default=None, help="Thickness of the coronal averaged slab (default: --slab-mm)."
    )
    parser.add_argument("--orientation", default="RPI", help="Orientation, SCT convention (default: RPI).")
    parser.add_argument("--resolution", type=float, default=1.0, help="Isotropic resolution in mm (default: 1).")
    args = parser.parse_args()

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = preprocess(args.image, args.orientation, args.resolution)
    cor_slab = args.slab_mm if args.coronal_slab_mm is None else args.coronal_slab_mm
    rows = [
        ("sagittal", sagittal_views(data, args.slab_mm, args.resolution), to_display),
        ("coronal", coronal_views(data, cor_slab, args.resolution), to_display_coronal),
    ]

    n_cols = max(len(v) for _, v, _ in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(6 * n_cols, 8 * len(rows)), squeeze=False)
    panels = [
        (axes[r, c], f"{plane} — {title}", view, disp_fn)
        for r, (plane, views, disp_fn) in enumerate(rows)
        for c, (title, view) in enumerate(views.items())
    ]
    for ax, title, view, disp_fn in panels:
        disp = disp_fn(view)
        vmin, vmax = np.percentile(disp, [1.0, 99.0])
        if vmax <= vmin:
            vmin, vmax = float(disp.min()), float(disp.max()) + 1e-6
        ax.imshow(disp, cmap="gray", vmin=vmin, vmax=vmax)  # isotropic -> aspect 1
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    name = os.path.basename(args.image)
    fig.suptitle(f"{name}  —  {args.orientation}, {args.resolution:g} mm iso, shape {data.shape}", fontsize=12)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=100, bbox_inches="tight")
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
