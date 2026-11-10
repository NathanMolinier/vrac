"""Sagittal "flattened" previews of a spine MRI.

The image is preprocessed like the discslabel training inputs (reoriented to RPI,
SCT "from" convention, then resampled to 1 mm isotropic with linear interpolation),
then four sagittal views are shown side by side:

  1. mean over a slab of ``--slab-mm`` (default 10 mm) centred on the middle R-L slice
  2. the middle sagittal slice (``shape[0] // 2``, same as discslabel_inference)
  3. the standard deviation over the same ``--slab-mm`` slab
  4. the standard deviation along the whole R-L axis

Display: superior up, anterior left (same as the evaluation previews).

Usage:
    python flatten_image.py <image.nii.gz> [-o preview.png] [--slab-mm 10]

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


def sagittal_views(data: np.ndarray, slab_mm: float, resolution_mm: float = 1.0) -> dict:
    """Return the three (P-A, I-S) sagittal views described in the module docstring."""
    n_rl = data.shape[0]
    mid = n_rl // 2
    half = max(1, int(round(slab_mm / resolution_mm))) / 2.0
    lo = max(0, int(np.floor(mid - half + 0.5)))
    hi = min(n_rl, int(np.floor(mid + half + 0.5)))
    slab = data[lo:hi]
    return {
        f"mean over {hi - lo} mm (slices {lo}-{hi - 1})": slab.mean(axis=0),
        f"middle slice ({mid}/{n_rl})": data[mid],
        f"std over {hi - lo} mm (slices {lo}-{hi - 1})": slab.std(axis=0),
        f"std along R-L ({n_rl} slices)": data.std(axis=0),
    }


def to_display(view: np.ndarray) -> np.ndarray:
    """(P-A, I-S) -> rows = superior to inferior, cols = anterior to posterior."""
    return np.ascontiguousarray(view.T[::-1, ::-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Input NIfTI image.")
    parser.add_argument("-o", "--out", default=None, help="Save the figure to this PNG instead of showing it.")
    parser.add_argument("--slab-mm", type=float, default=10.0, help="Thickness of the averaged slab (default: 10).")
    parser.add_argument("--orientation", default="RPI", help="Orientation, SCT convention (default: RPI).")
    parser.add_argument("--resolution", type=float, default=1.0, help="Isotropic resolution in mm (default: 1).")
    args = parser.parse_args()

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = preprocess(args.image, args.orientation, args.resolution)
    views = sagittal_views(data, args.slab_mm, args.resolution)

    fig, axes = plt.subplots(1, len(views), figsize=(6 * len(views), 8))
    for ax, (title, view) in zip(axes, views.items()):
        disp = to_display(view)
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
