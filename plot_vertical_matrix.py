#!/usr/bin/env python3
"""Plot a vertical 3×7 matrix (7 rows × 3 cols) as colored squares.

Examples:
  python plot_vertical_matrix.py
  python plot_vertical_matrix.py --cmap viridis --annotate
  python plot_vertical_matrix.py --values "0,1,2;3,4,5;6,7,8;9,10,11;12,13,14;15,16,17;18,19,20" --save out.png

Notes:
- Default shape is 7x3 ("vertical" 3×7). Use --rows/--cols to change.
- Provide --values as semicolon-separated rows, comma-separated columns.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_values(values: str) -> np.ndarray:
    rows: List[List[float]] = []
    for row_str in values.strip().split(";"):
        row_str = row_str.strip()
        if not row_str:
            continue
        row = [float(x.strip()) for x in row_str.split(",") if x.strip()]
        rows.append(row)

    if not rows:
        raise ValueError("--values is empty")

    ncols = {len(r) for r in rows}
    if len(ncols) != 1:
        raise ValueError("All rows in --values must have the same number of columns")

    return np.array(rows, dtype=float)


def make_matrix(rows: int, cols: int, rng_seed: Optional[int], mode: str) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)

    if mode == "random_int":
        return rng.integers(low=0, high=100, size=(rows, cols))
    if mode == "random_float":
        return rng.random(size=(rows, cols))

    raise ValueError(f"Unknown mode: {mode}")


def plot_matrix(
    data: np.ndarray,
    *,
    cmap: str,
    title: Optional[str],
    annotate: bool,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4.0, 8.0))

    im = ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    # Grid lines to emphasize squares
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Hide axis values (ticks/labels) while keeping the grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value")

    if annotate:
        # Choose text color based on cell intensity for readability
        norm = im.norm
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                color = "black" if norm(val) > 0.6 else "white"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    return fig, ax


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot a vertical 3×7 matrix with a colormap.")
    p.add_argument("--rows", type=int, default=7, help="Number of rows (default: 7)")
    p.add_argument("--cols", type=int, default=3, help="Number of columns (default: 3)")
    p.add_argument(
        "--values",
        type=str,
        default=None,
        help='Explicit values as "row;row;..." with commas in each row, e.g. "1,2,3;4,5,6"',
    )
    p.add_argument(
        "--mode",
        choices=["random_int", "random_float"],
        default="random_int",
        help="How to generate values when --values is not provided",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible random values")
    p.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap name")
    p.add_argument("--title", type=str, default="3×7 Vertical Matrix", help="Plot title")
    p.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the numeric value in each square (default: enabled)",
    )
    p.add_argument("--vmin", type=float, default=None, help="Minimum value for colormap normalization")
    p.add_argument("--vmax", type=float, default=None, help="Maximum value for colormap normalization")
    p.add_argument("--save", type=str, default=None, help="Save figure to this path (png/pdf/svg)")
    p.add_argument("--show", action="store_true", help="Show interactively (default if --save not set)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    # data = np.array([
    #     [0.082, 0.899, 0.753, 0.029, 0.703, 0.462, 0.105],
    #     [0.122, 0.901, 0.758, 0.061, 0.704, 0.559, 0.171],
    #     [0.767, 0.901, 0.887, 0.593, 0.610, 0.607, 0.514]
    # ]) # Spider
    # data = np.array([
    #     [0.148, 0.761, 0.827, 0.160, 0.768, 0.900, 0.794],
    #     [0.055, 0.774, 0.825, 0.085, 0.802, 0.898, 0.799],
    #     [0.831, 0.817, 0.827, 0.788, 0.857, 0.891, 0.847]
    # ]) # spinegan dixon inphase
    data = np.array([
        [0.905, 0.091, 0.041, 0.885, 0.355, 0.131, 0.039],
        [0.910, 0.244, 0.136, 0.884, 0.466, 0.144, 0.094],
        [0.897, 0.788, 0.777, 0.881, 0.746, 0.761, 0.680]
    ]) # spinegan ct

    fig, _ = plot_matrix(
        data.transpose(),
        cmap="autumn",
        title=args.title,
        annotate=args.annotate,
        vmin=0,
        vmax=1,
    )

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
