"""Plot per-modality comparisons between methods from a metrics folder.

Expects a folder containing ``comparison_summary_by_modality.csv`` with columns:
    model, modality, label,
    <metric>_mean, <metric>_std, <metric>_count, ...

For each metric and each modality a grouped bar chart is written next to the
CSV, with one group per label and one bar per model (std shown as error bars).
When there are many labels (e.g. TotalSegmentator's 117 classes) the labels
are split across several stacked rows so the figure stays roughly square.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_FILE = "comparison_summary_by_modality.csv"
LABELS_PER_ROW_TARGET = 25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing comparison_summary_by_modality.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder for plots (default: <folder>/plots)",
    )
    return parser.parse_args()


def discover_metrics(df: pd.DataFrame) -> list[str]:
    return sorted(
        c[: -len("_mean")] for c in df.columns if c.endswith("_mean")
    )


def choose_nrows(n_labels: int) -> int:
    """Pick a row count that keeps the figure roughly square."""
    if n_labels <= LABELS_PER_ROW_TARGET:
        return 1
    return max(1, math.ceil(n_labels / LABELS_PER_ROW_TARGET))


def plot_metric_modality(
    df: pd.DataFrame,
    metric: str,
    modality: str,
    out_path: Path,
) -> None:
    sub = df[df["modality"] == modality].copy()
    if sub.empty:
        return

    models = sorted(sub["model"].unique())
    labels = sorted(sub["label"].unique())

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    per_model = {
        m: sub[sub["model"] == m].set_index("label").reindex(labels)
        for m in models
    }

    nrows = choose_nrows(len(labels))
    per_row = math.ceil(len(labels) / nrows)
    width = 0.8 / max(len(models), 1)

    row_width = max(8, 0.45 * per_row + 2)
    row_height = 3.2
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(row_width, row_height * nrows + 0.8),
        squeeze=False,
    )

    y_vals = sub[mean_col].to_numpy(dtype=float)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size:
        y_lo, y_hi = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else max(abs(y_hi), 1.0))
        y_lim = (y_lo - pad, y_hi + pad)
    else:
        y_lim = None

    for r in range(nrows):
        ax = axes[r, 0]
        start = r * per_row
        end = min(start + per_row, len(labels))
        chunk = labels[start:end]
        x = np.arange(len(chunk))
        for i, model in enumerate(models):
            row = per_model[model].loc[chunk]
            means = row[mean_col].to_numpy(dtype=float)
            stds = row[std_col].to_numpy(dtype=float)
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=np.nan_to_num(stds),
                capsize=3,
                label=model if r == 0 else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(chunk, rotation=45, ha="right")
        ax.set_ylabel(metric)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_xlim(-0.5, per_row - 0.5)

    axes[0, 0].set_title(f"{metric} — {modality}")
    axes[0, 0].legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.folder / SUMMARY_FILE
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {SUMMARY_FILE} in {args.folder}")

    df = pd.read_csv(csv_path)
    out_dir = args.out if args.out is not None else args.folder / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = discover_metrics(df)
    modalities = sorted(df["modality"].unique())

    for metric in metrics:
        for modality in modalities:
            out_path = out_dir / f"{metric}_{modality}.png"
            plot_metric_modality(df, metric, modality, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
