"""Plot per-modality comparisons between methods as boxplots.

Expects a per-case CSV (long format, one row per case/model/modality/label)
with columns:
    model, modality, label, <metric_1>, <metric_2>, ...
Any other numeric column is treated as a metric unless ``--metrics`` is given.

For each metric and each modality a grouped boxplot is written, with one group
per label and one box per selected method. When there are many labels (e.g.
TotalSegmentator's 117 classes) the labels are split across several stacked
rows so the figure stays roughly square.

Methods are selected (and renamed) with ``--methods MODEL[:DISPLAY_NAME] ...``,
e.g. ``--methods nnUNetTrainer:baseline synthseg:SynthSeg``. Methods are
plotted in the order given. Without ``--methods`` every model is plotted under
its original name.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PER_CASE_FILE = "comparison_long.csv"
ID_COLUMNS = ("model", "modality", "label")
BOXES_PER_ROW_TARGET = 50  # labels per row shrink as more methods are compared
FIXED_COLORS = (
    ("synthseg_noflip", "#1a6b1a"),  # dark green; must precede "synthseg"
    ("synthseg", "tab:green"),
    ("ossta_noflip", "#b35900"),  # dark orange; must precede "ossta" (first substring match wins)
    ("ossta", "tab:orange"),
)
FALLBACK_COLORS = ["tab:blue", "tab:red", "tab:purple", "tab:brown",
                   "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def model_colors(models: list[str]) -> dict[str, str]:
    """SynthSeg -> green (noFlip darker), OSSTA -> orange (noFlip darker), remaining models -> blue, then others."""
    colors = {}
    remaining = []
    for m in models:
        name = m.lower()
        color = next((c for key, c in FIXED_COLORS if key in name), None)
        if color is None:
            remaining.append(m)
        else:
            colors[m] = color
    for m, c in zip(remaining, FALLBACK_COLORS):
        colors[m] = c
    return colors


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "folder",
        type=Path,
        help=f"Folder containing the per-case CSV (default name: {PER_CASE_FILE})",
    )
    parser.add_argument(
        "--csv",
        default=PER_CASE_FILE,
        help=f"Per-case CSV file name inside folder, or a full path (default: {PER_CASE_FILE})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="MODEL[:NAME]",
        help="Methods to plot, in order, optionally renamed: "
             "e.g. nnUNetTrainer:baseline synthseg:SynthSeg",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric columns to plot (default: every numeric non-id column)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder for plots (default: <folder>/plots)",
    )
    return parser.parse_args()


def parse_methods(specs: list[str] | None, available: list[str]) -> dict[str, str]:
    """Return an ordered {model_name: display_name} mapping."""
    if specs is None:
        return {m: m for m in sorted(available)}
    methods = {}
    for spec in specs:
        model, _, display = spec.partition(":")
        methods[model] = display or model
    missing = [m for m in methods if m not in available]
    if missing:
        raise ValueError(
            f"Unknown method(s) {missing}. Available: {sorted(available)}"
        )
    return methods


def discover_metrics(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c not in ID_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    )


def choose_nrows(n_labels: int, n_models: int) -> int:
    """Pick a row count that keeps the figure readable (~BOXES_PER_ROW_TARGET boxes per row)."""
    labels_per_row = max(4, BOXES_PER_ROW_TARGET // max(n_models, 1))
    return max(1, math.ceil(n_labels / labels_per_row))


def plot_metric_modality(
    df: pd.DataFrame,
    metric: str,
    modality: str,
    methods: dict[str, str],
    out_path: Path,
) -> bool:
    sub = df[(df["modality"] == modality) & df["model"].isin(methods)]
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]
    if sub.empty:
        return False

    models = [m for m in methods if m in set(sub["model"])]
    labels = sorted(sub["label"].unique())
    colors = model_colors(models)
    values = {
        (m, lab): grp[metric].to_numpy(dtype=float)
        for (m, lab), grp in sub.groupby(["model", "label"])
    }

    nrows = choose_nrows(len(labels), len(models))
    per_row = math.ceil(len(labels) / nrows)
    width = 0.8 / max(len(models), 1)

    row_width = max(8, 0.22 * per_row * max(len(models), 2) + 2)
    row_height = 4.0
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(row_width, row_height * nrows + 0.8),
        squeeze=False,
        sharey=True,
    )

    for r in range(nrows):
        ax = axes[r, 0]
        chunk = labels[r * per_row:(r + 1) * per_row]
        x = np.arange(len(chunk))
        for i, model in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * width
            data, positions = [], []
            for xi, lab in zip(x, chunk):
                vals = values.get((model, lab))
                if vals is not None and vals.size:
                    data.append(vals)
                    positions.append(xi + offset)
            if not data:
                continue
            ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=colors.get(model), alpha=0.8, linewidth=0.8),
                medianprops=dict(color="black", linewidth=1),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=2, alpha=0.5),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(chunk, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_xlim(-0.5, per_row - 0.5)

    handles = [Patch(facecolor=colors.get(m), alpha=0.8, label=methods[m]) for m in models]
    axes[0, 0].set_title(f"{metric} — {modality}", pad=24)
    axes[0, 0].legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles), fontsize="small", frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute() and not csv_path.is_file():
        csv_path = args.folder / args.csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing per-case CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in ID_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} is missing column(s) {missing_cols}")

    out_dir = args.out if args.out is not None else args.folder / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_methods(args.methods, list(df["model"].unique()))
    metrics = args.metrics if args.metrics is not None else discover_metrics(df)
    modalities = sorted(df["modality"].unique())

    for metric in metrics:
        for modality in modalities:
            out_path = out_dir / f"{metric}_{modality}.png"
            if plot_metric_modality(df, metric, modality, methods, out_path):
                print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
