"""Plot per-modality comparisons between methods as boxplots.

Expects a per-case CSV (long format, one row per case/model/modality/label)
with columns:
    model, modality, label, <metric_1>, <metric_2>, ...
Any other numeric column is treated as a metric unless ``--metrics`` is given.

For each metric and each modality a grouped boxplot is written, with one group
per label and one box per selected method. When there are many labels (e.g.
TotalSegmentator's 117 classes) the labels are split across several stacked
rows so the figure stays roughly square.

Next to each plot, a table image (``<metric>_<modality>_table.png``) gives the
boxplot values per label and method as ``median [Q1, Q3]``, and a CSV
(``<metric>_<modality>_table.csv``) holds the full box statistics (n, median,
quartiles, whisker ends, mean, std).

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
from matplotlib.transforms import Bbox


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


def collect_values(
    df: pd.DataFrame,
    metric: str,
    modality: str,
    methods: dict[str, str],
):
    """Return (models, labels, {(model, label): finite values}) or None if empty."""
    sub = df[(df["modality"] == modality) & df["model"].isin(methods)]
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]
    if sub.empty:
        return None
    models = [m for m in methods if m in set(sub["model"])]
    labels = sorted(sub["label"].unique())
    values = {
        (m, lab): grp[metric].to_numpy(dtype=float)
        for (m, lab), grp in sub.groupby(["model", "label"])
    }
    return models, labels, values


def box_stats_table(models, labels, values, methods) -> pd.DataFrame:
    """Long table of the statistics drawn by the boxplot (Tukey whiskers, 1.5 IQR)."""
    rows = []
    for lab in labels:
        for m in models:
            vals = values.get((m, lab))
            if vals is None or not vals.size:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            in_range = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
            rows.append(dict(
                label=lab, method=methods[m], n=vals.size,
                median=med, q1=q1, q3=q3,
                whisker_low=in_range.min(), whisker_high=in_range.max(),
                mean=vals.mean(), std=vals.std(ddof=1) if vals.size > 1 else np.nan,
            ))
    return pd.DataFrame(rows)


def save_table_image(
    stats: pd.DataFrame,
    models: list[str],
    labels: list[str],
    methods: dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    """Render a label x method table with 'median [Q1, Q3]' cells as an image."""
    names = [methods[m] for m in models]
    span = np.nanmax(np.abs(stats[["median", "q1", "q3"]].to_numpy(dtype=float)))
    fmt = "{:.3f}" if span < 10 else "{:.1f}"
    lookup = stats.set_index(["label", "method"])
    cells = []
    for lab in labels:
        row = []
        for name in names:
            if (lab, name) in lookup.index:
                r = lookup.loc[(lab, name)]
                row.append(f"{fmt.format(r['median'])} [{fmt.format(r['q1'])}, {fmt.format(r['q3'])}]")
            else:
                row.append("—")
        cells.append(row)

    colors = model_colors(models)
    label_w = max(len(str(lab)) for lab in labels) * 0.09 + 0.4
    col_w = max(1.9, max(len(c) for row in cells for c in row) * 0.085)
    fig_w = label_w + col_w * len(names)
    fig_h = 0.3 * (len(labels) + 1) + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        rowLabels=labels,
        colLabels=names,
        colColours=[colors.get(m) for m in models],
        colWidths=[col_w / fig_w] * len(names),
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_alpha(0.85)
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    ax.set_title(f"{title}  —  median [Q1, Q3]", fontsize=10, pad=8)
    # Crop to the table + title only (the hidden axes would otherwise add blank space).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([table.get_window_extent(renderer), ax.title.get_window_extent(renderer)])
    bbox = bbox.padded(10).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out_path, dpi=200, bbox_inches=bbox)
    plt.close(fig)


def plot_metric_modality(
    models: list[str],
    labels: list[str],
    values: dict,
    metric: str,
    modality: str,
    methods: dict[str, str],
    out_path: Path,
) -> None:
    colors = model_colors(models)

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
            collected = collect_values(df, metric, modality, methods)
            if collected is None:
                continue
            models, labels, values = collected

            plot_path = out_dir / f"{metric}_{modality}.png"
            plot_metric_modality(models, labels, values, metric, modality, methods, plot_path)

            stats = box_stats_table(models, labels, values, methods)
            csv_out = out_dir / f"{metric}_{modality}_table.csv"
            stats.to_csv(csv_out, index=False, float_format="%.4f")
            table_path = out_dir / f"{metric}_{modality}_table.png"
            save_table_image(stats, models, labels, methods, f"{metric} — {modality}", table_path)
            print(f"wrote {plot_path}, {table_path}, {csv_out}")


if __name__ == "__main__":
    main()
