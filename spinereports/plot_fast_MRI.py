#!/usr/bin/env python3
"""Plot fast-MRI metric trends across measurement time points.

Expected input layout (same subject, multiple measurements):

root/
  <subject>_meas001/
    files/
      canal_subject.csv
      csf_subject.csv
      discs_subject.csv
      foramens_subject.csv
      vertebrae_subject.csv
  <subject>_meas002/
    files/
      ...

For each structure CSV found in `files/`, this script creates line plots of numeric
metrics over measure points and writes them under `--outdir`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
	value = re.sub(r"\s+", "_", str(value).strip())
	value = re.sub(r"[^A-Za-z0-9_.\-]+", "", value)
	value = re.sub(r"_+", "_", value)
	return value.strip("_")


def _extract_measure_point(folder_name: str) -> Optional[int]:
	match = re.search(r"meas(\d+)|(\d+)meas", folder_name, flags=re.IGNORECASE)
	if not match:
		return None
	digits = match.group(1) or match.group(2)
	try:
		return int(digits)
	except ValueError:
		return None


def _find_measure_dirs(root_dir: Path) -> List[Tuple[int, Path]]:
	dirs: List[Tuple[int, Path]] = []
	for child in root_dir.iterdir():
		if not child.is_dir():
			continue
		m = _extract_measure_point(child.name)
		if m is None:
			continue
		if not (child / "files").exists():
			continue
		dirs.append((m, child))
	return sorted(dirs, key=lambda x: x[0])


def _numeric_cols(df: pd.DataFrame) -> List[str]:
	excluded = {"structure", "slice_interp", "measure_point"}
	return [
		c for c in df.columns
		if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
	]


def _choose_group_col(df: pd.DataFrame) -> Optional[str]:
	for candidate in ("structure_name", "vertebra_level"):
		if candidate in df.columns:
			n_unique = df[candidate].dropna().nunique()
			if n_unique > 1:
				return candidate
	return None


def _collect_structure_data(
	measure_dirs: Sequence[Tuple[int, Path]],
	structure_csv_name: str,
) -> pd.DataFrame:
	rows: List[pd.DataFrame] = []
	for measure_point, measure_dir in measure_dirs:
		csv_path = measure_dir / "files" / structure_csv_name
		if not csv_path.exists():
			continue
		df = pd.read_csv(csv_path)
		df["measure_point"] = measure_point
		rows.append(df)
	if not rows:
		return pd.DataFrame()
	return pd.concat(rows, ignore_index=True)


def _is_slice_data(df: pd.DataFrame) -> bool:
	return (
		"slice_interp" in df.columns
		and df["slice_interp"].dropna().nunique() > 1
	)


def _save_fig(fig, outdir: Path, structure_name: str, metric: str, suffix: str = "") -> None:
	structure_stem = _safe_name(structure_name.replace("_subject.csv", ""))
	metric_stem = _safe_name(metric)
	extra = f"__{_safe_name(suffix)}" if suffix else ""
	fig_path = outdir / f"{structure_stem}__{metric_stem}{extra}.png"
	fig.tight_layout()
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)


def _apply_measure_xaxis(ax, measure_points: List[int]) -> None:
	if measure_points:
		ax.set_xticks(measure_points)
		ax.set_xticklabels([str(int(m)) for m in measure_points])


def _plot_slice_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
) -> int:
	"""x=slice_interp, one line per measure_point, vertebra_level as vertical dividers.

	If `structure_name` has multiple values (e.g. canal vs spinalcord), emit one
	figure per value so magnitudes stay comparable.
	"""
	struct_col = None
	if "structure_name" in df.columns and df["structure_name"].astype(str).nunique() > 1:
		struct_col = "structure_name"

	if struct_col is None:
		partitions: List[Tuple[Optional[str], pd.DataFrame]] = [(None, df)]
	else:
		partitions = [
			(str(name), sub)
			for name, sub in df.groupby(struct_col, sort=True)
		]

	measure_points = sorted(df["measure_point"].dropna().unique().tolist())
	count = 0

	for label, sub in partitions:
		if sub.empty:
			continue

		slices = sorted(sub["slice_interp"].dropna().unique().tolist())
		if not slices:
			continue

		# Map each slice to its vertebra_level (mode across scans, since it is
		# supposed to be aligned).
		vlevel_col = "vertebra_level" if "vertebra_level" in sub.columns else None
		slice_to_vlevel: Dict[float, Optional[str]] = {}
		if vlevel_col is not None:
			for sl in slices:
				vals = sub.loc[sub["slice_interp"] == sl, vlevel_col].dropna()
				slice_to_vlevel[sl] = str(vals.mode().iloc[0]) if not vals.empty else None

		# Color one curve per measure_point using a sequential map so time reads.
		cmap = plt.get_cmap("viridis", max(len(measure_points), 1))
		mp_to_color = {mp: cmap(i) for i, mp in enumerate(measure_points)}

		fig, ax = plt.subplots(figsize=(10.0, 5.0))
		for mp in measure_points:
			row = sub[sub["measure_point"] == mp].sort_values("slice_interp")
			if row.empty:
				continue
			ax.plot(
				row["slice_interp"],
				row[metric],
				marker=".",
				markersize=3,
				linewidth=1.0,
				alpha=0.85,
				color=mp_to_color[mp],
				label=str(int(mp)),
			)

		# Vertebra_level regions: draw a vertical divider at each boundary and
		# label each region at the top of the plot.
		if vlevel_col is not None:
			boundaries: List[Tuple[float, float, str]] = []  # (start_slice, end_slice, level)
			run_level: Optional[str] = None
			run_start: Optional[float] = None
			prev_slice: Optional[float] = None
			for sl in slices:
				v = slice_to_vlevel.get(sl)
				if v != run_level:
					if run_level is not None and run_start is not None and prev_slice is not None:
						boundaries.append((run_start, prev_slice, run_level))
					run_level = v
					run_start = sl
				prev_slice = sl
			if run_level is not None and run_start is not None and prev_slice is not None:
				boundaries.append((run_start, prev_slice, run_level))

			for i, (start, _end, _lvl) in enumerate(boundaries):
				if i == 0:
					continue
				ax.axvline(start - 0.5, color="0.4", linewidth=0.8, linestyle="--", alpha=0.7)

			ymin, ymax = ax.get_ylim()
			y_text = ymax - (ymax - ymin) * 0.03
			for start, end, lvl in boundaries:
				if lvl is None:
					continue
				ax.text(
					(start + end) / 2.0,
					y_text,
					str(lvl),
					ha="center",
					va="top",
					fontsize=8,
					color="0.25",
				)

		ax.legend(
			loc="center left",
			bbox_to_anchor=(1.01, 0.5),
			fontsize=7,
			frameon=False,
			ncol=1,
			title="Measure",
		)

		title = f"{structure_name} - {metric}"
		if label is not None:
			title += f" ({label})"
		ax.set_title(title)
		ax.set_xlabel("Slice (slice_interp)")
		ax.set_ylabel(metric)
		ax.grid(True, alpha=0.25)

		_save_fig(fig, outdir, structure_name, metric, suffix=label or "")
		count += 1

	return count


def _plot_aggregate_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
	aggregate: str,
	max_groups: int,
) -> int:
	group_col = _choose_group_col(df)
	fig, ax = plt.subplots(figsize=(7.5, 4.2))

	measure_points = sorted(df["measure_point"].dropna().unique().tolist())

	if group_col is None:
		agg = df.groupby("measure_point", as_index=False)[metric].agg(aggregate)
		agg = agg.sort_values("measure_point")
		ax.plot(agg["measure_point"], agg[metric], marker="o", linewidth=1.8)
	else:
		group_counts = (
			df[group_col]
			.dropna()
			.astype(str)
			.value_counts()
			.head(max_groups)
			.index.tolist()
		)
		plot_df = df[df[group_col].astype(str).isin(group_counts)].copy()
		agg = (
			plot_df
			.groupby(["measure_point", group_col], as_index=False)[metric]
			.agg(aggregate)
			.sort_values([group_col, "measure_point"])
		)

		for group_name, sub in agg.groupby(group_col):
			ax.plot(
				sub["measure_point"],
				sub[metric],
				marker="o",
				linewidth=1.5,
				label=str(group_name),
			)
		ax.legend(loc="best", fontsize=8, frameon=False)

	ax.set_title(f"{structure_name} - {metric}")
	ax.set_xlabel("Measure point")
	ax.set_ylabel(metric)
	_apply_measure_xaxis(ax, measure_points)
	ax.grid(True, alpha=0.25)

	_save_fig(fig, outdir, structure_name, metric)
	return 1


def _plot_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
	aggregate: str,
	max_groups: int,
) -> int:
	if _is_slice_data(df):
		return _plot_slice_metric(
			df,
			structure_name=structure_name,
			metric=metric,
			outdir=outdir,
		)
	return _plot_aggregate_metric(
		df,
		structure_name=structure_name,
		metric=metric,
		outdir=outdir,
		aggregate=aggregate,
		max_groups=max_groups,
	)


def generate_plots(
	root_dir: Path,
	outdir: Path,
	*,
	aggregate: str,
	max_groups: int,
) -> Dict[str, int]:
	measure_dirs = _find_measure_dirs(root_dir)
	if not measure_dirs:
		raise SystemExit(
			f"No measurement directories found in {root_dir}. "
			"Expected folders containing 'meas###' and a 'files/' subfolder."
		)

	outdir.mkdir(parents=True, exist_ok=True)

	# Discover structure file names from available files folders.
	structure_files: set[str] = set()
	for _, meas_dir in measure_dirs:
		for csv_path in (meas_dir / "files").glob("*.csv"):
			if csv_path.name.endswith("_subject.csv"):
				structure_files.add(csv_path.name)

	if not structure_files:
		raise SystemExit("No '*_subject.csv' files found in measurement folders.")

	plot_count_by_structure: Dict[str, int] = {}
	for structure_csv_name in sorted(structure_files):
		structure_df = _collect_structure_data(measure_dirs, structure_csv_name)
		if structure_df.empty:
			continue

		num_cols = _numeric_cols(structure_df)
		if not num_cols:
			continue

		count = 0
		for metric in num_cols:
			count += _plot_metric(
				structure_df,
				structure_name=structure_csv_name,
				metric=metric,
				outdir=outdir,
				aggregate=aggregate,
				max_groups=max_groups,
			)

		plot_count_by_structure[structure_csv_name] = count

	return plot_count_by_structure


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Generate per-structure metric trend plots across fast-MRI measure points "
			"for one subject folder."
		)
	)
	parser.add_argument(
		"input_dir",
		type=Path,
		help="Folder containing measurement subfolders (e.g., reports_08/).",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=None,
		help="Output folder for PNG plots (default: <input_dir>/plots_fast_MRI).",
	)
	parser.add_argument(
		"--aggregate",
		choices=["mean", "median"],
		default="mean",
		help="Aggregation for rows within a measure point.",
	)
	parser.add_argument(
		"--max-groups",
		type=int,
		default=10,
		help="Maximum number of group curves to plot when a grouping column exists.",
	)
	return parser


def main() -> None:
	args = build_argparser().parse_args()
	input_dir = args.input_dir
	outdir = args.outdir or (input_dir / "plots_fast_MRI")
	aggregate = args.aggregate
	max_groups = args.max_groups

	if not input_dir.exists():
		raise SystemExit(f"Input folder not found: {input_dir}")

	counts = generate_plots(
		input_dir,
		outdir,
		aggregate=aggregate,
		max_groups=max(1, int(max_groups)),
	)

	total = int(sum(counts.values()))
	print(f"Generated {total} plot(s) in: {outdir}")
	for structure_name, n in sorted(counts.items()):
		print(f"  - {structure_name}: {n}")


if __name__ == "__main__":
	main()
