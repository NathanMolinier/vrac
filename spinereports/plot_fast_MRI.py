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
	match = re.search(r"meas(\d+)", folder_name, flags=re.IGNORECASE)
	if not match:
		return None
	try:
		return int(match.group(1))
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


def _plot_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
	aggregate: str,
	max_groups: int,
) -> None:
	group_col = _choose_group_col(df)
	fig, ax = plt.subplots(figsize=(7.5, 4.2))

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
	ax.grid(True, alpha=0.25)

	structure_stem = _safe_name(structure_name.replace("_subject.csv", ""))
	metric_stem = _safe_name(metric)
	fig_path = outdir / f"{structure_stem}__{metric_stem}.png"
	fig.tight_layout()
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)


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
			_plot_metric(
				structure_df,
				structure_name=structure_csv_name,
				metric=metric,
				outdir=outdir,
				aggregate=aggregate,
				max_groups=max_groups,
			)
			count += 1

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
	if not input_dir.exists():
		raise SystemExit(f"Input folder not found: {input_dir}")

	outdir = args.outdir or (input_dir / "plots_fast_MRI")
	counts = generate_plots(
		input_dir,
		outdir,
		aggregate=args.aggregate,
		max_groups=max(1, int(args.max_groups)),
	)

	total = int(sum(counts.values()))
	print(f"Generated {total} plot(s) in: {outdir}")
	for structure_name, n in sorted(counts.items()):
		print(f"  - {structure_name}: {n}")


if __name__ == "__main__":
	main()
