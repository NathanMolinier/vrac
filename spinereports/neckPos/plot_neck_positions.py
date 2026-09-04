#!/usr/bin/env python3
"""Plot spine-report metrics across neck positions and subjects.

Expected input layout:

root/
  sub-002_ses-headDown_T2w/
    files/
      canal_subject.csv
      csf_subject.csv
      discs_subject.csv
      foramens_subject.csv
      vertebrae_subject.csv
  sub-002_ses-headNormal_T2w/
    files/
      ...
  sub-002_ses-headUp_T2w/
    files/
      ...
  sub-003_ses-headDown_T2w/
    ...

For every structure CSV present, two families of plots are written:

- per-subject: one figure per (subject, structure, metric), with the three
  head positions overlaid so intra-subject variation is easy to read.
- cross-subject: one figure per (structure, metric), summarising every
  subject at once so position differences can be compared across the group.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITION_ORDER: Tuple[str, ...] = ("headDown", "headNormal", "headUp")
POSITION_COLORS: Dict[str, str] = {
	"headDown": "#1f77b4",
	"headNormal": "#2ca02c",
	"headUp": "#d62728",
}

SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SESSION_RE = re.compile(r"ses-(head[A-Za-z0-9]+)")


def _safe_name(value: str) -> str:
	value = re.sub(r"\s+", "_", str(value).strip())
	value = re.sub(r"[^A-Za-z0-9_.\-]+", "", value)
	value = re.sub(r"_+", "_", value)
	return value.strip("_")


def _parse_subject_position(folder_name: str) -> Optional[Tuple[str, str]]:
	sub_match = SUBJECT_RE.search(folder_name)
	pos_match = SESSION_RE.search(folder_name)
	if sub_match is None or pos_match is None:
		return None
	return sub_match.group(1), pos_match.group(1)


def _find_session_dirs(root_dir: Path) -> List[Tuple[str, str, Path]]:
	"""Return (subject, position, folder) for every session folder found."""
	entries: List[Tuple[str, str, Path]] = []
	for child in sorted(root_dir.iterdir()):
		if not child.is_dir():
			continue
		if not (child / "files").exists():
			continue
		parsed = _parse_subject_position(child.name)
		if parsed is None:
			continue
		subject, position = parsed
		entries.append((subject, position, child))
	return entries


def _position_sort_key(position: str) -> Tuple[int, str]:
	try:
		return (POSITION_ORDER.index(position), position)
	except ValueError:
		return (len(POSITION_ORDER), position)


def _position_color(position: str) -> str:
	if position in POSITION_COLORS:
		return POSITION_COLORS[position]
	# Deterministic fallback for unexpected labels.
	cmap = plt.get_cmap("tab10")
	return cmap(hash(position) % 10)


def _numeric_cols(df: pd.DataFrame) -> List[str]:
	excluded = {"structure", "slice_interp", "subject", "position"}
	return [
		c for c in df.columns
		if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
	]


def _is_slice_data(df: pd.DataFrame) -> bool:
	return (
		"slice_interp" in df.columns
		and df["slice_interp"].dropna().nunique() > 1
	)


def _choose_group_col(df: pd.DataFrame) -> Optional[str]:
	for candidate in ("structure_name", "vertebra_level"):
		if candidate in df.columns:
			if df[candidate].dropna().nunique() > 1:
				return candidate
	return None


def _collect_structure_data(
	session_dirs: Sequence[Tuple[str, str, Path]],
	structure_csv_name: str,
) -> pd.DataFrame:
	rows: List[pd.DataFrame] = []
	for subject, position, session_dir in session_dirs:
		csv_path = session_dir / "files" / structure_csv_name
		if not csv_path.exists():
			continue
		df = pd.read_csv(csv_path)
		df["subject"] = subject
		df["position"] = position
		rows.append(df)
	if not rows:
		return pd.DataFrame()
	return pd.concat(rows, ignore_index=True)


def _save_fig(fig, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(path, dpi=180)
	plt.close(fig)


def _vertebra_boundaries(
	slices: Sequence[float],
	slice_to_vlevel: Dict[float, Optional[str]],
) -> List[Tuple[float, float, str]]:
	boundaries: List[Tuple[float, float, str]] = []
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
	return boundaries


def _draw_vertebra_dividers(ax, boundaries: Sequence[Tuple[float, float, str]]) -> None:
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


def _slice_to_vlevel_map(
	sub: pd.DataFrame,
	slices: Sequence[float],
	vlevel_col: str,
) -> Dict[float, Optional[str]]:
	slice_to_vlevel: Dict[float, Optional[str]] = {}
	for sl in slices:
		vals = sub.loc[sub["slice_interp"] == sl, vlevel_col].dropna()
		slice_to_vlevel[sl] = str(vals.mode().iloc[0]) if not vals.empty else None
	return slice_to_vlevel


def _partitions_by_structure(df: pd.DataFrame) -> List[Tuple[Optional[str], pd.DataFrame]]:
	if "structure_name" in df.columns and df["structure_name"].astype(str).nunique() > 1:
		return [(str(name), sub) for name, sub in df.groupby("structure_name", sort=True)]
	return [(None, df)]


# ---------------------------------------------------------------------------
# Per-subject plots
# ---------------------------------------------------------------------------


def _plot_subject_slice_metric(
	subject: str,
	sub_df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
) -> int:
	count = 0
	for label, sub in _partitions_by_structure(sub_df):
		if sub.empty:
			continue
		slices = sorted(sub["slice_interp"].dropna().unique().tolist())
		if not slices:
			continue

		vlevel_col = "vertebra_level" if "vertebra_level" in sub.columns else None
		slice_to_vlevel = (
			_slice_to_vlevel_map(sub, slices, vlevel_col) if vlevel_col else {}
		)

		positions = sorted(
			sub["position"].dropna().unique().tolist(), key=_position_sort_key
		)
		if not positions:
			continue

		fig, ax = plt.subplots(figsize=(10.0, 5.0))
		for pos in positions:
			row = sub[sub["position"] == pos].sort_values("slice_interp")
			if row.empty:
				continue
			ax.plot(
				row["slice_interp"],
				row[metric],
				marker=".",
				markersize=3,
				linewidth=1.2,
				alpha=0.9,
				color=_position_color(pos),
				label=pos,
			)

		if vlevel_col is not None:
			boundaries = _vertebra_boundaries(slices, slice_to_vlevel)
			_draw_vertebra_dividers(ax, boundaries)

		ax.legend(loc="best", fontsize=8, frameon=False, title="Position")
		title = f"{subject} - {structure_name} - {metric}"
		if label is not None:
			title += f" ({label})"
		ax.set_title(title)
		ax.set_xlabel("Slice (slice_interp)")
		ax.set_ylabel(metric)
		ax.grid(True, alpha=0.25)

		suffix = f"__{_safe_name(label)}" if label else ""
		struct_stem = _safe_name(structure_name.replace("_subject.csv", ""))
		fig_path = outdir / f"{struct_stem}__{_safe_name(metric)}{suffix}.png"
		_save_fig(fig, fig_path)
		count += 1
	return count


def _plot_subject_grouped_metric(
	subject: str,
	sub_df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
) -> int:
	group_col = _choose_group_col(sub_df)
	positions = sorted(
		sub_df["position"].dropna().unique().tolist(), key=_position_sort_key
	)
	if not positions:
		return 0

	fig, ax = plt.subplots(figsize=(8.5, 4.5))

	if group_col is None:
		values = [
			sub_df.loc[sub_df["position"] == pos, metric].dropna().mean()
			for pos in positions
		]
		x = np.arange(len(positions))
		ax.bar(
			x,
			values,
			color=[_position_color(p) for p in positions],
			edgecolor="0.2",
		)
		ax.set_xticks(x)
		ax.set_xticklabels(positions)
	else:
		groups = sorted(
			sub_df[group_col].dropna().astype(str).unique().tolist()
		)
		if not groups:
			plt.close(fig)
			return 0

		x = np.arange(len(groups))
		total_width = 0.8
		bar_width = total_width / max(len(positions), 1)
		for i, pos in enumerate(positions):
			values = []
			for g in groups:
				sel = (
					(sub_df["position"] == pos)
					& (sub_df[group_col].astype(str) == g)
				)
				vals = sub_df.loc[sel, metric].dropna()
				values.append(vals.mean() if not vals.empty else np.nan)
			offset = (i - (len(positions) - 1) / 2.0) * bar_width
			ax.bar(
				x + offset,
				values,
				width=bar_width,
				color=_position_color(pos),
				edgecolor="0.2",
				label=pos,
			)
		ax.set_xticks(x)
		ax.set_xticklabels(groups, rotation=30, ha="right")
		ax.legend(loc="best", fontsize=8, frameon=False, title="Position")

	ax.set_title(f"{subject} - {structure_name} - {metric}")
	ax.set_xlabel(group_col if group_col else "Position")
	ax.set_ylabel(metric)
	ax.grid(True, axis="y", alpha=0.25)

	struct_stem = _safe_name(structure_name.replace("_subject.csv", ""))
	fig_path = outdir / f"{struct_stem}__{_safe_name(metric)}.png"
	_save_fig(fig, fig_path)
	return 1


def _plot_subject_metric(
	subject: str,
	sub_df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
) -> int:
	if _is_slice_data(sub_df):
		return _plot_subject_slice_metric(
			subject,
			sub_df,
			structure_name=structure_name,
			metric=metric,
			outdir=outdir,
		)
	return _plot_subject_grouped_metric(
		subject,
		sub_df,
		structure_name=structure_name,
		metric=metric,
		outdir=outdir,
	)


# ---------------------------------------------------------------------------
# Cross-subject plots
# ---------------------------------------------------------------------------


def _plot_cross_subject_slice_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
	max_subjects: int,
) -> int:
	"""x=slice_interp, mean +/- std across subjects, one band per position."""
	count = 0
	for label, sub in _partitions_by_structure(df):
		if sub.empty:
			continue
		positions = sorted(
			sub["position"].dropna().unique().tolist(), key=_position_sort_key
		)
		if not positions:
			continue

		slices_all = sorted(sub["slice_interp"].dropna().unique().tolist())
		if not slices_all:
			continue

		vlevel_col = "vertebra_level" if "vertebra_level" in sub.columns else None
		slice_to_vlevel = (
			_slice_to_vlevel_map(sub, slices_all, vlevel_col) if vlevel_col else {}
		)

		# One figure: summary (mean +/- std across subjects) per position.
		fig, ax = plt.subplots(figsize=(10.0, 5.0))
		for pos in positions:
			pos_df = sub[sub["position"] == pos]
			if pos_df.empty:
				continue
			agg = (
				pos_df.groupby("slice_interp")[metric]
				.agg(["mean", "std", "count"])
				.reset_index()
				.sort_values("slice_interp")
			)
			color = _position_color(pos)
			ax.plot(
				agg["slice_interp"],
				agg["mean"],
				color=color,
				linewidth=1.6,
				label=f"{pos} (n={int(agg['count'].max() or 0)})",
			)
			ax.fill_between(
				agg["slice_interp"],
				agg["mean"] - agg["std"].fillna(0.0),
				agg["mean"] + agg["std"].fillna(0.0),
				color=color,
				alpha=0.18,
				linewidth=0,
			)

		if vlevel_col is not None:
			boundaries = _vertebra_boundaries(slices_all, slice_to_vlevel)
			_draw_vertebra_dividers(ax, boundaries)

		ax.legend(loc="best", fontsize=8, frameon=False, title="Position")
		title = f"All subjects - {structure_name} - {metric}"
		if label is not None:
			title += f" ({label})"
		ax.set_title(title)
		ax.set_xlabel("Slice (slice_interp)")
		ax.set_ylabel(metric)
		ax.grid(True, alpha=0.25)

		suffix = f"__{_safe_name(label)}" if label else ""
		struct_stem = _safe_name(structure_name.replace("_subject.csv", ""))
		fig_path = outdir / f"{struct_stem}__{_safe_name(metric)}{suffix}__summary.png"
		_save_fig(fig, fig_path)
		count += 1

		# Second figure: per-subject overlay, one panel per position, so
		# subject-level variability is inspectable.
		subjects = sorted(sub["subject"].dropna().unique().tolist())
		if max_subjects and len(subjects) > max_subjects:
			subjects = subjects[:max_subjects]

		fig, axes = plt.subplots(
			1,
			len(positions),
			figsize=(4.8 * len(positions), 4.5),
			sharey=True,
			squeeze=False,
		)
		cmap = plt.get_cmap("tab20", max(len(subjects), 1))
		subj_color = {s: cmap(i) for i, s in enumerate(subjects)}

		for ax_pos, pos in zip(axes[0], positions):
			pos_df = sub[sub["position"] == pos]
			for subj in subjects:
				row = pos_df[pos_df["subject"] == subj].sort_values("slice_interp")
				if row.empty:
					continue
				ax_pos.plot(
					row["slice_interp"],
					row[metric],
					color=subj_color[subj],
					linewidth=1.0,
					alpha=0.85,
					label=subj,
				)
			ax_pos.set_title(pos)
			ax_pos.set_xlabel("Slice (slice_interp)")
			ax_pos.grid(True, alpha=0.25)
			if vlevel_col is not None:
				boundaries = _vertebra_boundaries(slices_all, slice_to_vlevel)
				_draw_vertebra_dividers(ax_pos, boundaries)

		axes[0][0].set_ylabel(metric)
		handles, labels = axes[0][0].get_legend_handles_labels()
		if handles:
			fig.legend(
				handles,
				labels,
				loc="center right",
				fontsize=7,
				frameon=False,
				title="Subject",
				bbox_to_anchor=(1.0, 0.5),
			)
		title = f"Per-subject - {structure_name} - {metric}"
		if label is not None:
			title += f" ({label})"
		fig.suptitle(title)

		fig_path = outdir / f"{struct_stem}__{_safe_name(metric)}{suffix}__per_subject.png"
		fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.96))
		fig.savefig(fig_path, dpi=180)
		plt.close(fig)
		count += 1

	return count


def _plot_cross_subject_grouped_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
) -> int:
	group_col = _choose_group_col(df)
	positions = sorted(
		df["position"].dropna().unique().tolist(), key=_position_sort_key
	)
	if not positions:
		return 0

	if group_col is None:
		# Box plot of per-subject values grouped by position.
		fig, ax = plt.subplots(figsize=(6.5, 4.5))
		data = [df.loc[df["position"] == p, metric].dropna().values for p in positions]
		box = ax.boxplot(
			data,
			labels=positions,
			showfliers=True,
			patch_artist=True,
		)
		for patch, pos in zip(box["boxes"], positions):
			patch.set_facecolor(_position_color(pos))
			patch.set_alpha(0.55)
			patch.set_edgecolor("0.2")

		# Overlay individual subjects as jittered points for readability.
		rng = np.random.default_rng(0)
		for i, pos in enumerate(positions, start=1):
			vals = df.loc[df["position"] == pos, metric].dropna().values
			if vals.size == 0:
				continue
			jitter = rng.uniform(-0.12, 0.12, size=vals.size)
			ax.plot(
				np.full_like(vals, i, dtype=float) + jitter,
				vals,
				"o",
				markersize=3.5,
				color="0.15",
				alpha=0.6,
			)

		ax.set_title(f"All subjects - {structure_name} - {metric}")
		ax.set_xlabel("Position")
		ax.set_ylabel(metric)
		ax.grid(True, axis="y", alpha=0.25)

		struct_stem = _safe_name(structure_name.replace("_subject.csv", ""))
		fig_path = outdir / f"{struct_stem}__{_safe_name(metric)}.png"
		_save_fig(fig, fig_path)
		return 1

	groups = sorted(df[group_col].dropna().astype(str).unique().tolist())
	if not groups:
		return 0

	# Grouped boxplot: for each group value, a small cluster of position boxes.
	fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(groups) + 2.0), 4.8))
	total_width = 0.75
	box_width = total_width / max(len(positions), 1)

	for i, pos in enumerate(positions):
		data = []
		for g in groups:
			sel = (df["position"] == pos) & (df[group_col].astype(str) == g)
			data.append(df.loc[sel, metric].dropna().values)
		positions_x = np.arange(len(groups)) + (i - (len(positions) - 1) / 2.0) * box_width
		box = ax.boxplot(
			data,
			positions=positions_x,
			widths=box_width * 0.85,
			showfliers=False,
			patch_artist=True,
			manage_ticks=False,
		)
		color = _position_color(pos)
		for patch in box["boxes"]:
			patch.set_facecolor(color)
			patch.set_alpha(0.55)
			patch.set_edgecolor("0.2")
		for median in box["medians"]:
			median.set_color("0.1")
		# Legend proxy.
		ax.plot([], [], color=color, linewidth=6, alpha=0.55, label=pos)

	ax.set_xticks(np.arange(len(groups)))
	ax.set_xticklabels(groups, rotation=30, ha="right")
	ax.set_title(f"All subjects - {structure_name} - {metric}")
	ax.set_xlabel(group_col)
	ax.set_ylabel(metric)
	ax.grid(True, axis="y", alpha=0.25)
	ax.legend(loc="best", fontsize=8, frameon=False, title="Position")

	struct_stem = _safe_name(structure_name.replace("_subject.csv", ""))
	fig_path = outdir / f"{struct_stem}__{_safe_name(metric)}.png"
	_save_fig(fig, fig_path)
	return 1


def _plot_cross_subject_metric(
	df: pd.DataFrame,
	*,
	structure_name: str,
	metric: str,
	outdir: Path,
	max_subjects: int,
) -> int:
	if _is_slice_data(df):
		return _plot_cross_subject_slice_metric(
			df,
			structure_name=structure_name,
			metric=metric,
			outdir=outdir,
			max_subjects=max_subjects,
		)
	return _plot_cross_subject_grouped_metric(
		df,
		structure_name=structure_name,
		metric=metric,
		outdir=outdir,
	)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_plots(
	root_dir: Path,
	outdir: Path,
	*,
	max_subjects: int,
) -> Dict[str, Dict[str, int]]:
	session_dirs = _find_session_dirs(root_dir)
	if not session_dirs:
		raise SystemExit(
			f"No session folders found in {root_dir}. "
			"Expected 'sub-*_ses-head*' directories each containing 'files/'."
		)

	outdir.mkdir(parents=True, exist_ok=True)
	per_subject_root = outdir / "per_subject"
	cross_root = outdir / "cross_subject"

	structure_files: set[str] = set()
	for _, _, session_dir in session_dirs:
		for csv_path in (session_dir / "files").glob("*_subject.csv"):
			structure_files.add(csv_path.name)

	if not structure_files:
		raise SystemExit("No '*_subject.csv' files found in session folders.")

	counts: Dict[str, Dict[str, int]] = {"per_subject": {}, "cross_subject": {}}

	for structure_csv_name in sorted(structure_files):
		full_df = _collect_structure_data(session_dirs, structure_csv_name)
		if full_df.empty:
			continue

		num_cols = _numeric_cols(full_df)
		if not num_cols:
			continue

		# Per-subject plots.
		subj_total = 0
		for subject, sub_df in full_df.groupby("subject", sort=True):
			subject_out = per_subject_root / _safe_name(subject)
			for metric in num_cols:
				subj_total += _plot_subject_metric(
					subject,
					sub_df,
					structure_name=structure_csv_name,
					metric=metric,
					outdir=subject_out,
				)
		counts["per_subject"][structure_csv_name] = subj_total

		# Cross-subject plots.
		cross_total = 0
		for metric in num_cols:
			cross_total += _plot_cross_subject_metric(
				full_df,
				structure_name=structure_csv_name,
				metric=metric,
				outdir=cross_root,
				max_subjects=max_subjects,
			)
		counts["cross_subject"][structure_csv_name] = cross_total

	return counts


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Generate per-subject and cross-subject metric plots comparing "
			"neck positions (headDown / headNormal / headUp)."
		)
	)
	parser.add_argument(
		"input_dir",
		type=Path,
		help="Folder containing 'sub-*_ses-head*' session subfolders.",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=None,
		help="Output folder for PNG plots (default: <input_dir>/plots_neckPos).",
	)
	parser.add_argument(
		"--max-subjects",
		type=int,
		default=20,
		help="Maximum number of subjects to overlay on per-subject panels.",
	)
	return parser


def main() -> None:
	args = build_argparser().parse_args()
	input_dir = args.input_dir
	outdir = args.outdir or (input_dir / "plots_neckPos")
	max_subjects = max(1, int(args.max_subjects))

	if not input_dir.exists():
		raise SystemExit(f"Input folder not found: {input_dir}")

	counts = generate_plots(input_dir, outdir, max_subjects=max_subjects)

	subj_total = int(sum(counts["per_subject"].values()))
	cross_total = int(sum(counts["cross_subject"].values()))
	print(f"Generated {subj_total} per-subject plot(s) and {cross_total} cross-subject plot(s) in: {outdir}")
	for scope in ("per_subject", "cross_subject"):
		print(f"  [{scope}]")
		for structure_name, n in sorted(counts[scope].items()):
			print(f"    - {structure_name}: {n}")


if __name__ == "__main__":
	main()
