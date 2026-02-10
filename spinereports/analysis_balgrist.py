#!/usr/bin/env python3
"""Correlate morphometrics with Balgrist severity scores.

This script scans a `reports/` folder produced by the pipeline (see `tree.txt`),
loads per-subject morphometrics from `reports/sub-*/files/*_subject.csv`, merges
them with the severity scores from `Readout_lumbar_23112025.csv` via `Lfd_Nr`,
and runs correlation analyses.

Inputs
------
- reports directory containing:
  - `Readout_lumbar_23112025.csv` (or pass `--readout`)
  - subject folders like `sub-001_acq-sag_T2w/files/*.csv`

Outputs
-------
Written to `--outdir` (default: `<reports>/analysis_balgrist_out`):
- `merged_subject_level.csv`
- one `correlations__<outcome>.csv` per outcome
- `top_correlations.csv`
- quick plots under `plots/`

Dependencies
------------
Requires `pandas` and `scipy` (plus matplotlib/seaborn already in this repo).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

def _safe_col(col: str) -> str:
	col = col.strip()
	col = re.sub(r"\s+", "_", col)
	col = re.sub(r"[^A-Za-z0-9_\-]+", "", col)
	col = re.sub(r"_+", "_", col)
	return col.strip("_")


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
	"""
	Benjamini–Hochberg FDR-adjusted p-values.
	Returns array of same shape with NaNs preserved.
	"""
	p = np.asarray(pvals, dtype=float)
	out = np.full_like(p, np.nan, dtype=float)
	mask = np.isfinite(p)
	if not mask.any():
		return out

	p_nonan = p[mask]
	order = np.argsort(p_nonan)
	ranked = p_nonan[order]
	m = ranked.size
	q = ranked * m / (np.arange(1, m + 1))
	q = np.minimum.accumulate(q[::-1])[::-1]
	q = np.clip(q, 0.0, 1.0)

	tmp = np.empty_like(q)
	tmp[order] = q
	out[mask] = tmp
	return out

def _parse_subject_number(subject_dir_name: str) -> Optional[int]:
	"""Extract subject number from a folder like `sub-001_acq-sag_T2w`."""
	m = re.search(r"\bsub-(\d+)", subject_dir_name)
	if not m:
		return None
	try:
		return int(m.group(1))
	except ValueError:
		return None


def _numeric_columns(df: "pd.DataFrame") -> List[str]:
	return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _read_csv(path: Path) -> "pd.DataFrame":
	# Be liberal with separators/encodings; this dataset sometimes includes special chars.
	return pd.read_csv(path, sep=",", engine="python")


def _flatten_grouped_stats(
	df: "pd.DataFrame",
	*,
	prefix: str,
	group_col: str,
	stats: Sequence[str] = ("mean",),
	drop_cols: Sequence[str] = (),
) -> Dict[str, float]:
	"""Group by `group_col`, compute stats for numeric columns, flatten into dict."""
	out: Dict[str, float] = {}

	numeric_cols = [
		c
		for c in df.columns
		if c not in set(drop_cols) | {group_col}
		and pd.api.types.is_numeric_dtype(df[c])
	]
	if not numeric_cols:
		return out

	grouped = df.groupby(group_col)[numeric_cols].agg(list(stats))
	# Columns become MultiIndex: (col, stat)
	for (col, stat_name), series in grouped.items():
		for level, val in series.items():
			key = f"{prefix}_{_safe_col(col)}_{stat_name}_{_safe_col(str(level))}"
			out[key] = float(val) if pd.notna(val) else np.nan

	# Add global stats across all rows
	global_stats = df[numeric_cols].agg(list(stats))
	for col in numeric_cols:
		for stat_name in stats:
			val = global_stats.loc[stat_name, col]
			key = f"{prefix}_{_safe_col(col)}_{stat_name}_ALL"
			out[key] = float(val) if pd.notna(val) else np.nan

	return out


def _rowwise_to_features(
	df: "pd.DataFrame",
	*,
	prefix: str,
	id_col: str,
	drop_cols: Sequence[str] = (),
) -> Dict[str, float]:
	"""Each row corresponds to an entity (e.g., disc level). Flatten numeric fields."""
	out: Dict[str, float] = {}

	numeric_cols = [
		c
		for c in df.columns
		if c not in set(drop_cols) | {id_col} and pd.api.types.is_numeric_dtype(df[c])
	]
	if not numeric_cols:
		return out

	for _, row in df.iterrows():
		entity = row.get(id_col)
		if pd.isna(entity):
			continue
		entity_str = _safe_col(str(entity))
		for col in numeric_cols:
			key = f"{prefix}_{_safe_col(col)}_{entity_str}"
			val = row[col]
			out[key] = float(val) if pd.notna(val) else np.nan

	return out


def load_subject_features(reports_dir: Path) -> "pd.DataFrame":
	"""Load all `*_subject.csv` files under `reports/sub-*/files/` into feature table."""
	subject_dirs = [p for p in reports_dir.iterdir() if p.is_dir() and "sub-" in p.name]
	rows: List[Dict[str, float]] = []
	missing: List[str] = []

	for subj_dir in sorted(subject_dirs):
		subj_num = _parse_subject_number(subj_dir.name)
		if subj_num is None:
			continue

		files_dir = subj_dir / "files"
		if not files_dir.exists():
			missing.append(str(files_dir))
			continue

		feat: Dict[str, float] = {"Lfd_Nr": float(subj_num)}

		# canal
		canal_path = files_dir / "canal_subject.csv"
		if canal_path.exists():
			canal = _read_csv(canal_path)
			if "vertebra_level" in canal.columns:
				feat.update(
					_flatten_grouped_stats(
						canal,
						prefix="canal",
						group_col="vertebra_level",
						stats=("mean", "std", "median"),
						drop_cols=("slice_interp", "structure", "structure_name"),
					)
				)

		# csf
		csf_path = files_dir / "csf_subject.csv"
		if csf_path.exists():
			csf = _read_csv(csf_path)
			if "vertebra_level" in csf.columns:
				feat.update(
					_flatten_grouped_stats(
						csf,
						prefix="csf",
						group_col="vertebra_level",
						stats=("mean", "std", "median"),
						drop_cols=("slice_interp", "structure", "structure_name"),
					)
				)

		# discs
		discs_path = files_dir / "discs_subject.csv"
		if discs_path.exists():
			discs = _read_csv(discs_path)
			if "structure_name" in discs.columns:
				feat.update(
					_rowwise_to_features(
						discs,
						prefix="discs",
						id_col="structure_name",
						drop_cols=("structure",),
					)
				)

		# foramens
		foramens_path = files_dir / "foramens_subject.csv"
		if foramens_path.exists():
			foramens = _read_csv(foramens_path)
			if "structure_name" in foramens.columns:
				feat.update(
					_rowwise_to_features(
						foramens,
						prefix="foramens",
						id_col="structure_name",
						drop_cols=("structure",),
					)
				)

		# vertebrae
		vertebrae_path = files_dir / "vertebrae_subject.csv"
		if vertebrae_path.exists():
			vertebrae = _read_csv(vertebrae_path)
			if "structure_name" in vertebrae.columns:
				feat.update(
					_rowwise_to_features(
						vertebrae,
						prefix="vertebrae",
						id_col="structure_name",
						drop_cols=("structure",),
					)
				)

		rows.append(feat)

	if missing:
		print(f"Warning: missing files dirs for {len(missing)} subject(s)")

	features = pd.DataFrame(rows)
	if not features.empty:
		# keep `Lfd_Nr` as int-like
		features["Lfd_Nr"] = features["Lfd_Nr"].astype(int)
	return features


def load_readout(readout_csv: Path) -> "pd.DataFrame":
	df = _read_csv(readout_csv)
	if "Lfd_Nr" not in df.columns:
		raise SystemExit(f"Expected column 'Lfd_Nr' in {readout_csv}")

	# Drop completely empty unnamed columns
	df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]]

	# Coerce Lfd_Nr to int
	df["Lfd_Nr"] = pd.to_numeric(df["Lfd_Nr"], errors="coerce").astype("Int64")
	df = df[df["Lfd_Nr"].notna()].copy()
	df["Lfd_Nr"] = df["Lfd_Nr"].astype(int)
	return df


def select_outcome_columns(
	readout: "pd.DataFrame",
	*,
	severity_cols: Optional[List[str]],
	severity_regex: Optional[str],
	exclude_regex: Optional[str],
) -> List[str]:

	if severity_cols:
		missing = [c for c in severity_cols if c not in readout.columns]
		if missing:
			raise SystemExit(f"Requested --severity-cols not found: {missing}")
		return severity_cols

	regex = re.compile(severity_regex, re.IGNORECASE) if severity_regex else None
	exclude = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None

	# Default: focus on stenosis-related columns
	candidates: List[str] = []
	for c in readout.columns:
		if c == "Lfd_Nr":
			continue
		if exclude and exclude.search(str(c)):
			continue
		if regex:
			if regex.search(str(c)):
				candidates.append(c)
		else:
			if "stenosis" in str(c).lower() and "intra" not in str(c).lower():
				candidates.append(c)

	outcomes: List[str] = []
	for c in candidates:
		if pd.api.types.is_numeric_dtype(readout[c]):
			outcomes.append(c)
			continue
		# If the column got parsed as object due to missing values or mixed types,
		# attempt coercion for outcome candidates.
		coerced = pd.to_numeric(readout[c], errors="coerce")
		if int(coerced.notna().sum()) >= 3:
			readout[c] = coerced
			outcomes.append(c)

	return outcomes


def compute_correlations(
	merged: "pd.DataFrame",
	*,
	outcomes: Sequence[str],
	feature_prefixes: Optional[Sequence[str]],
	min_n: int,
) -> "pd.DataFrame":

	# Feature columns: numeric and not outcomes
	numeric_cols = _numeric_columns(merged)
	feature_cols = [c for c in numeric_cols if c not in set(outcomes) | {"Lfd_Nr"}]
	if feature_prefixes:
		prefixes = tuple(feature_prefixes)
		feature_cols = [c for c in feature_cols if c.startswith(prefixes)]

	results: List[Dict[str, object]] = []
	for outcome in outcomes:
		if outcome not in merged.columns:
			continue

		y_full = merged[outcome]
		if not pd.api.types.is_numeric_dtype(y_full):
			continue

		for feature in feature_cols:
			x_full = merged[feature]
			mask = x_full.notna() & y_full.notna()
			n = int(mask.sum())
			if n < min_n:
				continue

			x = x_full[mask].astype(float).to_numpy()
			y = y_full[mask].astype(float).to_numpy()

			# Pearson
			try:
				pearson_r, pearson_p = stats.pearsonr(x, y)
			except Exception:
				pearson_r, pearson_p = np.nan, np.nan

			# Spearman
			try:
				spearman_r, spearman_p = stats.spearmanr(x, y)
			except Exception:
				spearman_r, spearman_p = np.nan, np.nan

			results.append(
				{
					"outcome": outcome,
					"feature": feature,
					"n": n,
					"pearson_r": float(pearson_r),
					"pearson_p": float(pearson_p),
					"spearman_r": float(spearman_r),
					"spearman_p": float(spearman_p),
				}
			)

	res = pd.DataFrame(results)
	if res.empty:
		return res

	# FDR correction per outcome
	res["pearson_q"] = np.nan
	res["spearman_q"] = np.nan
	for outcome in res["outcome"].unique():
		idx = res["outcome"] == outcome
		res.loc[idx, "pearson_q"] = _bh_fdr(res.loc[idx, "pearson_p"].to_numpy())
		res.loc[idx, "spearman_q"] = _bh_fdr(res.loc[idx, "spearman_p"].to_numpy())

	# Sort: prefer spearman p then abs rho
	res = res.sort_values(["outcome", "spearman_q", "spearman_p"], ascending=[True, True, True])
	return res


def save_plots(
	merged: "pd.DataFrame",
	correlations: "pd.DataFrame",
	*,
	outdir: Path,
	top_k: int,
) -> None:
	import matplotlib.pyplot as plt
	import seaborn as sns

	plots_dir = outdir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	if correlations.empty:
		return

	top = correlations.copy()
	top["abs_spearman_r"] = top["spearman_r"].abs()
	top = top.sort_values(["spearman_q", "spearman_p", "abs_spearman_r"], ascending=[True, True, False])
	top = top.head(top_k)

	# Scatter plots for top correlations
	for _, row in top.iterrows():
		outcome = row["outcome"]
		feature = row["feature"]
		df = merged[[outcome, feature]].dropna()
		if df.shape[0] < 3:
			continue

		plt.figure(figsize=(5.0, 4.0))
		sns.regplot(data=df, x=feature, y=outcome, scatter_kws={"s": 25, "alpha": 0.8}, line_kws={"alpha": 0.8})
		plt.title(f"{outcome} vs {feature}\nSpearman r={row['spearman_r']:.3f}, q={row['spearman_q']:.3g}, n={int(row['n'])}")
		plt.tight_layout()
		fname = f"scatter__{_safe_col(str(outcome))}__{_safe_col(str(feature))}.png"
		plt.savefig(plots_dir / fname, dpi=200)
		plt.close()

	# Heatmap: outcomes x (top features per outcome)
	heatmap_outcomes = list(top["outcome"].unique())
	heatmap_features = list(top["feature"].unique())
	mat = (
		correlations[correlations["outcome"].isin(heatmap_outcomes) & correlations["feature"].isin(heatmap_features)]
		.pivot_table(index="feature", columns="outcome", values="spearman_r")
		.reindex(index=heatmap_features)
	)

	if not mat.empty:
		plt.figure(figsize=(max(6.0, 0.6 * len(heatmap_outcomes)), max(6.0, 0.25 * len(heatmap_features))))
		sns.heatmap(mat, cmap="vlag", center=0.0, linewidths=0.5, linecolor="white")
		plt.title("Spearman correlations (top pairs)")
		plt.tight_layout()
		plt.savefig(plots_dir / "heatmap_spearman_top.png", dpi=200)
		plt.close()


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Correlate morphometrics measurements with severity scores from Balgrist readout CSV."
	)
	p.add_argument(
		"reports_dir",
		type=Path,
		help="Path to reports folder (contains subject sub-*/files/*.csv and the readout CSV)",
	)
	p.add_argument(
		"--readout",
		type=Path,
		default=None,
		help="Optional path to Readout_lumbar_23112025.csv (default: <reports_dir>/Readout_lumbar_23112025.csv)",
	)
	p.add_argument(
		"--outdir",
		type=Path,
		default=None,
		help="Output directory (default: <reports_dir>/analysis_balgrist_out)",
	)
	p.add_argument(
		"--severity-cols",
		type=str,
		default=None,
		help="Comma-separated list of severity/outcome columns to analyze (exact names)",
	)
	p.add_argument(
		"--severity-regex",
		type=str,
		default=None,
		help="Regex used to select outcome columns (default: columns containing 'stenosis' and not 'Intra')",
	)
	p.add_argument(
		"--exclude-regex",
		type=str,
		default=r"\bIntra\b",
		help="Regex of columns to exclude from outcomes selection",
	)
	p.add_argument(
		"--feature-prefixes",
		type=str,
		default="canal,csf,discs,foramens,vertebrae",
		help="Comma-separated feature prefixes to include",
	)
	p.add_argument("--min-n", type=int, default=10, help="Minimum number of subjects required per correlation")
	p.add_argument("--top-k", type=int, default=20, help="Top correlations to plot/export")
	return p


def main() -> None:
	args = build_argparser().parse_args()

	reports_dir: Path = args.reports_dir
	if not reports_dir.exists():
		raise SystemExit(f"Reports directory not found: {reports_dir}")

	readout_path = args.readout or (reports_dir / "Readout_lumbar_23112025.csv")
	if not readout_path.exists():
		raise SystemExit(
			f"Readout CSV not found: {readout_path}. Provide it explicitly with --readout."
		)

	outdir = args.outdir or (reports_dir / "analysis_balgrist_out")
	outdir.mkdir(parents=True, exist_ok=True)

	readout = load_readout(readout_path)
	features = load_subject_features(reports_dir)

	if features.empty:
		raise SystemExit(f"No subject features found under: {reports_dir}")

	merged = readout.merge(features, on="Lfd_Nr", how="inner")
	merged.to_csv(outdir / "merged_subject_level.csv", index=False)
	print(f"Merged subjects: {merged.shape[0]} (readout={readout.shape[0]}, features={features.shape[0]})")

	severity_cols = [c.strip() for c in args.severity_cols.split(",")] if args.severity_cols else None
	outcomes = select_outcome_columns(
		readout,
		severity_cols=severity_cols,
		severity_regex=args.severity_regex,
		exclude_regex=args.exclude_regex,
	)
	if not outcomes:
		raise SystemExit(
			"No outcome columns selected. Use --severity-regex or --severity-cols to specify outcomes."
		)
	print(f"Outcomes: {len(outcomes)}")

	feature_prefixes = [p.strip() for p in args.feature_prefixes.split(",") if p.strip()]
	correlations = compute_correlations(
		merged,
		outcomes=outcomes,
		feature_prefixes=feature_prefixes,
		min_n=int(args.min_n),
	)

	if correlations.empty:
		raise SystemExit("No correlations computed (check --min-n and that outcomes/features are numeric)")

	# Write per-outcome CSVs
	for outcome in correlations["outcome"].unique():
		df_o = correlations[correlations["outcome"] == outcome].copy()
		df_o.to_csv(outdir / f"correlations__{_safe_col(str(outcome))}.csv", index=False)

	# Combined top-k CSV
	top = correlations.copy()
	top["abs_spearman_r"] = top["spearman_r"].abs()
	top = top.sort_values(["spearman_q", "spearman_p", "abs_spearman_r"], ascending=[True, True, False])
	top.head(int(args.top_k)).to_csv(outdir / "top_correlations.csv", index=False)

	# Plots
	try:
		save_plots(merged, correlations, outdir=outdir, top_k=int(args.top_k))
	except Exception as e:
		print(f"Warning: plotting failed: {e}")

	print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
	main()