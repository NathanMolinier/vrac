#!/usr/bin/env python3
"""Correlate SpineNet IVD grades with expert Balgrist severity scores.

This script compares SpineNet automated predictions (Central Canal Stenosis, 
Foraminal Stenosis Left/Right) with expert radiological grading from Balgrist readout.

Inputs
------
- pred/ folder containing:
  - `Readout_lumbar_23112025.csv` (expert grading)
  - subject folders like `sub-001_acq-sag_T2w/ivd_grades.csv`

Outputs
-------
Written to `--outdir` (default: `<pred>/analysis_balgrist_spinenet_out`):
- `merged_subject_level.csv`
- one `correlations__<outcome>.csv` per outcome
- `top_correlations.csv`
- quick plots under `plots/`
"""

from __future__ import annotations

import argparse
import importlib
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
    Detects real effects without inflating false positives too much.
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


def _read_csv(path: Path) -> "pd.DataFrame":
    # Be liberal with separators/encodings; this dataset sometimes includes special chars.
    return pd.read_csv(path, sep=",", engine="python")


def load_spinenet_predictions(pred_dir: Path) -> "pd.DataFrame":
    """Load all ivd_grades.csv files from pred/sub-*/ivd_grades.csv into feature table."""
    subject_dirs = [p for p in pred_dir.iterdir() if p.is_dir() and "sub-" in p.name]
    rows: List[Dict] = []
    
    # Disc level to standard naming mapping
    disc_mapping = {
        "T11-T12": "T11-T12",
        "T12-L1": "T12-L1",
        "L1-L2": "L1-L2",
        "L2-L3": "L2-L3",
        "L3-L4": "L3-L4",
        "L4-L5": "L4-L5",
        "L5-S1": "L5-S1"
    }

    for subj_dir in sorted(subject_dirs):
        subj_num = _parse_subject_number(subj_dir.name)
        if subj_num is None:
            continue

        ivd_file = subj_dir / "ivd_grades.csv"
        if not ivd_file.exists():
            continue

        try:
            df = _read_csv(ivd_file)
        except Exception as e:
            print(f"Warning: Could not read {ivd_file}: {e}")
            continue
        
        # Name first column
        df.rename(columns={'Unnamed: 0': 'Level'}, inplace=True)

        # Process each discrete level
        for idx, row in df.iterrows():
            level = str(row.get("Level", "")).strip()
            if level not in disc_mapping:
                continue

            record = {"subject": subj_num, "Level": disc_mapping[level]}

            # Extract the three outcomes we care about
            for outcome in ["CentralCanalStenosis", "ForaminalStenosisLeft", "ForaminalStenosisRight"]:
                if outcome in row:
                    if outcome == "CentralCanalStenosis":
                        val = row[outcome] - 1 # Convert from 1-4 to 0-3
                    else:
                        val = row[outcome]
                    try:
                        record[f"spinenet_{outcome}"] = float(val) if pd.notna(val) else np.nan
                    except (ValueError, TypeError):
                        record[f"spinenet_{outcome}"] = np.nan
                else:
                    record[f"spinenet_{outcome}"] = np.nan

            if any(
                pd.notna(record.get(f"spinenet_{outcome}"))
                for outcome in ["CentralCanalStenosis", "ForaminalStenosisLeft", "ForaminalStenosisRight"]
            ):
                rows.append(record)

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return predictions

    # Sort by subject and level
    level_order = {disk: i for i, disk in enumerate(disc_mapping.keys())}
    predictions["_level_order"] = predictions["Level"].map(level_order)
    predictions = predictions.sort_values(["subject", "_level_order"]).drop(columns=["_level_order"])
    predictions = predictions.reset_index(drop=True)

    return predictions


def load_readout(readout_csv: Path) -> "pd.DataFrame":
    """Load expert readout and prepare for merging."""
    df = _read_csv(readout_csv)
    if "Lfd_Nr" not in df.columns:
        raise SystemExit(f"Expected column 'Lfd_Nr' in {readout_csv}")

    # Map level numbers to disc names
    level_mapping = {
        1: "L5-S1", 5: "L4-L5", 4: "L3-L4", 3: "L2-L3", 2: "L1-L2"
    }

    # Drop completely empty unnamed columns
    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]]

    # Coerce Lfd_Nr to int (this becomes subject ID)
    df["Lfd_Nr"] = pd.to_numeric(df["Lfd_Nr"], errors="coerce").astype("Int64")
    df = df[df["Lfd_Nr"].notna()].copy()
    df["Lfd_Nr"] = df["Lfd_Nr"].astype(int)
    df = df.rename(columns={"Lfd_Nr": "subject"})

    # Remap Level column using level_mapping
    if "Level" in df.columns:
        level_numeric = pd.to_numeric(df["Level"], errors="coerce")
        df["Level"] = level_numeric.map(level_mapping).fillna(df["Level"])

    # Create columns with averaged reader ratings
    reader_cols = [c for c in df.columns if "READER" in c and not "Intra" in c and "Senior" in c]
    for col in reader_cols:
        df[col.replace('_READER 1 (Senior)', ' ALL')]=(df[col]+df[col.replace('READER 1 (Senior)', 'READER 2 (Junior)')])/2
    
    return df


def compute_correlations(
    merged: "pd.DataFrame",
    outcomes: Sequence[str],
    feature_cols: Optional[Sequence[str]],
    min_n: int,
) -> "pd.DataFrame":
    """Compute Pearson and Spearman correlations."""
    results: List[Dict[str, object]] = []

    for outcome in outcomes:
        y_full = merged[outcome]
        for feature in feature_cols:
            x_full = merged[feature]
            mask = x_full.notna() & y_full.notna()
            n = int(mask.sum())
            if n < min_n:
                continue

            x = x_full[mask].astype(float).to_numpy()
            y = y_full[mask].astype(float).to_numpy()

            # Pearson with 95% CI
            pearson_r, pearson_p = np.nan, np.nan
            pearson_ci_lower, pearson_ci_upper = np.nan, np.nan
            try:
                pearson_r, pearson_p = stats.pearsonr(x, y)
                # Fisher z-transformation for CI
                z = 0.5 * np.log((1 + pearson_r) / (1 - pearson_r))
                se_z = 1.0 / np.sqrt(n - 3)
                z_crit = 1.96
                z_lower = z - z_crit * se_z
                z_upper = z + z_crit * se_z
                pearson_ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                pearson_ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            except Exception:
                pass

            # Spearman with 95% CI
            spearman_r, spearman_p = np.nan, np.nan
            spearman_ci_lower, spearman_ci_upper = np.nan, np.nan
            try:
                spearman_r, spearman_p = stats.spearmanr(x, y)
                # Fisher z-transformation for CI
                z = 0.5 * np.log((1 + spearman_r) / (1 - spearman_r))
                se_z = 1.0 / np.sqrt(n - 3)
                z_crit = 1.96
                z_lower = z - z_crit * se_z
                z_upper = z + z_crit * se_z
                spearman_ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                spearman_ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            except Exception:
                pass

            results.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "n": n,
                    "pearson_r": float(pearson_r),
                    "pearson_ci_lower": float(pearson_ci_lower),
                    "pearson_ci_upper": float(pearson_ci_upper),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_ci_lower": float(spearman_ci_lower),
                    "spearman_ci_upper": float(spearman_ci_upper),
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

    # Sort: prefer spearman q then p
    res = res.sort_values(["outcome", "spearman_q", "spearman_p"], ascending=[True, True, True])
    return res


def compute_ordinal_logit(
    merged: "pd.DataFrame",
    outcomes: Sequence[str],
    feature_cols: Optional[Sequence[str]],
    min_n: int,
) -> "pd.DataFrame":
    """Univariate ordinal logistic regression (proportional odds) per outcome/feature."""
    try:
        ordinal_module = importlib.import_module("statsmodels.miscmodels.ordinal_model")
        OrderedModel = getattr(ordinal_module, "OrderedModel")
    except Exception:
        print("Warning: statsmodels not available; skipping ordinal logistic regression.")
        return pd.DataFrame()

    if not feature_cols:
        return pd.DataFrame()

    results: List[Dict[str, object]] = []
    for outcome in outcomes:
        y_full = pd.to_numeric(merged[outcome], errors="coerce")
        for feature in feature_cols:
            x_full = pd.to_numeric(merged[feature], errors="coerce")
            mask = x_full.notna() & y_full.notna()
            n = int(mask.sum())
            if n < min_n:
                continue

            df = pd.DataFrame(
                {
                    "x": x_full[mask].astype(float),
                    "y": y_full[mask].astype(float),
                }
            ).dropna()
            if df.shape[0] < min_n:
                continue

            # Need at least 2 ordered classes
            y_unique = np.sort(df["y"].unique())
            if y_unique.size < 2:
                continue

            x_std = float(df["x"].std(ddof=0))
            if not np.isfinite(x_std) or x_std == 0:
                continue

            df["x_z"] = (df["x"] - float(df["x"].mean())) / x_std
            y_ord = pd.Categorical(df["y"], categories=y_unique, ordered=True)

            try:
                model = OrderedModel(y_ord, df[["x_z"]], distr="logit")
                fit = model.fit(method="bfgs", disp=False)
            except Exception:
                continue

            coef = float(fit.params.get("x_z", np.nan))
            se = float(fit.bse.get("x_z", np.nan))
            z_val = float(fit.tvalues.get("x_z", np.nan))
            p_val = float(fit.pvalues.get("x_z", np.nan))
            or_val = float(np.exp(coef)) if np.isfinite(coef) else np.nan

            # 95% CI for odds ratio
            or_ci_lower = np.nan
            or_ci_upper = np.nan
            if np.isfinite(coef) and np.isfinite(se):
                coef_ci_lower = coef - 1.96 * se
                coef_ci_upper = coef + 1.96 * se
                or_ci_lower = float(np.exp(coef_ci_lower))
                or_ci_upper = float(np.exp(coef_ci_upper))

            results.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "n": int(df.shape[0]),
                    "n_classes": int(y_unique.size),
                    "odds_std": x_std,
                    "coef_log_odds_per_sd": coef,
                    "odds_ratio_per_sd": or_val,
                    "or_ci_lower": or_ci_lower,
                    "or_ci_upper": or_ci_upper,
                    "se": se,
                    "z": z_val,
                    "p": p_val,
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "bic": float(getattr(fit, "bic", np.nan)),
                    "llf": float(getattr(fit, "llf", np.nan)),
                }
            )

    res = pd.DataFrame(results)
    if res.empty:
        return res

    res["q"] = np.nan
    for outcome in res["outcome"].unique():
        idx = res["outcome"] == outcome
        res.loc[idx, "q"] = _bh_fdr(res.loc[idx, "p"].to_numpy())

    res = res.sort_values(["outcome", "q", "p"], ascending=[True, True, True])
    return res


def compute_auc_regrouped_binary(
    merged: "pd.DataFrame",
    outcomes: Sequence[str],
    feature_cols: Optional[Sequence[str]],
    min_n: int,
) -> "pd.DataFrame":
    """Compute ROC AUC per outcome/feature for regrouped severity: 0-1 vs 2-3."""
    if not feature_cols:
        return pd.DataFrame()

    low_levels = {0, 1}
    high_levels = {2, 3}

    results: List[Dict[str, object]] = []
    for outcome in outcomes:
        y_full = pd.to_numeric(merged[outcome], errors="coerce")
        for feature in feature_cols:
            x_full = pd.to_numeric(merged[feature], errors="coerce")
            mask = x_full.notna() & y_full.notna()
            if int(mask.sum()) < min_n:
                continue

            df = pd.DataFrame({"x": x_full[mask].astype(float), "y": y_full[mask].astype(float)}).dropna()
            if df.shape[0] < min_n:
                continue

            df = df[df["y"].isin(low_levels | high_levels)].copy()
            if df.empty:
                continue

            df["y_bin"] = df["y"].isin(high_levels).astype(int)
            n_list = [int((df["y_bin"] == 0).sum()), int((df["y_bin"] == 1).sum())]
            n = int(df.shape[0])
            if n < min_n or n_list[0] < 2 or n_list[1] < 2:
                continue

            auc_ci_lower = np.nan
            auc_ci_upper = np.nan
            auc = np.nan
            p_val = np.nan
            try:
                u_stat, p_val = stats.mannwhitneyu(
                    df.loc[df["y_bin"] == 1, "x"].to_numpy(),
                    df.loc[df["y_bin"] == 0, "x"].to_numpy(),
                    alternative="two-sided",
                )
                auc = float(u_stat / (n_list[1] * n_list[0]))
                if auc < 0.5:
                    u_stat, p_val = stats.mannwhitneyu(
                        df.loc[df["y_bin"] == 0, "x"].to_numpy(),
                        df.loc[df["y_bin"] == 1, "x"].to_numpy(),
                        alternative="two-sided",
                    )
                    auc = float(u_stat / (n_list[1] * n_list[0]))
                    n_pos = n_list[0]
                    n_neg = n_list[1]
                else:
                    n_pos = n_list[1]
                    n_neg = n_list[0]

                # Calculate 95% CI for AUC using normal approximation
                if np.isfinite(auc) and auc > 0 and auc < 1:
                    se_auc = np.sqrt(auc * (1 - auc) / (n_pos * n_neg))
                    auc_ci_lower = np.clip(auc - 1.96 * se_auc, 0, 1)
                    auc_ci_upper = np.clip(auc + 1.96 * se_auc, 0, 1)
            except Exception:
                pass

            results.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "n": n,
                    "n_low": n_list[0],
                    "n_high": n_list[1],
                    "auc": auc,
                    "auc_ci_lower": auc_ci_lower,
                    "auc_ci_upper": auc_ci_upper,
                    "p": float(p_val) if pd.notna(p_val) else np.nan,
                }
            )

    res = pd.DataFrame(results)
    if res.empty:
        return res

    res["q"] = np.nan
    for outcome in res["outcome"].unique():
        idx = res["outcome"] == outcome
        res.loc[idx, "q"] = _bh_fdr(res.loc[idx, "p"].to_numpy())

    res["abs_auc_from_chance"] = (res["auc"] - 0.5).abs()
    res = res.sort_values(["outcome", "q", "abs_auc_from_chance"], ascending=[True, True, False])
    return res


def merge_results_tables(
    correlations: "pd.DataFrame",
    ordinal: "pd.DataFrame",
    auc: "pd.DataFrame",
) -> "pd.DataFrame":
    """Merge correlation, ordinal, and AUC results."""
    base_cols = ["outcome", "feature"]

    if correlations.empty:
        merged = pd.DataFrame(columns=base_cols)
    else:
        merged = correlations.rename(columns={"n": "n_corr"}).copy()

    if not ordinal.empty:
        ord_df = ordinal.rename(columns={"n": "n_ordinal", "q": "ordinal_q", "p": "ordinal_p"}).copy()
        merged = merged.merge(ord_df, on=base_cols, how="outer") if not merged.empty else ord_df

    if not auc.empty:
        auc_df = auc.rename(columns={"n": "n_auc", "q": "auc_q", "p": "auc_p"}).copy()
        merged = merged.merge(auc_df, on=base_cols, how="outer") if not merged.empty else auc_df

    if merged.empty:
        merged = pd.DataFrame(columns=["outcome", "feature"])

    return merged


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Correlate SpineNet IVD grades with expert Balgrist severity scores."
    )
    p.add_argument(
        "pred_dir",
        type=Path,
        help="Path to pred folder (contains subject sub-*/ivd_grades.csv and the readout CSV)",
    )
    p.add_argument(
        "--readout",
        type=Path,
        default=None,
        help="Optional path to Readout_lumbar_23112025.csv (default: <pred_dir>/Readout_lumbar_23112025.csv)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <pred_dir>/analysis_balgrist_spinenet_out)",
    )
    p.add_argument("--min-n", type=int, default=5, help="Minimum number of subjects required per correlation")
    p.add_argument("--top-k", type=int, default=10, help="Top correlations to export")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    pred_dir: Path = args.pred_dir
    if not pred_dir.is_dir():
        raise SystemExit(f"Pred directory does not exist: {pred_dir}")

    readout_csv: Path = args.readout or pred_dir / "Readout_lumbar_23112025.csv"
    if not readout_csv.is_file():
        raise SystemExit(f"Readout CSV not found: {readout_csv}")

    outdir: Path = args.outdir or pred_dir / "analysis_balgrist_spinenet_out"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SpineNet predictions from {pred_dir}")
    predictions = load_spinenet_predictions(pred_dir)
    if predictions.empty:
        raise SystemExit("No SpineNet predictions loaded")
    print(f"  Loaded {len(predictions)} disc-level predictions")

    print(f"Loading expert readout from {readout_csv}")
    readout = load_readout(readout_csv)
    print(f"  Loaded {len(readout)} readout records")

    # Merge predictions with readout by subject and level
    print("Merging predictions with expert readout...")
    merged = predictions.merge(readout, on=["subject", "Level"], how="inner")
    if merged.empty:
        raise SystemExit("No matching records after merge")
    print(f"  After merge: {len(merged)} records")

    # Save merged data
    merged_path = outdir / "merged_subject_level.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  Saved merged data to {merged_path}")

    # Select outcomes (expert grading columns we want to correlate with)
    print("Selecting outcome columns...")
    outcomes = ["spinal canal stenosis ALL", "foraminal stenosis ALL"]
    print(f"  Found outcomes: {outcomes}")

    # Add column spinenet_ForaminalStenosisSide
    side_dict = {"links": "Left", "rechts": "Right"}
    for idx, row in merged.iterrows():
        side = row["Side"]
        if side in side_dict:
            col_name = f"spinenet_ForaminalStenosis{side_dict[side]}"
            merged.loc[idx, "spinenet_ForaminalStenosisSide"] = row[col_name]

    # Feature columns (SpineNet predictions)
    feature_cols = ['spinenet_CentralCanalStenosis', 'spinenet_ForaminalStenosisSide']
    print(f"  Using features: {feature_cols}")

    # Compute correlations
    print("Computing correlations...")
    correlations = compute_correlations(merged, outcomes, feature_cols, args.min_n)
    if not correlations.empty:
        for outcome in correlations["outcome"].unique():
            corr_path = outdir / f"correlations__{_safe_col(outcome)}.csv"
            correlations[correlations["outcome"] == outcome].to_csv(corr_path, index=False)
            print(f"  Saved {corr_path}")

    # Compute ordinal logistic regression
    print("Computing ordinal logistic regression...")
    ordinal = compute_ordinal_logit(merged, outcomes, feature_cols, args.min_n)

    # Compute AUC
    print("Computing AUC for binary classification (0-1 vs 2-3)...")
    auc = compute_auc_regrouped_binary(merged, outcomes, feature_cols, args.min_n)

    # Merge all results
    print("Merging result tables...")
    results = merge_results_tables(correlations, ordinal, auc)
    if not results.empty:
        results_path = outdir / "all_results.csv"
        results.to_csv(results_path, index=False)
        print(f"  Saved {results_path}")

        # Save top correlations
        if "spearman_r" in results.columns:
            top_results = results.copy()
            top_results = top_results[top_results["spearman_r"].notna()]
            top_results["abs_spearman_r"] = top_results["spearman_r"].abs()
            top_results = top_results.sort_values(["abs_spearman_r", "spearman_q"], ascending=[False, True])
            top_results = top_results.head(args.top_k)
            top_path = outdir / "top_correlations.csv"
            top_results.to_csv(top_path, index=False)
            print(f"  Saved top {len(top_results)} correlations to {top_path}")

    print(f"\nAnalysis complete. Results saved to {outdir}")


if __name__ == "__main__":
    main()
