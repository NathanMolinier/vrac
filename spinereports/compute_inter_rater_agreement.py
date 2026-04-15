import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import contingency

def cohens_kappa(rater1, rater2):
    """
    Calculate Cohen's kappa coefficient for inter-rater agreement.
    Handles missing values (NaN).
    """
    # Remove NaN values
    mask = (~np.isnan(rater1)) & (~np.isnan(rater2))
    r1 = rater1[mask]
    r2 = rater2[mask]
    
    if len(r1) == 0:
        return np.nan, 0
    
    # Calculate observed agreement
    agreement = np.sum(r1 == r2) / len(r1)
    
    # Calculate expected agreement by chance
    categories = np.unique(np.concatenate([r1, r2]))
    pe = 0
    for cat in categories:
        p1 = np.sum(r1 == cat) / len(r1)
        p2 = np.sum(r2 == cat) / len(r2)
        pe += p1 * p2
    
    # Calculate Cohen's kappa
    if pe == 1:
        kappa = np.nan
    else:
        kappa = (agreement - pe) / (1 - pe)
    
    return kappa, len(r1)

def percent_agreement(rater1, rater2):
    """Calculate percent agreement between two raters."""
    mask = (~np.isnan(rater1)) & (~np.isnan(rater2))
    r1 = rater1[mask]
    r2 = rater2[mask]
    
    if len(r1) == 0:
        return np.nan, 0
    
    agreement = np.sum(r1 == r2) / len(r1) * 100
    return agreement, len(r1)

def intraclass_correlation(rater1, rater2):
    """
    Calculate ICC(1,1) - Intraclass Correlation Coefficient.
    One-way random effects model.
    """
    mask = (~np.isnan(rater1)) & (~np.isnan(rater2))
    r1 = rater1[mask]
    r2 = rater2[mask]
    
    if len(r1) < 2:
        return np.nan
    
    n = len(r1)
    k = 2  # two raters
    
    mean_subj = (r1 + r2) / 2
    grand_mean = np.mean(mean_subj)
    
    SSB = k * np.sum((mean_subj - grand_mean)**2)
    MSB = SSB / (n - 1)
    
    SSW = np.sum((r1 - mean_subj)**2 + (r2 - mean_subj)**2)
    MSW = SSW / n
    
    if (MSB + (k - 1) * MSW) == 0:
        return np.nan
    
    icc = (MSB - MSW) / (MSB + (k - 1) * MSW)
    return icc

def main():
    # Load the CSV
    csv_path = Path("/home/ge.polymtl.ca/p118739/code/vrac/Readout_lumbar_23112025.csv")
    df = pd.read_csv(csv_path)
    
    # Define the three pathologies and their column names
    pathologies = {
        'Foraminal Stenosis': ('foraminal stenosis_READER 1 (Senior)', 'foraminal stenosis_READER 2 (Junior)'),
        'Recessal Stenosis': ('recessal stenosis_READER 1 (Senior)', 'recessal stenosis_READER 2 (Junior)'),
        'Spinal Canal Stenosis': ('spinal canal stenosis_READER 1 (Senior)', 'spinal canal stenosis_READER 2 (Junior)')
    }
    
    # Store results
    results = []
    
    print("=" * 80)
    print("INTER-RATER AGREEMENT ANALYSIS")
    print("=" * 80)
    print()
    
    # Compute agreement for each pathology
    for pathology_name, (col_r1, col_r2) in pathologies.items():
        # Convert to numeric, replacing non-numeric with NaN
        r1 = pd.to_numeric(df[col_r1], errors='coerce').values
        r2 = pd.to_numeric(df[col_r2], errors='coerce').values
        
        # Calculate metrics
        kappa, n_kappa = cohens_kappa(r1, r2)
        pct_agree, n_pct = percent_agreement(r1, r2)
        icc = intraclass_correlation(r1, r2)
        
        results.append({
            'Pathology': pathology_name,
            'N_Pairs': n_kappa,
            'Cohens_Kappa': kappa,
            'Percent_Agreement': pct_agree,
            'ICC': icc
        })
        
        print(f"{pathology_name}")
        print("-" * 80)
        print(f"  Number of paired observations: {n_kappa}")
        print(f"  Cohen's Kappa: {kappa:.4f}" + (" (Fair agreement)" if 0.21 <= kappa < 0.41 
                                                 else " (Moderate agreement)" if 0.41 <= kappa < 0.61
                                                 else " (Substantial agreement)" if 0.61 <= kappa < 0.81
                                                 else " (Perfect agreement)" if kappa >= 0.81
                                                 else " (Slight agreement)" if kappa >= 0.01
                                                 else ""))
        print(f"  Percent Agreement: {pct_agree:.2f}%")
        print(f"  ICC(1,1): {icc:.4f}")
        print()
    
    # Compute overall agreement (across all pathologies)
    all_r1 = []
    all_r2 = []
    for pathology_name, (col_r1, col_r2) in pathologies.items():
        r1 = pd.to_numeric(df[col_r1], errors='coerce').values
        r2 = pd.to_numeric(df[col_r2], errors='coerce').values
        all_r1.extend(r1[~np.isnan(r1) & ~np.isnan(r2)])
        all_r2.extend(r2[~np.isnan(r1) & ~np.isnan(r2)])
    
    all_r1 = np.array(all_r1)
    all_r2 = np.array(all_r2)
    
    overall_kappa, n_overall = cohens_kappa(all_r1, all_r2)
    overall_pct_agree, _ = percent_agreement(all_r1, all_r2)
    overall_icc = intraclass_correlation(all_r1, all_r2)
    
    results.append({
        'Pathology': 'Overall',
        'N_Pairs': n_overall,
        'Cohens_Kappa': overall_kappa,
        'Percent_Agreement': overall_pct_agree,
        'ICC': overall_icc
    })
    
    print("OVERALL AGREEMENT (All Pathologies Combined)")
    print("-" * 80)
    print(f"  Number of paired observations: {n_overall}")
    print(f"  Cohen's Kappa: {overall_kappa:.4f}" + (" (Fair agreement)" if 0.21 <= overall_kappa < 0.41 
                                                     else " (Moderate agreement)" if 0.41 <= overall_kappa < 0.61
                                                     else " (Substantial agreement)" if 0.61 <= overall_kappa < 0.81
                                                     else " (Perfect agreement)" if overall_kappa >= 0.81
                                                     else " (Slight agreement)" if overall_kappa >= 0.01
                                                     else ""))
    print(f"  Percent Agreement: {overall_pct_agree:.2f}%")
    print(f"  ICC(1,1): {overall_icc:.4f}")
    print()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = Path("/home/ge.polymtl.ca/p118739/code/vrac").parent / "inter_rater_agreement_report.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print()
    
    # Print interpretation guide
    print("=" * 80)
    print("INTERPRETATION GUIDE FOR COHEN'S KAPPA:")
    print("=" * 80)
    print("  < 0.00: Poor agreement")
    print("  0.00-0.20: Slight agreement")
    print("  0.21-0.40: Fair agreement")
    print("  0.41-0.60: Moderate agreement")
    print("  0.61-0.80: Substantial agreement")
    print("  0.81-1.00: Perfect agreement")
    print()

if __name__ == "__main__":
    main()
