"""
analyse.py — Statistical Analysis Pipeline
===========================================
Computes all metrics needed for the paper:
  1. Amplification Ratio (AR) — core finding
  2. Distribution comparison (Mann-Whitney U, Cohen's d)
  3. Verified vs unverified breakdown
  4. Temporal trends
  5. Per-query breakdown
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from config import DATA_DIR, OUTPUT_DIR

def load_data() -> pd.DataFrame:
    path = f"{DATA_DIR}tweets_raw.csv"
    if not Path(path).exists():
        raise FileNotFoundError(f"No data found at {path}. Run collect.py first.")
    return pd.read_csv(path)

# ── CORE METRIC 1: Amplification Ratio ───────────────────────────────────────

def compute_amplification_ratio(df: pd.DataFrame) -> dict:
    """
    AR = mean amplification (capital corpus) / mean amplification (labour corpus)
    
    AR > 1 means optimistic AI discourse gets more reach.
    AR < 1 would mean worker grief gets more reach (unlikely but possible).
    
    We compute three versions:
      - Raw (absolute engagement)
      - Normalised (controlled for follower count)
      - Median-based (robust to outliers / viral tweets)
    """
    cap = df[df.corpus == "capital"]
    lab = df[df.corpus == "labour"]
    
    results = {}
    
    for metric in ["amplification", "normalised_amp", "likes", "retweets"]:
        cap_mean   = cap[metric].mean()
        lab_mean   = lab[metric].mean()
        cap_median = cap[metric].median()
        lab_median = lab[metric].median()
        
        results[metric] = {
            "capital_mean":   round(cap_mean, 2),
            "labour_mean":    round(lab_mean, 2),
            "ratio_mean":     round(cap_mean / lab_mean, 3) if lab_mean > 0 else None,
            "capital_median": round(cap_median, 2),
            "labour_median":  round(lab_median, 2),
            "ratio_median":   round(cap_median / lab_median, 3) if lab_median > 0 else None,
        }
    
    return results

# ── CORE METRIC 2: Statistical Significance ──────────────────────────────────

def compute_significance(df: pd.DataFrame) -> dict:
    """
    Mann-Whitney U test: non-parametric, appropriate for skewed engagement data.
    Cohen's d: effect size.
    """
    cap_amp = df[df.corpus == "capital"]["amplification"].values
    lab_amp = df[df.corpus == "labour"]["amplification"].values
    
    # Mann-Whitney U (does not assume normality)
    u_stat, p_value = stats.mannwhitneyu(cap_amp, lab_amp, alternative="greater")
    
    # Cohen's d effect size
    pooled_std = np.sqrt((cap_amp.std()**2 + lab_amp.std()**2) / 2)
    cohens_d   = (cap_amp.mean() - lab_amp.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Bootstrap confidence interval for the ratio
    n_bootstrap = 1000
    ratios = []
    for _ in range(n_bootstrap):
        c_sample = np.random.choice(cap_amp, size=len(cap_amp), replace=True)
        l_sample = np.random.choice(lab_amp, size=len(lab_amp), replace=True)
        if l_sample.mean() > 0:
            ratios.append(c_sample.mean() / l_sample.mean())
    
    ci_low, ci_high = np.percentile(ratios, [2.5, 97.5])
    
    return {
        "mann_whitney_u":   round(u_stat, 2),
        "p_value":          round(p_value, 6),
        "significant":      p_value < 0.05,
        "cohens_d":         round(cohens_d, 3),
        "effect_size":      "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
        "bootstrap_ci_low":  round(ci_low, 3),
        "bootstrap_ci_high": round(ci_high, 3),
    }

# ── CORE METRIC 3: Verified Account Breakdown ────────────────────────────────

def compute_verified_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Is the asymmetry driven by verified (institutional) accounts
    or does it hold for regular users too?
    This is a critical control — if only verified accounts drive it,
    the finding is about institutional amplification, not organic reach.
    """
    return df.groupby(["corpus", "is_verified"])["amplification"].agg(
        count="count",
        mean="mean",
        median="median",
        std="std"
    ).round(2)

# ── CORE METRIC 4: Temporal Trends ───────────────────────────────────────────

def compute_temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly mean amplification per corpus.
    Plot this to see if asymmetry widens after major layoff events.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    
    monthly = df.groupby(["month", "corpus"])["amplification"].mean().unstack()
    monthly.columns.name = None
    
    if "capital" in monthly.columns and "labour" in monthly.columns:
        monthly["ratio"] = (monthly["capital"] / monthly["labour"]).round(3)
    
    return monthly.reset_index()

# ── CORE METRIC 5: Per-Query Breakdown ───────────────────────────────────────

def compute_per_query(df: pd.DataFrame) -> pd.DataFrame:
    """
    Which specific queries drive the asymmetry?
    Helps identify the most potent narratives on each side.
    """
    return df.groupby(["corpus", "query"])["amplification"].agg(
        count="count",
        mean="mean",
        median="median"
    ).sort_values("mean", ascending=False).round(2)

# ── SUMMARY TABLE (paper-ready) ───────────────────────────────────────────────

def generate_summary(df: pd.DataFrame) -> dict:
    ar     = compute_amplification_ratio(df)
    sig    = compute_significance(df)
    ver    = compute_verified_breakdown(df)
    temp   = compute_temporal_trends(df)
    query  = compute_per_query(df)
    
    summary = {
        "amplification_ratios": ar,
        "significance":         sig,
        "verified_breakdown":   ver.to_dict(),
        "temporal":             temp.to_dict(orient="records"),
        "per_query":            query.to_dict(),
    }
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Print paper-ready summary
    print("\n" + "="*60)
    print("ATTENTION ASYMMETRY — CORE FINDINGS")
    print("="*60)
    
    amp = ar["amplification"]
    print(f"\nAmplification Ratio (mean):   {amp['ratio_mean']}x")
    print(f"Amplification Ratio (median): {amp['ratio_median']}x")
    print(f"\nCapital corpus mean: {amp['capital_mean']}")
    print(f"Labour corpus mean:  {amp['labour_mean']}")
    
    print(f"\nStatistical significance:")
    print(f"  Mann-Whitney U p-value: {sig['p_value']}")
    print(f"  Significant (p<0.05):   {sig['significant']}")
    print(f"  Cohen's d:              {sig['cohens_d']} ({sig['effect_size']} effect)")
    print(f"  95% CI on ratio:        [{sig['bootstrap_ci_low']}, {sig['bootstrap_ci_high']}]")
    
    norm = ar["normalised_amp"]
    print(f"\nNormalised ratio (follower-adjusted):")
    print(f"  {norm['ratio_mean']}x (mean), {norm['ratio_median']}x (median)")
    
    print("\nVerified account breakdown:")
    print(ver.to_string())
    
    return summary


if __name__ == "__main__":
    df = load_data()
    summary = generate_summary(df)
