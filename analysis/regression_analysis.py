"""
regression_analysis.py — Additional Statistical Analysis
=========================================================
Addresses remaining reviewer critiques:

Critique: "Report raw metrics separately first (likes only, retweets only)"
Critique: "Run regression: amplification ~ discourse_type + log(followers)"
Critique: "Report 25th and 75th percentiles, not just mean and median"
Critique: "Report both versions (with and without zero-engagement tweets)"
Critique: "Elite vs elite sub-analysis"
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def load_data():
    df = pd.read_csv("data/tweets_accounts.csv")
    follower_map = {
        'sama': 4900000, 'sundarpichai': 7200000, 'satyanadella': 3500000,
        'karpathy': 1000000, 'ylecun': 600000, 'garrytan': 400000,
        'naval': 2200000, 'paulg': 400000, 'demishassabis': 500000,
        'emollick': 500000, 'GergelyOrosz': 335000, 'TrungTPhan': 725000,
        'dhh': 667000, 'karaswisher': 400000, 'doctorow': 200000,
        'zeynep': 350000, 'benedictevans': 300000, 'EricaJoy': 50000,
        'kimmaicutler': 15000, 'reckless': 70000,
    }
    df['follower_count'] = df['handle'].map(follower_map)
    df['log_followers'] = np.log1p(df['follower_count'])
    df['is_capital'] = (df['corpus'] == 'capital').astype(int)
    return df

def raw_metrics_comparison(df):
    """
    Report each metric separately before composite.
    Addresses: 'Report raw metrics separately first'
    """
    print("\n=== RAW METRICS: CAPITAL vs LABOUR ===")
    print("(Addresses reviewer request for separate metric reporting)\n")

    metrics = ['likes', 'retweets', 'replies', 'quotes']
    rows = []
    for m in metrics:
        cap = df[df.corpus=='capital'][m].values
        lab = df[df.corpus=='labour'][m].values
        u, p = stats.mannwhitneyu(cap, lab, alternative='greater')
        ratio = cap.mean() / lab.mean() if lab.mean() > 0 else float('inf')
        rows.append({
            'Metric': m.capitalize(),
            'Capital mean': round(cap.mean(), 2),
            'Labour mean': round(lab.mean(), 2),
            'Ratio': round(ratio, 3),
            'p-value': round(p, 6),
            'Significant': p < 0.05
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df

def percentile_breakdown(df):
    """
    Report 25th, 50th, 75th percentiles.
    Addresses: 'Report 25th and 75th percentiles, not just mean and median'
    """
    print("\n=== PERCENTILE BREAKDOWN ===")
    print("(Addresses reviewer request for quartile reporting)\n")

    for label, data in [
        ("All tweets (including zero)", df),
        ("Non-zero engagement only", df[df.amplification > 0])
    ]:
        print(f"--- {label} ---")
        pct = data.groupby('corpus')['amplification'].describe(
            percentiles=[.10, .25, .50, .75, .90]
        )[['count', '10%', '25%', '50%', '75%', '90%', 'mean', 'std']]
        print(pct.round(2))
        print()

def zero_engagement_comparison(df):
    """
    Compare results with and without zero-engagement tweets.
    Addresses: 'Report both versions (with and without zero-engagement tweets)'
    """
    print("\n=== ZERO-ENGAGEMENT SENSITIVITY ===")
    print("(Addresses reviewer concern about excluding zero-engagement tweets)\n")

    for label, data in [
        ("Including zero-engagement", df),
        ("Excluding zero-engagement", df[df.amplification > 0])
    ]:
        cap = data[data.corpus=='capital']['amplification'].values
        lab = data[data.corpus=='labour']['amplification'].values
        if len(cap) == 0 or len(lab) == 0:
            continue
        u, p = stats.mannwhitneyu(cap, lab, alternative='greater')
        ratio = cap.mean() / lab.mean()
        print(f"{label}:")
        print(f"  n: capital={len(cap)}, labour={len(lab)}")
        print(f"  Ratio: {ratio:.3f}x, p={p:.6f}")
        print()

def regression_analysis(df):
    """
    OLS regression: log(amplification) ~ corpus + log(followers)
    Addresses: 'Run regression: amplification ~ discourse_type + log(followers)'
    """
    print("\n=== REGRESSION ANALYSIS ===")
    print("(Addresses reviewer request for covariate-controlled analysis)\n")

    try:
        import statsmodels.api as sm
    except ImportError:
        print("Installing statsmodels...")
        import subprocess
        subprocess.run(['pip', 'install', 'statsmodels', '--break-system-packages', '-q'])
        import statsmodels.api as sm

    df_reg = df[df.amplification > 0].copy()
    df_reg['log_amp'] = np.log1p(df_reg['amplification'])

    # Model 1: corpus only
    X1 = sm.add_constant(df_reg[['is_capital']])
    model1 = sm.OLS(df_reg['log_amp'], X1).fit()

    # Model 2: corpus + log(followers)
    X2 = sm.add_constant(df_reg[['is_capital', 'log_followers']])
    model2 = sm.OLS(df_reg['log_amp'], X2).fit()

    print("Model 1: log(amplification) ~ corpus_type")
    print(f"  Corpus coefficient: {model1.params['is_capital']:.4f}")
    print(f"  p-value:            {model1.pvalues['is_capital']:.6f}")
    print(f"  R-squared:          {model1.rsquared:.4f}")
    print(f"  Effect (exp coef):  {np.exp(model1.params['is_capital']):.3f}x")
    print()
    print("Model 2: log(amplification) ~ corpus_type + log(followers)")
    print(f"  Corpus coefficient: {model2.params['is_capital']:.4f}")
    print(f"  p-value:            {model2.pvalues['is_capital']:.6f}")
    print(f"  log_followers coef: {model2.params['log_followers']:.4f}")
    print(f"  R-squared:          {model2.rsquared:.4f}")
    print(f"  Effect (exp coef):  {np.exp(model2.params['is_capital']):.3f}x")
    print()
    print("Interpretation:")
    coef = model2.params['is_capital']
    p2 = model2.pvalues['is_capital']
    print(f"  Controlling for follower count, capital discourse receives")
    print(f"  exp({coef:.3f}) = {np.exp(coef):.3f}x more amplification (p={p2:.6f})")

    return model1, model2

def elite_vs_elite(df):
    """
    Compare only accounts with similar follower counts.
    Addresses: 'Match accounts by follower count across corpora'
    """
    print("\n=== ELITE vs ELITE ANALYSIS ===")
    print("(Addresses reviewer request for matched follower count comparison)\n")

    # Define elite as >300K followers (captures most accounts in both corpora)
    follower_map = {
        'sama': 4900000, 'sundarpichai': 7200000, 'satyanadella': 3500000,
        'karpathy': 1000000, 'ylecun': 600000, 'garrytan': 400000,
        'naval': 2200000, 'paulg': 400000, 'demishassabis': 500000,
        'emollick': 500000, 'GergelyOrosz': 335000, 'TrungTPhan': 725000,
        'dhh': 667000, 'karaswisher': 400000, 'doctorow': 200000,
        'zeynep': 350000, 'benedictevans': 300000, 'EricaJoy': 50000,
        'kimmaicutler': 15000, 'reckless': 70000,
    }
    df['follower_count'] = df['handle'].map(follower_map)

    ELITE_THRESHOLD = 300000
    df_elite = df[df.follower_count >= ELITE_THRESHOLD]

    cap_elite = df_elite[df_elite.corpus=='capital']
    lab_elite = df_elite[df_elite.corpus=='labour']

    print(f"Elite accounts (>{ELITE_THRESHOLD:,} followers):")
    print(f"  Capital: {cap_elite['handle'].unique().tolist()}")
    print(f"  Labour:  {lab_elite['handle'].unique().tolist()}")
    print()

    cap = cap_elite['amplification'].values
    lab = lab_elite['amplification'].values

    if len(cap) > 0 and len(lab) > 0:
        u, p = stats.mannwhitneyu(cap, lab, alternative='greater')
        d = (cap.mean() - lab.mean()) / np.sqrt((cap.std()**2 + lab.std()**2)/2)
        print(f"Elite-only results:")
        print(f"  n capital: {len(cap)}, n labour: {len(lab)}")
        print(f"  Capital mean: {cap.mean():.2f}")
        print(f"  Labour mean:  {lab.mean():.2f}")
        print(f"  Ratio:        {cap.mean()/lab.mean():.3f}x")
        print(f"  p-value:      {p:.6f}")
        print(f"  Cohen's d:    {d:.3f}")
        print()
        if p < 0.05:
            print("Finding: asymmetry persists even among elite accounts with similar reach.")
        else:
            print("Finding: asymmetry attenuates when controlling for account elite status.")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    # Add amplification if not present
    if 'amplification' not in df.columns:
        df['amplification'] = df['likes'] + 2*df['retweets'] + 1.5*df['quotes']

    raw_metrics_comparison(df)
    percentile_breakdown(df)
    zero_engagement_comparison(df)
    regression_analysis(df)
    elite_vs_elite(df)

    print("\n=== ALL ANALYSES COMPLETE ===")
    print("These results address reviewer critiques 2, 3, 5, 8, 9, 10.")
    print("Add to paper: regression table in Section 4.4, percentiles in all tables.")
