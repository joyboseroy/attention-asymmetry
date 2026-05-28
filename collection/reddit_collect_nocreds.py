"""
reddit_collect_nocreds.py
=========================
Collect Reddit posts for AI discourse amplification study.
No API credentials required — uses Reddit public JSON API.

Author: Joy Bose
Usage:  python3 reddit_collect_nocreds.py

Output: data/reddit_posts_clean.csv
        Prints paper-ready statistics at the end.
"""

import urllib.request
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

# ── SETTINGS ─────────────────────────────────────────────────────────────────

USER_AGENT  = "attention_asymmetry_research/1.0"
TIME_FILTER = "year"       # top posts from last year
LIMIT       = 100          # posts per subreddit
SLEEP       = 1.5          # seconds between requests (be polite)

OUTPUT_PATH = "data/reddit_posts_clean.csv"

# ── CORPUS DEFINITION ─────────────────────────────────────────────────────────
# Capital: communities where AI is framed as opportunity/transformation
# Labour:  communities where AI is framed as displacement/threat
# Subscriber counts used for size normalisation

CAPITAL_SUBS = {
    "MachineLearning":  3_000_000,
    "artificial":         800_000,
    "OpenAI":           1_500_000,
    "ChatGPT":          4_000_000,
    "singularity":        900_000,
}

LABOUR_SUBS = {
    "layoffs":            500_000,
    "cscareerquestions":  800_000,
    "ExperiencedDevs":    200_000,
    "antiwork":         2_800_000,
    "recruitinghell":     600_000,
}

# ── KEYWORDS ─────────────────────────────────────────────────────────────────
# Used to filter posts by relevance within each subreddit

AI_KEYWORDS = [
    "ai", "llm", "chatgpt", "gpt", "claude", "gemini", "openai", "anthropic",
    "automation", "layoff", "laid off", "job loss", "replaced", "fired",
    "artificial intelligence", "workforce", "job cut", "tech layoff",
    "machine learning", "copilot", "displacement", "reskilling", "upskilling",
    "model", "agent", "agi", "generative"
]

# ── AMPLIFICATION METRIC ─────────────────────────────────────────────────────
# Reddit equivalent of X amplification:
# score      = upvotes - downvotes  (reach within community)
# comments   * 1.5                  (active engagement, like quotes on X)
# crossposts * 2.0                  (cross-community spread, like retweets on X)

def amplification(score, comments, crossposts):
    return score + comments * 1.5 + crossposts * 2.0


# ── FETCH ─────────────────────────────────────────────────────────────────────

def fetch_json(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"    Fetch error: {e}")
        return {"data": {"children": []}}


def collect_subreddit(name, corpus, subscribers, limit=LIMIT):
    """
    Fetch top posts from a subreddit and filter for AI relevance.
    Uses subreddit top posts endpoint, not global search.
    """
    url = (
        f"https://www.reddit.com/r/{name}/top.json"
        f"?t={TIME_FILTER}&limit={limit}"
    )
    data   = fetch_json(url)
    posts  = data["data"]["children"]
    result = []

    for p in posts:
        d     = p["data"]
        title = d.get("title", "")
        body  = d.get("selftext", "")
        text  = (title + " " + body).lower()

        # Relevance filter
        if not any(kw in text for kw in AI_KEYWORDS):
            continue

        score      = d.get("score", 0)
        comments   = d.get("num_comments", 0)
        crossposts = d.get("num_crossposts", 0)
        amp        = amplification(score, comments, crossposts)

        # Size-normalised amplification
        # Divides by log(subscribers) same as ANI on X divides by log(followers)
        norm_amp = amp / np.log1p(subscribers)

        result.append({
            "post_id":       d["id"],
            "subreddit":     name,
            "corpus":        corpus,
            "subscribers":   subscribers,
            "title":         title[:200],
            "date":          datetime.utcfromtimestamp(
                                 d["created_utc"]
                             ).strftime("%Y-%m-%d"),
            "score":         score,
            "upvote_ratio":  d.get("upvote_ratio", 0),
            "num_comments":  comments,
            "num_crossposts": crossposts,
            "amplification": amp,
            "norm_amp":      norm_amp,
            "platform":      "reddit",
        })

    print(
        f"  r/{name:<22} [{corpus}]"
        f"  {len(result):3d} relevant"
        f"  from {len(posts):3d} top posts"
        f"  (subscribers: {subscribers:,})"
    )
    time.sleep(SLEEP)
    return result


# ── MAIN COLLECTION ───────────────────────────────────────────────────────────

def main():
    Path("data").mkdir(exist_ok=True)

    print("=" * 65)
    print("REDDIT CROSS-PLATFORM REPLICATION")
    print("AI Discourse Amplification Study — No-Credentials Collection")
    print("=" * 65)

    all_posts = []

    print("\nCapital subreddits (AI as opportunity)...")
    for name, subs in CAPITAL_SUBS.items():
        all_posts.extend(collect_subreddit(name, "capital", subs))

    print("\nLabour subreddits (AI as displacement)...")
    for name, subs in LABOUR_SUBS.items():
        all_posts.extend(collect_subreddit(name, "labour", subs))

    df = pd.DataFrame(all_posts)

    # Deduplicate by post ID
    before = len(df)
    df = df.drop_duplicates(subset=["post_id"])
    dupes = before - len(df)

    print(f"\nCollected: {before} posts")
    print(f"Duplicates removed: {dupes}")
    print(f"Final dataset: {len(df)} posts")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    # ── STATISTICS ────────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("PER SUBREDDIT BREAKDOWN")
    print("=" * 65)
    breakdown = df.groupby(["corpus", "subreddit"]).agg(
        count    = ("amplification", "count"),
        mean_amp = ("amplification", "mean"),
        median   = ("amplification", "median"),
        subs     = ("subscribers", "first"),
    ).round(1)
    print(breakdown.to_string())

    print("\n" + "=" * 65)
    print("RAW AMPLIFICATION")
    print("=" * 65)

    cap_raw = df[df.corpus == "capital"]["amplification"].values
    lab_raw = df[df.corpus == "labour"]["amplification"].values

    u1, p1 = stats.mannwhitneyu(cap_raw, lab_raw, alternative="greater")
    d1 = (cap_raw.mean() - lab_raw.mean()) / np.sqrt(
        (cap_raw.std()**2 + lab_raw.std()**2) / 2
    )
    ratios1 = [
        np.random.choice(cap_raw, len(cap_raw), replace=True).mean() /
        np.random.choice(lab_raw, len(lab_raw), replace=True).mean()
        for _ in range(1000)
    ]
    ci1 = np.percentile(ratios1, [2.5, 97.5])

    print(f"n capital:      {len(cap_raw)}")
    print(f"n labour:       {len(lab_raw)}")
    print(f"Capital mean:   {cap_raw.mean():.2f}")
    print(f"Labour mean:    {lab_raw.mean():.2f}")
    print(f"Ratio mean:     {cap_raw.mean()/lab_raw.mean():.3f}x")
    print(f"Ratio median:   {np.median(cap_raw)/np.median(lab_raw):.3f}x")
    print(f"Mann-Whitney U: {u1:.1f}")
    print(f"p-value:        {p1:.6f}")
    print(f"Cohen d:        {d1:.3f}")
    print(f"95% CI:         [{ci1[0]:.3f}, {ci1[1]:.3f}]")

    print("\n" + "=" * 65)
    print("SUBSCRIBER-NORMALISED AMPLIFICATION (norm_amp)")
    print("Controls for subreddit size — analogous to ANI on X")
    print("=" * 65)

    cap_norm = df[df.corpus == "capital"]["norm_amp"].values
    lab_norm = df[df.corpus == "labour"]["norm_amp"].values

    u2, p2 = stats.mannwhitneyu(cap_norm, lab_norm, alternative="greater")
    d2 = (cap_norm.mean() - lab_norm.mean()) / np.sqrt(
        (cap_norm.std()**2 + lab_norm.std()**2) / 2
    )
    ratios2 = [
        np.random.choice(cap_norm, len(cap_norm), replace=True).mean() /
        np.random.choice(lab_norm, len(lab_norm), replace=True).mean()
        for _ in range(1000)
    ]
    ci2 = np.percentile(ratios2, [2.5, 97.5])

    print(f"Capital norm mean:  {cap_norm.mean():.4f}")
    print(f"Labour norm mean:   {lab_norm.mean():.4f}")
    print(f"Ratio mean:         {cap_norm.mean()/lab_norm.mean():.3f}x")
    print(f"Ratio median:       {np.median(cap_norm)/np.median(lab_norm):.3f}x")
    print(f"Mann-Whitney U:     {u2:.1f}")
    print(f"p-value:            {p2:.6f}")
    print(f"Cohen d:            {d2:.3f}")
    print(f"95% CI:             [{ci2[0]:.3f}, {ci2[1]:.3f}]")

    print("\n" + "=" * 65)
    print("SUMMARY FOR PAPER")
    print("=" * 65)
    print(f"Raw ratio:        {cap_raw.mean()/lab_raw.mean():.3f}x  (p={p1:.6f})")
    print(f"Normalised ratio: {cap_norm.mean()/lab_norm.mean():.3f}x  (p={p2:.6f})")
    print()
    if p2 < 0.05:
        print("Finding: asymmetry REPLICATES on Reddit after size normalisation.")
        print("Add to paper as cross-platform confirmation.")
    elif p1 < 0.05:
        print("Finding: raw asymmetry significant but disappears after normalisation.")
        print("Report as partial replication — subreddit size drives raw effect.")
    else:
        print("Finding: no significant asymmetry on Reddit.")
        print("Report as non-replication with methodological explanation.")

    return df


if __name__ == "__main__":
    main()
