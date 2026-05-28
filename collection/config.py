"""
Attention Asymmetry Study — Configuration
==========================================
Paper: "Whose Voice Gets Amplified? Attention Asymmetry in AI Layoff Discourse on X"

All parameters in one place. Edit here, nothing else needs changing.
"""

# ── Tweet collection keywords ────────────────────────────────────────────────

# Capital discourse: optimism, opportunity, transformation
CAPITAL_QUERIES = [
    "AI will create jobs",
    "AI creates opportunities",
    "AI productivity gains",
    "AI transformation opportunity",
    "AI will augment workers",
    "future of work AI opportunity",
    "AI reskilling upskilling",
    "AI empowers workers",
]

# Labour discourse: fear, loss, displacement
LABOUR_QUERIES = [
    "laid off because of AI",
    "AI took my job",
    "replaced by AI",
    "lost job to AI",
    "AI eliminated my role",
    "job cut AI automation",
    "AI layoff personal",
    "fired replaced automation",
]

# ── Major layoff events (ground truth anchor points) ─────────────────────────
# Source: layoffs.fyi — use these for temporal windowing in Paper 2
LAYOFF_EVENTS = [
    {"company": "Meta",       "date": "2022-11-09", "count": 11000},
    {"company": "Amazon",     "date": "2023-01-04", "count": 18000},
    {"company": "Google",     "date": "2023-01-20", "count": 12000},
    {"company": "Microsoft",  "date": "2023-01-18", "count": 10000},
    {"company": "Salesforce", "date": "2023-01-04", "count": 8000},
    {"company": "Meta",       "date": "2023-03-14", "count": 10000},
    {"company": "Amazon",     "date": "2023-03-20", "count": 9000},
    {"company": "Google",     "date": "2024-01-10", "count": 1000},
    {"company": "Microsoft",  "date": "2024-01-25", "count": 1900},
    {"company": "Infosys",    "date": "2023-04-13", "count": 1000},  # India angle
]

# ── Amplification metrics to collect per tweet ───────────────────────────────
AMPLIFICATION_METRICS = ["likes", "retweets", "replies", "quotes", "views"]

# ── Simulation parameters (for mock data pipeline) ───────────────────────────
# Real collection: replace with twscrape or X API v2
MOCK_TWEET_COUNT = 500   # per corpus (capital / labour)
RANDOM_SEED = 42

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR    = "data/"
OUTPUT_DIR  = "outputs/"
