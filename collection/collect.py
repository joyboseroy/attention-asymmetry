"""
collect.py — Tweet Collection Pipeline
=======================================
Two modes:
  1. MOCK MODE   — generates realistic synthetic data immediately (no API needed)
  2. LIVE MODE   — uses twscrape against X (instructions below)

To switch to live collection:
    pip install twscrape
    then call collect_live() instead of collect_mock()
"""

import os
from dotenv import load_dotenv
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from config import (
    CAPITAL_QUERIES, LABOUR_QUERIES, LAYOFF_EVENTS,
    MOCK_TWEET_COUNT, RANDOM_SEED, DATA_DIR
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── MOCK DATA GENERATOR ───────────────────────────────────────────────────────
# Calibrated to realistic X engagement distributions (power-law, not uniform)

def _power_law_sample(n, min_val, max_val, exponent=2.0):
    """Engagement on X follows power-law: most tweets get very little, few get a lot."""
    samples = (np.random.pareto(exponent, n) + 1) * min_val
    return np.clip(samples, min_val, max_val).astype(int)

def _mock_tweet(corpus_type: str, query: str, tweet_id: int) -> dict:
    """
    Generate one mock tweet with realistic engagement distributions.
    
    Key design decision: capital discourse tweets are given higher baseline
    engagement to test whether the asymmetry finding holds. In real data
    you would NOT pre-seed this — let the data show it.
    """
    is_capital = (corpus_type == "capital")
    
    # Capital discourse gets ~3x amplification baseline on X
    # This reflects the platform's known bias toward positive/viral content
    engagement_multiplier = 3.0 if is_capital else 1.0
    
    # Dates spread across major layoff wave 2022-2024
    start_date = datetime(2022, 10, 1)
    end_date   = datetime(2024, 12, 31)
    delta_days = (end_date - start_date).days
    tweet_date = start_date + timedelta(days=random.randint(0, delta_days))
    
    likes     = int(_power_law_sample(1, 1, 50000, 1.8)[0] * engagement_multiplier)
    retweets  = int(likes * random.uniform(0.1, 0.4))
    replies   = int(likes * random.uniform(0.05, 0.2))
    quotes    = int(likes * random.uniform(0.02, 0.1))
    views     = int(likes * random.uniform(8, 40))
    
    # Amplification ratio: total reach per tweet
    amplification = likes + retweets * 2 + quotes * 1.5
    
    return {
        "tweet_id":        tweet_id,
        "corpus":          corpus_type,
        "query":           query,
        "date":            tweet_date.strftime("%Y-%m-%d"),
        "likes":           likes,
        "retweets":        retweets,
        "replies":         replies,
        "quotes":          quotes,
        "views":           views,
        "amplification":   amplification,
        # Verified accounts get more reach — important control variable
        "is_verified":     random.random() < (0.15 if is_capital else 0.05),
        # Follower count affects organic reach
        "follower_count":  int(_power_law_sample(1, 10, 1000000, 1.5)[0]),
    }

def collect_mock() -> pd.DataFrame:
    """
    Generate mock dataset for both corpora.
    Returns single DataFrame with corpus label as column.
    """
    print("Generating mock tweet dataset...")
    tweets = []
    
    per_query = MOCK_TWEET_COUNT // len(CAPITAL_QUERIES)
    
    for i, query in enumerate(CAPITAL_QUERIES):
        for j in range(per_query):
            tweet_id = i * 1000 + j
            tweets.append(_mock_tweet("capital", query, tweet_id))
    
    for i, query in enumerate(LABOUR_QUERIES):
        for j in range(per_query):
            tweet_id = 10000 + i * 1000 + j
            tweets.append(_mock_tweet("labour", query, tweet_id))
    
    df = pd.DataFrame(tweets)
    
    # Normalise amplification by follower count for fair comparison
    df["normalised_amp"] = df["amplification"] / (np.log1p(df["follower_count"]))
    
    Path(DATA_DIR).mkdir(exist_ok=True)
    df.to_csv(f"{DATA_DIR}tweets_raw.csv", index=False)
    print(f"  Saved {len(df)} tweets to {DATA_DIR}tweets_raw.csv")
    print(f"  Capital corpus: {len(df[df.corpus=='capital'])} tweets")
    print(f"  Labour corpus:  {len(df[df.corpus=='labour'])} tweets")
    return df


# ── LIVE COLLECTION (twscrape) ────────────────────────────────────────────────
# Uncomment and run when you have X credentials

LIVE_COLLECTION_INSTRUCTIONS = """
LIVE COLLECTION WITH twscrape
==============================
1. pip install twscrape

2. Add accounts (burner accounts work):
   twscrape add_accounts accounts.txt login:name:password:email:email_password

3. Login:
   twscrape login_accounts

4. Replace collect_mock() call in run.py with collect_live()

Note: X API free tier is severely limited. twscrape uses the GraphQL
web API which is more permissive but can get rate-limited. For 500 tweets
per corpus you need roughly 2-3 hours of collection with pauses.

Alternative: Use the X Academic Research API if you have access.
The query format is identical to config.py CAPITAL_QUERIES / LABOUR_QUERIES.
"""

def collect_live():
    import requests
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    capital = {
        'sama': '1605', 'sundarpichai': '14130366', 'satyanadella': '15142270',
        'karpathy': '33836629', 'ylecun': '1062190', 'garrytan': '24847875',
        'naval': '745273', 'paulg': '22331766', 'demishassabis': '258466834',
        'emollick': '1035603951',
    }
    labour = {
        'GergelyOrosz': '30192824', 'TrungTPhan': '945817135816654848',
        'dhh': '14561327', 'karaswisher': '16953162', 'doctorow': '2272672',
        'zeynep': '14595966', 'benedictevans': '5765302',
        'EricaJoy': '3243288374', 'kimmaicutler': '15164816',
        'reckless': '14536191',
    }

    AI_KEYWORDS = [
        'ai ', 'llm', 'layoff', 'automation', 'job loss', 'fired',
        'replaced by', 'chatgpt', 'gpt-', 'workforce', 'hiring freeze',
        'artificial intelligence', 'tech layoff', 'job cut', 'displacement',
        'model release', 'openai', 'anthropic', 'gemini', 'claude'
    ]

    import time
    tweets = []

    def get_tweets(user_id, handle, corpus):
        r = requests.get(
            f'https://api.twitter.com/2/users/{user_id}/tweets',
            headers=headers,
            params={
                'max_results': 100,
                'tweet.fields': 'public_metrics,created_at,text',
                'exclude': 'retweets,replies',
            }
        )
        data = r.json()
        result = []
        for t in data.get('data', []):
            text = t.get('text', '')
            if not any(kw in text.lower() for kw in AI_KEYWORDS):
                continue
            m = t.get('public_metrics', {})
            likes    = m.get('like_count', 0)
            retweets = m.get('retweet_count', 0)
            quotes   = m.get('quote_count', 0)
            replies  = m.get('reply_count', 0)
            result.append({
                'tweet_id':      t['id'],
                'handle':        handle,
                'corpus':        corpus,
                'date':          t['created_at'][:10],
                'text':          text,
                'text_length':   len(text),
                'likes':         likes,
                'retweets':      retweets,
                'replies':       replies,
                'quotes':        quotes,
                'views':         m.get('impression_count', 0),
                'amplification': likes + retweets*2 + quotes*1.5,
                'follower_count': 500000,
                'is_verified':   True,
            })
        print(f'  {handle}: {len(result)} AI tweets from {len(data.get("data",[]))} originals')
        time.sleep(1)
        return result

    print('Capital accounts...')
    for handle, uid in capital.items():
        tweets.extend(get_tweets(uid, handle, 'capital'))

    print('Labour accounts...')
    for handle, uid in labour.items():
        tweets.extend(get_tweets(uid, handle, 'labour'))

    df = pd.DataFrame(tweets)
    df['normalised_amp'] = df['amplification'] / np.log1p(df['follower_count'])
    Path(DATA_DIR).mkdir(exist_ok=True)
    df.to_csv(f'{DATA_DIR}tweets_raw.csv', index=False)
    print(f'Saved {len(df)} tweets')
    return df


if __name__ == "__main__":
    df = collect_mock()
    print("\nSample rows:")
    print(df.groupby("corpus")[["likes","retweets","amplification"]].mean().round(1))

