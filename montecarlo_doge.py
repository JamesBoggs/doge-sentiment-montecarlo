#!/usr/bin/env python3
"""
Monte Carlo simulation for Dogecoin (DOGE) portfolio growth with sentiment adjustment.
- Fetches DOGE prices (yfinance).
- Polls Twitter/X API for sentiment about "Dogecoin" with fallback options.
- Runs Monte Carlo sim adjusting drift by sentiment.
- Saves results as CSV into results/.
"""
import os
import torch
import pandas as pd
from datetime import datetime
import yfinance as yf
import requests
from textblob import TextBlob
import time
import numpy as np

# ---------------------------
# Fetch DOGE returns
# ---------------------------
def fetch_doge_returns():
    data = yf.download("DOGE-USD", period="1y", interval="1d", progress=False)
    # Pick adjusted close if it exists, otherwise use close
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data["Close"]
    # use numpy log instead of torch.log here
    data["LogRet"] = np.log(prices / prices.shift(1))
    data.dropna(inplace=True)
    return data["LogRet"].tolist()

# ---------------------------
# Fetch Twitter sentiment (with fixes)
# ---------------------------
def fetch_twitter_sentiment(query="dogecoin", max_results=20):
    """
    Calls Twitter API v2 recent search.
    Requires BEARER_TOKEN in environment.
    Returns avg polarity (-1 bearish, +1 bullish).
    """
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        print("⚠️ No Twitter token found. Using neutral sentiment = 0.")
        return 0.0
    
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": f"{query} -is:retweet lang:en",
        "max_results": min(max_results, 100),  # API max is 100
        "tweet.fields": "created_at"
    }
    headers = {
        "Authorization": f"Bearer {bearer}",
        "User-Agent": "DogecoinMCBot/1.0"
    }
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        
        # Handle rate limiting
        if r.status_code == 429:
            reset_time = r.headers.get('x-rate-limit-reset', 0)
            wait_time = int(reset_time) - int(time.time()) if reset_time else 900
            print(f"⚠️ Rate limited. Need to wait {wait_time}s. Using neutral sentiment.")
            return 0.0
        
        # Handle other errors
        if r.status_code != 200:
            print(f"⚠️ Twitter API error {r.status_code}: {r.text[:200]}")
            return 0.0
        
        data = r.json()
        tweets = [t["text"] for t in data.get("data", [])]
        
        if not tweets:
            print("⚠️ No tweets found. Using neutral sentiment.")
            return 0.0
        
        # Calculate sentiment
        scores = []
        for tweet in tweets:
            try:
                polarity = TextBlob(tweet).sentiment.polarity
                scores.append(polarity)
            except:
                continue
        
        if not scores:
            return 0.0
        
        avg_sentiment = sum(scores) / len(scores)
        print(f"📊 Analyzed {len(scores)} tweets. Avg sentiment: {avg_sentiment:.3f}")
        return avg_sentiment
    
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error: {e}. Using neutral sentiment.")
        return 0.0

# ---------------------------
# Alternative: Reddit sentiment (free, no auth needed)
# ---------------------------
def fetch_reddit_sentiment(query="dogecoin", limit=25):
    """
    Fetches sentiment from Reddit via pushshift.io API (no auth required).
    Falls back if Twitter fails.
    """
    try:
        url = f"https://www.reddit.com/r/dogecoin/search.json"
        params = {
            "q": query,
            "limit": limit,
            "sort": "new",
            "t": "day"
        }
        headers = {"User-Agent": "DogecoinMCBot/1.0"}
        
        r = requests.get(url, params=params, headers=headers, timeout=10)
        
        if r.status_code != 200:
            return 0.0
        
        posts = r.json().get("data", {}).get("children", [])
        texts = [p["data"]["title"] + " " + p["data"].get("selftext", "") 
                 for p in posts]
        
        if not texts:
            return 0.0
        
        scores = [TextBlob(t).sentiment.polarity for t in texts]
        avg = sum(scores) / len(scores)
        print(f"📊 Reddit: Analyzed {len(scores)} posts. Avg sentiment: {avg:.3f}")
        return avg
    
    except Exception as e:
        print(f"⚠️ Reddit fetch error: {e}")
        return 0.0

# ---------------------------
# Monte Carlo simulation
# ---------------------------
def monte_carlo_with_sentiment(returns, sentiment,
                              start_value=1000.0, n_paths=5000, horizon=90):
    rets = torch.tensor(returns)
    mu = rets.mean()
    sigma = rets.std()
    
    # adjust drift by sentiment
    alpha = 0.5
    mu_adj = mu + alpha * torch.tensor(sentiment)
    
    sims = torch.exp(mu_adj + sigma * torch.randn((n_paths, horizon)))
    sims = sims.cumprod(dim=1) * start_value
    final = sims[:, -1]
    
    return {
        "mean_final": float(final.mean()),
        "median_final": float(final.median()),
        "p10_final": float(final.kthvalue(int(0.1 * n_paths))[0]),
        "p90_final": float(final.kthvalue(int(0.9 * n_paths))[0]),
        "sentiment_used": float(sentiment),
    }

# ---------------------------
def main():
    os.makedirs("results", exist_ok=True)
    
    print("📈 Fetching DOGE returns...")
    rets = fetch_doge_returns()
    print(f"✅ Got {len(rets)} daily returns")
    
    print("\n💬 Fetching sentiment...")
    # Try Twitter first, fall back to Reddit
    sentiment = fetch_twitter_sentiment("dogecoin", max_results=30)
    
    if sentiment == 0.0:
        print("🔄 Twitter failed/neutral. Trying Reddit...")
        sentiment = fetch_reddit_sentiment("dogecoin", limit=25)
    
    print(f"\n🎯 Final sentiment score: {sentiment:.3f}")
    
    print("\n🎲 Running Monte Carlo simulation...")
    stats = monte_carlo_with_sentiment(rets, sentiment)
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    outpath = f"results/doge_mc_sent_{ts}.csv"
    pd.DataFrame([stats]).to_csv(outpath, index=False)
    
    print(f"\n✅ Saved results to {outpath}")
    print("\n📊 Summary:")
    print(f"  Mean final value: ${stats['mean_final']:.2f}")
    print(f"  Median final value: ${stats['median_final']:.2f}")
    print(f"  10th percentile: ${stats['p10_final']:.2f}")
    print(f"  90th percentile: ${stats['p90_final']:.2f}")
    print(f"  Sentiment: {stats['sentiment_used']:.3f}")

if __name__ == "__main__":
    main()
