#!/usr/bin/env python3
"""
Monte Carlo simulation for Dogecoin (DOGE) portfolio growth with sentiment adjustment.
- Fetches DOGE prices (yfinance).
- Polls Twitter/X API for sentiment about "Dogecoin".
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
# Fetch Twitter sentiment
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
        "query": query + " -is:retweet lang:en",
        "max_results": str(max_results),
    }
    headers = {"Authorization": f"Bearer {bearer}"}
    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        print(f"⚠️ Twitter API error {r.status_code}: {r.text}")
        return 0.0
    tweets = [t["text"] for t in r.json().get("data", [])]
    if not tweets:
        return 0.0
    scores = [TextBlob(t).sentiment.polarity for t in tweets]
    return sum(scores) / len(scores)

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
    rets = fetch_doge_returns()
    sentiment = fetch_twitter_sentiment("dogecoin", max_results=30)
    stats = monte_carlo_with_sentiment(rets, sentiment)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    outpath = f"results/doge_mc_sent_{ts}.csv"
    pd.DataFrame([stats]).to_csv(outpath, index=False)
    print(f"✅ Saved results to {outpath}")

if __name__ == "__main__":
    main()
