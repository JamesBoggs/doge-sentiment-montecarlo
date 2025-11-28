#!/usr/bin/env python3
"""
Monte Carlo simulation for Dogecoin (DOGE) portfolio growth.
- Fetches DOGE prices (yfinance).
- Runs Monte Carlo sim with historical volatility.
- Saves results as CSV into results/.
"""
import os
import torch
import pandas as pd
from datetime import datetime
import yfinance as yf
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
    # Calculate log returns
    log_ret = np.log(prices / prices.shift(1))
    log_ret.dropna(inplace=True)
    return log_ret.tolist()

# ---------------------------
# Monte Carlo simulation
# ---------------------------
def monte_carlo_simulation(returns, start_value=1000.0, n_paths=5000, horizon=90):
    rets = torch.tensor(returns, dtype=torch.float32)
    mu = rets.mean()
    sigma = rets.std()
    
    # Generate random paths
    dt = 1.0 / 252  # trading days per year
    Z = torch.randn((n_paths, horizon))
    paths = torch.zeros((n_paths, horizon + 1))
    paths[:, 0] = start_value
    
    for t in range(horizon):
        paths[:, t + 1] = paths[:, t] * torch.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t])
    
    final = paths[:, -1]
    
    return {
        "mean_final": float(final.mean()),
        "median_final": float(final.median()),
        "p10_final": float(final.quantile(0.1)),
        "p90_final": float(final.quantile(0.9)),
        "std_final": float(final.std()),
        "min_final": float(final.min()),
        "max_final": float(final.max()),
    }

# ---------------------------
def main():
    os.makedirs("results", exist_ok=True)
    print("📊 Fetching DOGE prices...")
    rets = fetch_doge_returns()
    print(f"✅ Got {len(rets)} days of returns")
    
    print("🎲 Running Monte Carlo simulation...")
    stats = monte_carlo_simulation(rets, start_value=1000.0, n_paths=5000, horizon=90)
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = f"results/doge_mc_{ts}.csv"
    pd.DataFrame([stats]).to_csv(outpath, index=False)
    print(f"✅ Saved results to {outpath}")
    print(f"\n📈 90-day projection (starting with $1000):")
    print(f"   Mean: ${stats['mean_final']:.2f}")
    print(f"   Median: ${stats['median_final']:.2f}")
    print(f"   10th percentile: ${stats['p10_final']:.2f}")
    print(f"   90th percentile: ${stats['p90_final']:.2f}")

if __name__ == "__main__":
    main()
