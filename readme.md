# Crypto Investment Recommender (Rule-Based System)

A **rule-based statistical model crypto investment recommender** built with **Python** and **FastAPI**, powered by **CoinGecko API**.  
It analyzes the top cryptocurrencies based on recent **market data, volatility, momentum, liquidity, and value metrics** and provides **data-driven buy recommendations** matched to the user’s **risk preference** (Conservative, Moderate, Aggressive).


## Project Overview

### Goal
To design an **explainable investment recommender** that identifies promising cryptocurrencies using a **statistical and rule-based approach** instead of complex ML models.

The system:
- Collects real-time crypto data from CoinGecko API  
- Calculates derived financial indicators (momentum, volatility, market position, etc.)  
- Scores and ranks assets based on multi-factor heuristics  
- Generates recommendations like _“Strong Buy”_, _“Buy”_, or _“Avoid”_ depending on the coin’s performance and risk profile  

---

## How It Works

### Architecture Overview
crypto-investment-recommender/
│
├── src/
│ ├── data_collector.py # Pulls real-time data from CoinGecko API
│ └── feature_engineer.py # Performs feature engineering and generates investment scores
│
├── app.py # FastAPI for deployment
│
├── data/
│ ├── raw/ # Original market & historical data
| |  |──crypto_historical_data.csv
| |  |──crypto_market_data.csv
│ └── processed/
│ |  |── crypto_features.csv # Engineered dataset with recommendations
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files and folders to ignore in Git

## Features & Logic

| Category | What It Measures | Examples of Derived Features |
|-----------|------------------|-------------------------------|
| **Momentum** | Price movement trends | `momentum_score`, `mcap_momentum` |
| **Value** | Undervaluation & growth potential | `distance_from_ath`, `recovery_potential`, `scarcity_score` |
| **Stability** | Price consistency & volatility | `stability_score`, `volatility_score` |
| **Market Position** | Relative rank & dominance | `market_tier`, `rank_score`, `market_dominance` |
| **Composite Investment Score** | Weighted combination of all metrics | `investment_score` |
| **Risk Classification** | Labels coins as `Conservative`, `Moderate`, or `Aggressive` | `_determine_risk_level()` heuristic |

Each coin is scored and ranked, then the model outputs:
- `risk_level` → user or model-defined (Conservative / Moderate / Aggressive)
- `recommendation` → _Strong Buy_, _Buy_, _Avoid_
- `reasoning` → brief, human-readable explanation

## Example Output (`crypto_features.csv`)

| id | current_price | momentum_score | stability_score | investment_score | risk_level | recommendation | reasoning |
|----|----------------|----------------|------------------|------------------|-------------|----------------|------------|
| bitcoin | 107,112 | -2.04 | 68.8 | 74.7 | Conservative | **Buy** | Market leader, low risk. |
| tether | 1.00 | -0.02 | 95.3 | 95.8 | Conservative | **Strong Buy** | Market leader, low risk, high liquidity. |
| solana | 178.79 | -4.83 | 56.8 | 29.7 | Conservative | **Avoid** | Established player, moderate risk. |

