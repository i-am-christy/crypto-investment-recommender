import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv
from datetime import datetime

#load environement variables
load_dotenv()
API_KEY = os.getenv("COINGECKO_API_KEY")

def fetch_historical_data(coin_id, vs_currency="usd", days=30, max_retries=3):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": API_KEY
    }
    params = {
        "vs_currency": vs_currency,
        "days": days
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["coin_id"] = coin_id

            if "total_volumes" in data:
                volume_df = pd.DataFrame(data["total_volumes"], columns=["timestamp_vol", "volume"])
                df["volume"] = volume_df["volume"].values
            return df
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            elif response.status_code == 401:
                print(f"Unauthorized, check API Key or header for {coin_id}")
                break
            else:
                print(f"HTTP error for {coin_id}: {e}")
                break
        except Exception as e:
            print(f"General Error for {coin_id}: {e}")
            break
    return pd.DataFrame()
    
def fetch_market_data(coin_list, vs_currency="usd", max_retries=3):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": API_KEY
    }

    params = {
        "vs_currency": vs_currency,
        "ids": ",".join(coin_list),
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7h,30d"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data)
            df["data_collected_at"] = datetime.now()

            #select relevant columns
            columns_to_keep =[
                "id",  "current_price", "market_cap",
                "market_cap_rank", "total_volume", "price_change_percentage_1h_in_currency",
                "price_change_percentage_24h_in_currency", "price_change_percentage_7d_in_currency",
                "price_change_percentage_30d_in_currency","high_24h", "low_24h", "market_cap_change_24h", 
                "market_cap_change_percentage_24h", "circulating_supply", "total_supply", "max_supply",
                "ath", "ath_change_percentage", "ath_date", "atl", 
                "atl_change_percentage", "atl_date", "data_collected_at"
            ]

            available_cols = [c for c in columns_to_keep if c in df.columns]
            df = df[available_cols]

            #rename for consistency
            df = df.rename(columns={
                "price_change_percentage_1h_in_currency": "price_change_1h",
                "price_change_percentage_24h_in_currency": "price_change_24h",
                "price_change_percentage_7d_in_currency": "price_change_7d",
                "price_change_percentage_30d_in_currency": "price_change_30d"
            })
            return df
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            elif response.status_code == 401:
                print(f"Unauthorized, check API Key or header for {coin_id}")
                break
            else:
                print(f"HTTP error: {e}")
                break
        except Exception as e:
            print(f"General Error: {e}")
            break
    return pd.DataFrame()

def fetch_multiple_coins(coin_list, vs_currency="usd", days=30):
    all_data = []
    total = len(coin_list)

    for idx, coin in enumerate(coin_list, 1):
        df = fetch_historical_data(coin, vs_currency, days)
        if not df.empty:
            all_data.append(df)
        time.sleep(2)

        if idx < total:
            time.sleep(2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    return pd.DataFrame()

top_coins = [
        "bitcoin", "ethereum", "tether", "binancecoin", "solana",
        "ripple", "usd-coin", "cardano", "dogecoin", "tron", "usdc",
        "avalanche-2", "shiba-inu", "polkadot", "chainlink", "usds",
        "polygon", "litecoin", "uniswap", "near", "stellar", "monero",
        "sui"
    ]

market_data = fetch_market_data(top_coins)
if not market_data.empty:
    print("Fetched and Stored amrket data")
    market_data.to_csv("./data/raw/crypto_market_data.csv", index=False)

historical_data = fetch_multiple_coins(top_coins, days=7)
if not historical_data.empty:
    print("Fetched and Stored historic data")
    historical_data.to_csv("./data/raw/crypto_historical_data.csv", index=False)