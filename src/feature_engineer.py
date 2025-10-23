import os
import logging
from typing import Optional

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


class CryptoFeatureEngineer:
    """
    Transform raw market and historical data into features and
    generate a simple, explainable investment recommendation.
    """

    def __init__(self, market_data_path: str, historical_data_path: Optional[str] = None):
        """
        Load CSV inputs and prepare dataframes.

        Args:
            market_data_path: path to market snapshot CSV (coins/markets output)
            historical_data_path: optional path to historical price CSV (market_chart outputs)
        """
        if not os.path.exists(market_data_path):
            raise FileNotFoundError(f"Market data file not found: {market_data_path}")

        self.market_df = pd.read_csv(market_data_path)
        self.historical_df = None

        if historical_data_path:
            if not os.path.exists(historical_data_path):
                logging.warning("Historical data file not found: %s (continuing without it)", historical_data_path)
            else:
                self.historical_df = pd.read_csv(historical_data_path)

        # Ensure consistent column names (lowercase)
        self.market_df.columns = [c.strip() for c in self.market_df.columns]

        logging.info("Loaded market data: %d rows", len(self.market_df))
        if self.historical_df is not None:
            self.historical_df.columns = [c.strip() for c in self.historical_df.columns]
            logging.info("Loaded historical data: %d rows", len(self.historical_df))
        else:
            logging.info("No historical data provided; some features will use market snapshot only")

    #MOMENTUM FEATURES
    def calculate_momentum_features(self):
        """
        Create momentum features from available market snapshot columns.
        If percent-change columns exist (e.g., price_change_percentage_24h_in_currency),
        use them; otherwise momentum remains zero and should be computed from history.
        """
        logging.info("Calculating momentum features")

        # Look for common percent-change column name patterns
        col_candidates = [c for c in self.market_df.columns if 'price_change' in c.lower()]

        # Normalize candidate values to numeric and fill NaN with 0
        for c in col_candidates:
            self.market_df[c] = pd.to_numeric(self.market_df[c], errors='coerce').fillna(0.0)

        # Try to find specific timeframe columns in the candidates list
        pct_1h = next((c for c in col_candidates if '1h' in c), None)
        pct_24h = next((c for c in col_candidates if '24h' in c), None)
        pct_7d = next((c for c in col_candidates if '7d' in c), None)
        #pct_30d = next((c for c in col_candidates if '30d' in c), None)

        # Create explicit momentum columns; default to 0 if not found
        self.market_df['momentum_1h'] = self.market_df[pct_1h] if pct_1h in self.market_df.columns else 0.0
        self.market_df['momentum_24h'] = self.market_df[pct_24h] if pct_24h in self.market_df.columns else 0.0
        self.market_df['momentum_7d'] = self.market_df[pct_7d] if pct_7d in self.market_df.columns else 0.0
        #self.market_df['momentum_30d'] = self.market_df[pct_30d] if pct_30d in self.market_df.columns else 0.0

        # Define a weighted momentum score. Weights are chosen for interpretability.
        # If you compute momentum from historical time series later, replace these columns.
        self.market_df['momentum_score'] = (
            0.30 * pd.to_numeric(self.market_df['momentum_1h'], errors='coerce').fillna(0.0)
            + 0.30 * pd.to_numeric(self.market_df['momentum_24h'], errors='coerce').fillna(0.0)
            + 0.40 * pd.to_numeric(self.market_df['momentum_7d'], errors='coerce').fillna(0.0)
        )

        # Volume to market cap ratio: liquidity proxy. Use safe division.
        if 'total_volume' in self.market_df.columns and 'market_cap' in self.market_df.columns:
            vol = pd.to_numeric(self.market_df['total_volume'], errors='coerce').fillna(0.0)
            mcap = pd.to_numeric(self.market_df['market_cap'], errors='coerce').replace(0.0, np.nan)
            ratio = (vol / mcap).fillna(0.0) * 100.0
            self.market_df['volume_to_mcap_ratio'] = ratio
        else:
            self.market_df['volume_to_mcap_ratio'] = 0.0

        # Market cap momentum if available
        if 'market_cap_change_percentage_24h' in self.market_df.columns:
            self.market_df['mcap_momentum'] = pd.to_numeric(self.market_df['market_cap_change_percentage_24h'], errors='coerce').fillna(0.0)
        else:
            self.market_df['mcap_momentum'] = 0.0

        return self

    #VALUE FEATURES 
    def calculate_value_features(self):
        """
        Create value-based features:
         - distance_from_ath: how far current price is from all-time-high (percentage)
         - recovery_potential: heuristic combining distance_from_ath and momentum_score
         - growth_from_atl: percent change since all-time-low
         - price_range_24h: daily price range as percent of current price
         - supply_ratio and scarcity_score: tokenomics-based scarcity measure
        """
        logging.info("Calculating value features")

        # Distance from ATH (absolute percent)
        if 'ath_change_percentage' in self.market_df.columns:
            self.market_df['distance_from_ath'] = pd.to_numeric(self.market_df['ath_change_percentage'], errors='coerce').abs().fillna(100.0)
            self.market_df['distance_from_ath'] = self.market_df['distance_from_ath'].astype(float)
        else:
            self.market_df['distance_from_ath'] = 100.0

        # Recovery potential: use small helper that returns 0-100 score
        self.market_df['recovery_potential'] = self.market_df.apply(
            lambda r: self._calculate_recovery_score(
                float(r.get('distance_from_ath', 100.0)),
                float(r.get('momentum_score', 0.0))
            ), axis=1
        )

        # Growth from ATL if available
        if 'atl_change_percentage' in self.market_df.columns:
            self.market_df['growth_from_atl'] = pd.to_numeric(self.market_df['atl_change_percentage'], errors='coerce').fillna(0.0)
        else:
            self.market_df['growth_from_atl'] = 0.0

        # Price range in last 24h (volatility proxy)
        if 'high_24h' in self.market_df.columns and 'low_24h' in self.market_df.columns and 'current_price' in self.market_df.columns:
            high = pd.to_numeric(self.market_df['high_24h'], errors='coerce').fillna(0.0)
            low = pd.to_numeric(self.market_df['low_24h'], errors='coerce').fillna(0.0)
            cur = pd.to_numeric(self.market_df['current_price'], errors='coerce').replace(0.0, np.nan)
            self.market_df['price_range_24h'] = (((high - low) / cur) * 100.0).fillna(0.0)
        else:
            self.market_df['price_range_24h'] = 0.0

        # Supply ratio and scarcity score
        if 'circulating_supply' in self.market_df.columns and 'total_supply' in self.market_df.columns:
            circ = pd.to_numeric(self.market_df['circulating_supply'], errors='coerce').fillna(0.0)
            total = pd.to_numeric(self.market_df['total_supply'], errors='coerce').replace(0.0, np.nan)
            supply_ratio = (circ / total).fillna(1.0)
            self.market_df['supply_ratio'] = supply_ratio
            self.market_df['scarcity_score'] = ((1.0 - supply_ratio) * 100.0).clip(lower=0.0)
        else:
            self.market_df['supply_ratio'] = 1.0
            self.market_df['scarcity_score'] = 0.0

        return self

    #STABILITY FEATURES
    def calculate_stability_features(self):
        """
        Create stability and risk features:
         - volatility_score: derived from historical returns if available, else falls back to price_range_24h
         - mcap_stability_score: log-scaled market cap normalized to 0-100
         - volume_stability_score: log-scaled volume normalized to 0-100
         - stability_score: combined stability indicator
        """
        logging.info("Calculating stability features")

        # Volatility from historical data if available
        if self.historical_df is not None and 'coin_id' in self.historical_df.columns and 'price' in self.historical_df.columns:
            volatility_metrics = self._calculate_historical_volatility()
            # Merge volatility back into market_df by coin id (market_df uses 'id' for coin)
            self.market_df = self.market_df.merge(volatility_metrics, left_on='id', right_on='coin_id', how='left')
            # Fill missing volatility with a reasonable default
            if 'volatility_score' in self.market_df.columns:
                self.market_df['volatility_score'] = pd.to_numeric(self.market_df['volatility_score'], errors='coerce').fillna(self.market_df['volatility_score'].median())
            else:
                self.market_df['volatility_score'] = 0.0
        else:
            # Use 24h price range as a proxy for volatility if historical is not available
            self.market_df['volatility_score'] = pd.to_numeric(self.market_df.get('price_range_24h', 0.0), errors='coerce').fillna(0.0)

        # Market cap stability (log scale) normalized to 0-100
        if 'market_cap' in self.market_df.columns:
            mcap = pd.to_numeric(self.market_df['market_cap'], errors='coerce').fillna(0.0)
            max_mcap = mcap.replace(0, np.nan).max()
            if pd.notna(max_mcap) and max_mcap > 0:
                self.market_df['mcap_stability_score'] = (np.log10(mcap + 1) / np.log10(max_mcap + 1)) * 100.0
            else:
                self.market_df['mcap_stability_score'] = 0.0
        else:
            self.market_df['mcap_stability_score'] = 0.0

        # Volume stability normalized to 0-100
        if 'total_volume' in self.market_df.columns:
            vol = pd.to_numeric(self.market_df['total_volume'], errors='coerce').fillna(0.0)
            max_vol = vol.max()
            if max_vol > 0:
                self.market_df['volume_stability_score'] = (np.log10(vol + 1) / np.log10(max_vol + 1)) * 100.0
            else:
                self.market_df['volume_stability_score'] = 0.0
        else:
            self.market_df['volume_stability_score'] = 0.0

        # Combine inverted volatility and market cap stability into a single stability score
        vol = pd.to_numeric(self.market_df['volatility_score'], errors='coerce').fillna(0.0)
        max_vol = vol.max() if not vol.empty else 0.0
        if max_vol > 0:
            inverted_vol = ((max_vol - vol) / max_vol) * 100.0
        else:
            inverted_vol = np.zeros(len(vol))

        self.market_df['stability_score'] = (0.5 * inverted_vol) + (0.5 * self.market_df['mcap_stability_score'].fillna(0.0))

        return self

    # MARKET POSITION FEATURES
    def calculate_market_position_features(self):
        """
        Create market position features:
         - rank_score: inverted market cap rank (rank 1 highest score)
         - market_dominance: percentage share of total market cap
         - liquidity_score: normalized volume_to_mcap_ratio
         - market_tier: label for cap tier (Blue Chip, Large Cap, etc.)
        """
        logging.info("Calculating market position features")

        # Rank score
        if 'market_cap_rank' in self.market_df.columns:
            rank = pd.to_numeric(self.market_df['market_cap_rank'], errors='coerce').fillna(self.market_df['market_cap_rank'].max())
            max_rank = rank.max() if not np.isnan(rank.max()) else 1.0
            self.market_df['rank_score'] = ((max_rank - rank + 1) / max_rank) * 100.0
        else:
            self.market_df['rank_score'] = 0.0

        # Market dominance (share of total market cap)
        if 'market_cap' in self.market_df.columns:
            mcap = pd.to_numeric(self.market_df['market_cap'], errors='coerce').fillna(0.0)
            total_mcap = mcap.sum()
            if total_mcap > 0:
                self.market_df['market_dominance'] = (mcap / total_mcap) * 100.0
            else:
                self.market_df['market_dominance'] = 0.0
        else:
            self.market_df['market_dominance'] = 0.0

        # Liquidity score (normalize volume_to_mcap_ratio using 95th percentile)
        if 'volume_to_mcap_ratio' in self.market_df.columns:
            ratio = pd.to_numeric(self.market_df['volume_to_mcap_ratio'], errors='coerce').fillna(0.0)
            cap = np.nanpercentile(ratio, 95) if len(ratio) > 0 else 1.0
            cap = cap if cap > 0 else 1.0
            self.market_df['liquidity_score'] = (ratio.clip(upper=cap) / cap) * 100.0
        else:
            self.market_df['liquidity_score'] = 0.0

        # Market tier classification
        self.market_df['market_tier'] = self.market_df.apply(
            lambda r: self._classify_market_tier(int(r.get('market_cap_rank', 999)), float(r.get('market_cap', 0.0))),
            axis=1
        )

        return self

    # COMPOSITE SCORE & RISK
    def calculate_composite_investment_score(self):
        """
        Normalize component scores and compute final investment score.
        We normalize key components to 0-100 before weighting.
        """
        logging.info("Calculating composite investment score")

        # Columns to normalize (if present)
        components = {
            'momentum_score': None,
            'stability_score': None,
            'rank_score': None,
            'recovery_potential': None
        }

        # Normalize each component to 0-100 and store as <name>_norm
        for col in list(components.keys()):
            if col in self.market_df.columns:
                vals = pd.to_numeric(self.market_df[col], errors='coerce').fillna(0.0)
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    self.market_df[f"{col}_norm"] = ((vals - vmin) / (vmax - vmin)) * 100.0
                else:
                    # No variance: assign neutral 50
                    self.market_df[f"{col}_norm"] = 50.0
            else:
                self.market_df[f"{col}_norm"] = 50.0

        # Weighted formula (explainable weights)
        self.market_df['investment_score'] = (
            0.35 * self.market_df['momentum_score_norm']
            + 0.25 * self.market_df['recovery_potential_norm']
            + 0.20 * self.market_df['stability_score_norm']
            + 0.20 * self.market_df['rank_score_norm']
        ).fillna(50.0)

        # Clip to 0-100
        self.market_df['investment_score'] = self.market_df['investment_score'].clip(0.0, 100.0)

        return self

    def classify_risk_levels(self):
        """
        Assign a simple risk label based on volatility and market cap rank.
        """
        logging.info("Classifying risk levels")

        def risk_fn(row):
            vol = float(row.get('volatility_score', 5.0))
            rank = int(row.get('market_cap_rank', 999))
            return self._determine_risk_level(vol, rank, float(row.get('market_cap', 0.0)))

        self.market_df['risk_level'] = self.market_df.apply(risk_fn, axis=1)
        return self

    #RECOMMENDATIONS
    def generate_recommendations(self):
        """
        Convert investment_score into human-friendly recommendations and
        a concise reasoning string for each asset.
        """
        logging.info("Generating recommendations")

        def recommendation(score):
            if score >= 75:
                return "Strong Buy"
            if score >= 60:
                return "Buy"
            if score >= 45:
                return "Moderate Buy"
            if score >= 30:
                return "Hold"
            return "Avoid/Sell"

        self.market_df['recommendation'] = self.market_df['investment_score'].apply(recommendation)
        self.market_df['reasoning'] = self.market_df.apply(self._generate_reasoning, axis=1)

        return self

    # SAVE
    def save_features(self, output_path: str) -> pd.DataFrame:
        """
        Persist a curated set of columns to CSV and return the saved dataframe.
        """
        output_columns = [
            'id', 'symbol', 'name', 'current_price', 'market_cap', 'market_cap_rank',
            'momentum_score', 'recovery_potential', 'stability_score', 'rank_score',
            'investment_score', 'risk_level', 'recommendation', 'reasoning',
            'volatility_score', 'volume_to_mcap_ratio', 'market_tier',
            'distance_from_ath', 'liquidity_score', 'market_dominance'
        ]
        available = [c for c in output_columns if c in self.market_df.columns]

        out_df = self.market_df[available].copy()
        if 'investment_score' in out_df.columns:
            out_df = out_df.sort_values('investment_score', ascending=False)
        out_df.to_csv(output_path, index=False)

        logging.info("Saved features to %s (rows=%d)", output_path, len(out_df))
        return out_df

    # HELPER METHODS
    def _calculate_recovery_score(self, distance_from_ath: float, momentum: float) -> float:
        """
        Heuristic recovery score in range roughly 0-100.
        Designed to highlight assets that are moderately below ATH and have positive momentum.
        """
        # Ensure numeric
        d = float(distance_from_ath)
        m = float(momentum)

        if d < 5:
            base = 40.0
        elif d <= 15:
            base = 70.0
        elif d <= 35:
            base = 80.0
        elif d <= 60:
            base = 60.0
        else:
            base = 30.0

        # Add small influence from momentum
        score = base + (m * 0.3)
        return float(np.clip(score, 0.0, 100.0))

    def _calculate_historical_volatility(self) -> pd.DataFrame:
        """
        Compute volatility (std of returns) per coin from historical price series.
        Returns a DataFrame with columns: coin_id, volatility_score, annualized_volatility
        """
        logging.info("Calculating historical volatility per coin")
        results = []
        if self.historical_df is None:
            return pd.DataFrame(results)

        df = self.historical_df.copy()
        # Ensure timestamps are datetime and sorted per coin
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        for coin in df['coin_id'].unique():
            coin_df = df[df['coin_id'] == coin].sort_values('timestamp')
            coin_df['return'] = coin_df['price'].pct_change()
            vol = coin_df['return'].std()
            vol_pct = float(vol * 100.0) if pd.notna(vol) else 0.0
            annualized = vol_pct * np.sqrt(365.0)
            results.append({
                'coin_id': coin,
                'volatility_score': vol_pct,
                'annualized_volatility': annualized
            })

        return pd.DataFrame(results)

    def _classify_market_tier(self, rank: int, market_cap: float) -> str:
        """Label assets by market cap rank into simple tiers."""
        if rank <= 5:
            return "Blue Chip"
        if rank <= 20:
            return "Large Cap"
        if rank <= 50:
            return "Mid Cap"
        return "Small Cap"

    def _determine_risk_level(self, volatility: float, rank: int, market_cap: float) -> str:
        """
        Determine a simple risk label using volatility and rank.
        This is an explanatory heuristic, not investment advice.
        """
        vol = float(volatility)

        # Top 3 market leaders – very stable
        if rank <= 3 and vol < 5.0:
            return "Conservative"
        
        # Large caps (ranks 4–10) – moderately volatile
        elif 4 <= rank <= 10 and vol < 8.0:
            return "Moderate"
        
        # Mid to small caps (ranks 11–17) – higher volatility
        elif 11 <= rank <= 17 and vol < 12.0:
            return "Moderate"
        
        # Smaller coins or high-volatility assets
        else:
            return "Aggressive"

    def _generate_reasoning(self, row) -> str:
        """
        Build a short reasoning summary describing the main drivers
        for the recommendation.
        """
        reasons = []

        m = float(row.get('momentum_score', 0.0))
        if m > 5:
            reasons.append(f"strong momentum (+{m:.1f}%)")
        elif m > 0:
            reasons.append(f"positive momentum (+{m:.1f}%)")
        elif m < -5:
            reasons.append(f"negative momentum ({m:.1f}%)")

        rank = int(row.get('market_cap_rank', 999))
        if rank <= 5:
            reasons.append("market leader")
        elif rank <= 20:
            reasons.append("established player")

        risk = row.get('risk_level', "Moderate")
        if risk == "Conservative":
            reasons.append("low risk")
        elif risk == "Aggressive":
            reasons.append("high risk")

        vol_ratio = float(row.get('volume_to_mcap_ratio', 0.0))
        if vol_ratio > 10.0:
            reasons.append("high liquidity")

        if not reasons:
            return "Mixed signals."
        return (", ".join(reasons).capitalize() + ".")

#MAIN
if __name__ == "__main__":
    MARKET_PATH = r"C:\Users\HP\Desktop\notebooks\crypto-investment-recommender\data\raw\crypto_market_data.csv"
    HIST_PATH = r"C:\Users\HP\Desktop\notebooks\crypto-investment-recommender\data\raw\crypto_historical_data.csv"
    OUTPUT_PATH = r"C:\Users\HP\Desktop\notebooks\crypto-investment-recommender\data\processed\crypto_features.csv"

    engineer = CryptoFeatureEngineer(MARKET_PATH, HIST_PATH)
    (engineer
     .calculate_momentum_features()
     .calculate_value_features()
     .calculate_stability_features()
     .calculate_market_position_features()
     .calculate_composite_investment_score()
     .classify_risk_levels()
     .generate_recommendations()
    )
    out_df = engineer.save_features(OUTPUT_PATH)
    logging.info("Feature engineering pipeline complete. Top results:\n%s", out_df.head().to_string(index=False))
