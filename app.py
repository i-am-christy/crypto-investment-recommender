from fastapi import FastAPI, Query
import pandas as pd
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(
    title="Crypto Assest Recommendation API",
    descriptive="A simple rule-based statistical model crypto recommender system using market data.",
    version="1.0.0"
)

#load processed data
DATA_PATH = "data/processed/crypto_features.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Processed features not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

class InputData(BaseModel):
    top_n: int = 5
    risk_preference: str = "Moderate"

@app.get("/")
def home():
    return {"message": "Welcome to the Crypto Recommender API!"}

@app.get("/recommend")
def recommend(
    risk_preference: str = Query(
        "Conservative",
        description="Choose your risk level: Conservative, Moderate, or Aggressive",
        regex="(Conservative|Moderate|Aggressive)"
    ),
    top_n: int = Query(
        5,
        ge=1, le=20,
        description="Number of top coins to recommend (1-20)"
    )
):
    """
    Recommend top N coins based on the user's risk preference and model scores.
    """

    # Normalize risk input for comparison
    risk_preference = risk_preference.capitalize().strip()

    # Filter dataset by selected risk level
    filtered_df = df[df["risk_level"].str.lower() == risk_preference.lower()]

    if filtered_df.empty:
        return {
            "error": f"No coins found for risk level '{risk_preference}'. Try another level."
        }

    # Sort by investment score (descending)
    ranked_df = filtered_df.sort_values(by="investment_score", ascending=False).head(top_n)

    # Prepare output
    recommendations = ranked_df[[
        "id", "current_price", "market_cap_rank",
        "investment_score", "risk_level",
        "recommendation", "reasoning"
    ]].to_dict(orient="records")

    return {
        "user_preference": risk_preference,
        "total_available": len(filtered_df),
        "recommendations_returned": len(recommendations),
        "top_recommendations": recommendations
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)