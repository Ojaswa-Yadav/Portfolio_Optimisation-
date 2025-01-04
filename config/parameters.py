# Stock and market data
TICKERS_PULL = ['XOM', 'HON', 'RTX', 'AMZN', 'PEP', 'UNH', 'JNJ', 'V', 'NVDA', 'AAPL', 'MSFT', 'GOOGL']
SPY_TICKER = "SPY"
PERIOD = "60mo"  # 5 years of historical data

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "max_depth": 50,
    "learning_rate": 0.8,
    "reg_lambda": 8,
    "subsample": 0.4,
    "grow_policy": "lossguide",
}

# Sentiment analysis model
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Market capitalizations of stocks
MCAPS = {
    'XOM': 398158000000, 'HON': 132107000000, 'RTX': 117750000000,
    'AMZN': 1508000000000, 'PEP': 230729000000, 'UNH': 502863000000,
    'JNJ': 373273000000, 'V': 527197000000, 'NVDA': 1152000000000,
    'AAPL': 3004000000000, 'MSFT': 2760000000000, 'GOOGL': 1676000000000,
}

# Number of days for live predictions and lag
NO_OF_DAYS_LAG_LIVE = 30

# Google Cloud Storage bucket
GCS_BUCKET_NAME = "eecs6893_project"
GCS_FILES = {
    "weights": "apache_data/weights_ML2_GBReg_df.csv",
    "weights_history": "apache_data/w_hist.csv",
    "return_summary": "apache_data/return_summary_csv.csv",
    "sharpe_summary": "apache_data/sharpe_summary_csv",
}
