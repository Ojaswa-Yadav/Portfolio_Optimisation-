Portfolio Optimization Using Black-Litterman and Machine Learning
Overview
This project leverages financial data, machine learning, sentiment analysis, and the Black-Litterman model to optimize a stock portfolio. The workflow predicts stock performance, analyzes sentiment from news data, and calculates optimal portfolio weights while balancing risk and return.

The project integrates with Apache Airflow to automate the entire pipeline, making it suitable for daily execution. Reports are generated and stored locally or uploaded to Google Cloud Storage.

Key Features
Stock Data Collection:

Fetches historical stock data and returns for multiple companies using yfinance.
Machine Learning Predictions:

Trains XGBoost models to predict one-day returns for each stock.
Sentiment Analysis:

Analyzes sentiment from financial news using the FinBERT transformer model.
Portfolio Optimization:

Combines predictions and sentiment data to calculate confidence scores.
Optimizes portfolio weights using the Black-Litterman model and Efficient Frontier.
Automated Workflow:

Automates the entire process using Apache Airflow.
Reporting:

Generates reports for portfolio weights, Sharpe ratios, and returns.
Uploads reports to Google Cloud Storage for easy access.
Project Workflow
1. Pipeline Steps
The Airflow DAG orchestrates the following tasks:

Step 1: Fetch historical stock data and calculate returns.
Step 2: Train and predict future stock performance using machine learning.
Step 3: Perform sentiment analysis on financial news headlines.
Step 4: Optimize portfolio weights using the Black-Litterman model.
Step 5: Save reports locally and upload them to Google Cloud Storage.
2. Output
The pipeline generates the following reports:

Portfolio Weights: Optimal stock allocation (weights_ML2_GBReg_df.csv).
Sharpe Ratio Summary: Performance evaluation (sharpe_summary_csv.csv).
Return Summary: Portfolio returns statistics (return_summary_csv.csv).
Weights History: Historical allocation trends (w_hist.csv).
