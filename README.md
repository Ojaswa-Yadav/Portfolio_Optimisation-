# **Portfolio Optimization Using Black-Litterman and Machine Learning**

## **Overview**
This project leverages **financial data**, **machine learning**, **sentiment analysis**, and the **Black-Litterman model** to optimize a stock portfolio. The workflow predicts stock performance, analyzes sentiment from news data, and calculates optimal portfolio weights while balancing risk and return.

The project integrates with **Apache Airflow** to automate the entire pipeline, making it suitable for daily execution. Reports are generated and stored locally or uploaded to **Google Cloud Storage** for easy access.

Additionally, a **website interface** is available for users to view the portfolio in a user-friendly manner.

---

## **Key Features**

### **Stock Data Collection**
- Fetches historical stock data and returns for multiple companies using `yfinance`.

### **Machine Learning Predictions**
- Trains XGBoost models to predict one-day returns for each stock.

### **Sentiment Analysis**
- Analyzes sentiment from financial news using the **FinBERT** transformer model.

### **Portfolio Optimization**
- Combines predictions and sentiment data to calculate confidence scores.
- Optimizes portfolio weights using the **Black-Litterman model** and **Efficient Frontier**.

### **Automated Workflow**
- Automates the entire process using **Apache Airflow**.

### **Website Interface**
- A dedicated website allows users to view and interact with the optimized portfolio results.
- Users can visualize portfolio weights, returns, and other metrics in an intuitive layout.


---

## **Installation**

You can set up the Portfolio Optimization project by running these commands:

```bash
git clone https://github.com/Ojaswa-Yadav/Portfolio_Optimisation-.git
cd Portfolio_Optimisation-
pip install -r requirements.txt
airflow db init
airflow webserver
airflow scheduler
```

```bash
project/
├── dags/
│   └── airflow.py                           # Main Airflow DAG
├── modules/
│   ├── data_and_ml.py                       # Data fetching and machine learning
│   ├── sentiment_and_portfolio.py           # Sentiment analysis and portfolio optimization
│   ├── utils.py                             # Utility functions
├── config/
│   └── parameters.py                        # Centralized parameters
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
└── .gitignore                               # Ignored files for Git
```

