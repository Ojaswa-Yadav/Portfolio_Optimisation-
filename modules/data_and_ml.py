import numpy as np
import pandas as pd

from numpy.linalg import inv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def step1_pull_stock_data():
    tickers = tickers_pull
    ohlc = yf.download(tickers, period="60mo") # 60 months
    prices = ohlc["Adj Close"]
    # prices
    # market_prices_BL_raw = yf.download("SPY", period="60mo")["Adj Close"]
    returns_full = prices.pct_change()
    returns= returns_full.dropna()
    # returns
    # returns_full =  returns_full.reset_index()
    # df = returns        
    market_prices_BL_pull = yf.download("SPY", period="60mo")["Adj Close"]
    market_prices_BL_raw = market_prices_BL_pull
    returns.to_csv('/home/edsharecourse/projdatabucket/df.csv', index ='Date')
    prices.to_csv('/home/edsharecourse/projdatabucket/prices.csv', index ='Date')
    market_prices_BL_raw.to_csv('/home/edsharecourse/projdatabucket/market_prices_BL_raw.csv', index='Date') 
    
def step2_ml_xg():
    # ----- XGBoost Parameters ------------
    xg_max_depth=50
    xg_learning_rate=0.8
    xg_reg_lambda=8
    xg_subsample=0.4
    xg_grow_policy="lossguide"    
    # ----- ONE DAY PREDICTION:: LIVE  ------------    
    df = pd.read_csv('/home/edsharecourse/projdatabucket/df.csv', index_col='Date')    
    # training_prices_x = {}
    # training_prices_y = {}
    live_prices_x = {}
    live_prices_y = {}
    live_pred_x = {}
    pred_live_30 = df.tail(noofdays_lag_live)
    # live_pred_x = {}
    for col in df.columns:
        company_live = df[col].to_numpy()
        company_live_30 = pred_live_30[col].to_numpy()
        company_live_x = [company_live[i:i+15] for i in range(len(company_live)-15)]
        company_live_y = [company_live[i+1] for i in range(14,len(company_live)-1)]
        # company_live_pred_x = [company_live[i:i+30+1] for i in range(len(company_live)-30+1)]
        company_live_pred_x= [company_live_30[i:i+15] for i in range(len(company_live)-15)]
        live_prices_x[col] = company_live_x
        live_prices_y[col] = company_live_y
        live_pred_x[col] = [company_live_pred_x[0]]
        # live_pred_x
    # ----- ONE DAY PREDICTION LIVE ------------
    next_day_preds ={}
    for col in df.columns:
        bst = xgb.XGBRegressor(max_depth=xg_max_depth, learning_rate=xg_learning_rate,
                               reg_lambda=xg_reg_lambda, subsample=xg_subsample, grow_policy=xg_grow_policy)
        bst = bst.fit(live_prices_x[col],live_prices_y[col])
        next_day_preds[col] = bst.predict(live_pred_x[col])
    # next_day_preds    
    next_day_preds_df = pd.DataFrame.from_dict(next_day_preds)
    next_day_preds_df.index = ['pred']*1
    # next_day_preds_df
    next_day_preds_df.to_csv('/home/edsharecourse/projdatabucket/next_day_preds_df.csv', index=True)
    with open('/home/edsharecourse/projdatabucket/saved_dictionary.pkl', 'wb') as f_next_day_preds:
        pickle.dump(next_day_preds, f_next_day_preds)
