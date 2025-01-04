import pandas as pd
from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models, objective_functions
from datetime import date
import pickle
from modules.utils import categorise

# Define constants
today = date.today()

def step4_bl_weight():
    prices_BL = pd.read_csv('/home/edsharecourse/projdatabucket/prices.csv', index_col='Date')    
    df_market_csv_read = pd.read_csv('/home/edsharecourse/projdatabucket/market_prices_BL_raw.csv', index_col='Date')
    market_prices_BL_raw = df_market_csv_read[df_market_csv_read.columns[0]]    
    next_day_preds_df = pd.read_csv('/home/edsharecourse/projdatabucket/next_day_preds_df.csv', index_col=['Unnamed: 0'])
    sentiment_daily = pd.read_csv('/home/edsharecourse/projdatabucket/sentiment_daily.csv', index_col=['Unnamed: 0'])    
    with open('/home/edsharecourse/projdatabucket/saved_dictionary.pkl', 'rb') as f_next_day_preds:
        next_day_preds = pickle.load(f_next_day_preds)    

    next_day_preds_df_T = next_day_preds_df.T
    sentiment_daily_T = sentiment_daily.T

    # Combine predictions and sentiment
    combine_pred_sentiment = next_day_preds_df_T.join(sentiment_daily_T)
    combine_pred_sentiment['confidence'] = combine_pred_sentiment.apply(lambda row: categorise(row), axis=1)
    confidence = combine_pred_sentiment['confidence'].T

    # Black-Litterman optimization
    S_BL = risk_models.CovarianceShrinkage(prices_BL).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(market_prices_BL_raw)
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S_BL)

    bl = BlackLittermanModel(
        S_BL, pi=market_prior, absolute_views=next_day_preds, 
        omega="idzorek", view_confidences=confidence
    )
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()

    ef = EfficientFrontier(ret_bl, S_bl)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    weights = ef.clean_weights()

    # Save weights
    weights_df = pd.DataFrame(weights, index=[today])
    weights_df.to_csv('/home/edsharecourse/projdatabucket/weights_ML2_GBReg_df.csv')

    # Portfolio performance analysis
    w_hist = pd.read_csv('/home/edsharecourse/projdatabucket/w_hist.csv', index_col='Date')
    df = pd.read_csv('/home/edsharecourse/projdatabucket/df.csv', index_col='Date')
    df_ret_calc = df.add_prefix('ret_')
    w_hist_calc = w_hist.add_prefix('w_')

    port = w_hist_calc.merge(df_ret_calc, how='inner', left_index=True, right_index=True)
    for stock in prices_BL.columns:
        port[f'x_{stock}'] = port[f'w_{stock}'] * port[f'ret_{stock}']
        port.drop(columns=[f'w_{stock}', f'ret_{stock}'], inplace=True)

    port_wavg_ret = port.sum(axis=1)
    sharpe_ratio_port = port_wavg_ret.mean() / port_wavg_ret.std()

    rm_ret = market_prices_BL_raw.pct_change().dropna()
    rm = port_wavg_ret.to_frame().join(rm_ret.to_frame())
    sharpe_summary = pd.DataFrame({
        'Date': [port_wavg_ret.index.max()],
        'Sharpe Ratio Portfolio (30 days)': [sharpe_ratio_port],
        'Sharpe Ratio S&P (30 days)': [rm['Adj Close'].mean() / rm['Adj Close'].std()]
    }).set_index('Date')
    sharpe_summary.to_csv('/home/edsharecourse/projdatabucket/sharpe_summary_csv.csv')

    return_summary = pd.DataFrame({
        'Date': [port_wavg_ret.index.max()],
        'Portfolio Avg Return (30 days)': [port_wavg_ret.mean()],
        'Portfolio Return Std (30 days)': [port_wavg_ret.std()],
        'S&P Avg Return (30 days)': [rm['Adj Close'].mean()],
        'S&P Return Std (30 days)': [rm['Adj Close'].std()]
    }).set_index('Date')
    return_summary.to_csv('/home/edsharecourse/projdatabucket/return_summary_csv.csv')

    # Update weights history
    weights_df.index.name = 'Date'
    weights_df.index = pd.to_datetime(weights_df.index).strftime('%m/%d/%Y')
    w_hist_app = pd.concat([w_hist, weights_df])
    w_hist = w_hist_app.tail(30)
    w_hist.to_csv('/home/edsharecourse/projdatabucket/w_hist.csv', index_label='Date')
