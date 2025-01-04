from datetime import datetime, timedelta
from textwrap import dedent
import time

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization


####################################################
# DEFINE PYTHON FUNCTIONS
####################################################

# -------- parameters ---------------------
count = 0  # --- this is example of global variable in the def

tickers_pull = ['XOM', 'HON', 'RTX', 'AMZN', 'PEP', 'UNH', 'JNJ', 'V', 'NVDA', 'AAPL', 'MSFT', 'GOOGL']
# tickers_pull = ['XOM', 'HON', 'RTX', 'AMZN', 'PEP', 'UNH']
# ======= initial parameters ============

noofdays_test = 250
# ======= lag 30 days for live runs ============
noofdays_lag_live = 30

# Get today's date
today_real = date.today()
# after mid night 
# today = today_real - timedelta(days = 1)
today = today_real
# print("Today is: ", today)

# Yesterday date
yesterday = today - timedelta(days = 1)
# print("Yesterday was: ", yesterday)

# --- fixed mcaps dec 12
mcaps = {'XOM': 398158000000,'HON': 132107000000,'RTX': 117750000000,'AMZN': 1508000000000,'PEP': 230729000000,'UNH': 502863000000,'JNJ': 373273000000, 'V': 527197000000,'NVDA': 1152000000000,'AAPL': 3004000000000,'MSFT': 2760000000000,'GOOGL': 1676000000000}

# ---- 3 diff --- python commands ----
# ------  def correct_sleeping_function():   def sleeping_cmd_fn():   def count_cmd_fn():
# -------  def print_cmd_fn():  def wrong_sleeping_function():

# ----- regular functions ----------
def convert_date(x):
    func_date = dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d')
    return func_date


def convert_date_time(x):
    func_date_time = dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
    return func_date_time


def categorise(row):
    if row['pred'] > 0 and row['sentiment'] ==1:
        return 1
    elif row['pred'] < 0 and row['sentiment'] ==-1:
        return 1
    elif row['pred'] > 0 and row['sentiment'] ==-1:
        return 0
    elif row['pred'] < 0 and row['sentiment'] ==1:
        return 0
    return 0.5
