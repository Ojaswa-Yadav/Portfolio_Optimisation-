from datetime import datetime

def convert_date(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

def categorize_predictions(row):
    if row['pred'] > 0 and row['sentiment'] == 1:
        return 1
    elif row['pred'] < 0 and row['sentiment'] == -1:
        return 1
    elif row['pred'] > 0 and row['sentiment'] == -1:
        return 0
    elif row['pred'] < 0 and row['sentiment'] == 1:
        return 0
    return 0.5
