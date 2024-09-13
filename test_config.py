from flask import Flask
from config import Config

import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config.from_object(Config)

print("API Key from environment:", os.environ.get('ALPHA_VANTAGE_API_KEY'))
# Usage
symbol = 'MSFT'  # Replace with desired stock symbol
api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')

def get_earnings_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    
    if 'annualReports' not in data:
        print("Error: Unable to fetch data. Check your API key and symbol.")
        return None

    annual_reports = data['annualReports']
    
    df = pd.DataFrame(annual_reports)
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df.set_index('fiscalDateEnding', inplace=True)
    
    numeric_columns = ['totalRevenue', 'operatingIncome', 'netIncome']
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    df.sort_index(ascending=True, inplace=True)
    
    return df

def plot_earnings(df, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = df.index.year
    revenues = df['totalRevenue'] / 1e9  # Convert to billions
    operating_profits = df['operatingIncome'] / 1e9  # Convert to billions
    
    x = range(len(years))
    width = 0.35
    
    # Morgan Stanley blue color scheme
    revenue_color = '#0033A0'  # Dark blue
    profit_color = '#75B2DD'   # Light blue
    
    ax.bar([i - width/2 for i in x], revenues, width, label='Revenue', color=revenue_color, alpha=0.8)
    ax.bar([i + width/2 for i in x], operating_profits, width, label='Operating Income', color=profit_color, alpha=0.8)
    
    ax.set_ylabel('Amount (Billion USD)')
    ax.set_title(f'{symbol} Revenue and Operating Income by Year')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    
    # Add value labels on top of each bar
    for i, v in enumerate(revenues):
        ax.text(i - width/2, v, f'${v:.1f}B', ha='center', va='bottom')
    for i, v in enumerate(operating_profits):
        ax.text(i + width/2, v, f'${v:.1f}B', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


if api_key is None:
    print("Please set the ALPHA_VANTAGE_API_KEY environment variable.")
else:
    earnings_df = get_earnings_data(symbol, api_key)
    if earnings_df is not None:
        print(f"\nEarnings Data for {symbol}:")
        print(earnings_df[['totalRevenue', 'operatingIncome', 'netIncome']])
        
        plot_earnings(earnings_df, symbol)