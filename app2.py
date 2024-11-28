import os
from dotenv import load_dotenv
import io
import uuid
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't require a GUI
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import traceback
import logging
import requests
import json
import functools
from flask import Flask, request, jsonify, render_template, send_file, after_this_request
import pandas as pd
import numpy as np
import difflib
import re
import unicodedata

# Load environment variables from .env file
load_dotenv()

# Now you can access the API key as an environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_gpt_response(user_input):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    if 'choices' not in result:
        raise ValueError(f"Unexpected API response: {result}")

    return result['choices'][0]['message']['content']

# Load the CSV file (add this near the top of the file, after other imports)
jstock_df = pd.read_csv('jstock.csv')

def normalize_japanese(text):
    return unicodedata.normalize('NFKC', text)

@functools.lru_cache(maxsize=100)
def get_info_from_gpt(user_input):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    
    today = datetime.now().strftime("%Y-%m-%d")
    user_input = normalize_japanese(user_input)
    
    logger.debug(f"Processing user input: {user_input}")
    
    initial_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": f"""You are a financial assistant. Today's date is {today}. 
            Determine if the query is about a chart, comparative chart, or general stock information. 
            If it's a chart request, provide the ticker, start_date, and end_date. 
            If it's a comparative chart request, provide all tickers mentioned, start_date, and end_date.
            For general queries, just provide the ticker. 
            If a Japanese company or stock is mentioned, identify the 4-digit ticker and append '.T' to it.
            Respond in JSON format with keys 'query_type' (either 'chart', 'comparative_chart', or 'info'), 'tickers' (list for comparative_chart, single string otherwise), and if applicable, 'start_date' and 'end_date'. 
            For 'latest' or current data, use today's date. 
            The user may ask questions in Japanese, so please interpret accordingly."""},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 200,
        "temperature": 0.3
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=initial_data, headers=headers)
    response.raise_for_status()
    result = response.json()
    content = result['choices'][0]['message']['content']
    logger.debug(f"Initial GPT response: {content}")
    info = json.loads(content)

    # Normalize the key to 'ticker' if 'tickers' is present
    if 'tickers' in info and 'ticker' not in info:
        info['ticker'] = info['tickers']
        del info['tickers']

    if info['query_type'] in ['chart', 'comparative_chart']:
        # Post-process the date for YTD queries
        if 'ytd' in user_input.lower():
            current_year = datetime.now().year
            info['start_date'] = f"{current_year}-01-01"
            info['end_date'] = 'latest'
    elif info['query_type'] == 'info':
        try:
            ticker = info.get('ticker', info.get('tickers'))
            if not ticker:
                raise ValueError("No ticker found in the response")
            
            stock_data = get_stock_data(ticker)
            
            # Use GPT to interpret the stock data
            response = interpret_stock_data_with_gpt(user_input, stock_data)
            
            info['response'] = response
        except Exception as e:
            logger.error(f"Error processing info query: {str(e)}")
            logger.error(traceback.format_exc())
            return get_gpt_fallback_response(user_input)

    logger.debug(f"Processed GPT response: {info}")
    return info

def interpret_stock_data_with_gpt(user_input, stock_data):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    
    prompt = f"""Given the following stock data:
    {stock_data}
    
    Please answer the following user query:
    {user_input}
    
    Provide a concise and informative answer based on the given stock data."""

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a financial assistant. Interpret the given stock data to answer the user's query accurately and concisely."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    if 'choices' not in result:
        raise ValueError(f"Unexpected API response: {result}")

    return result['choices'][0]['message']['content'].strip()

def get_gpt_fallback_response(user_input):
    try:
        response = get_gpt_response(user_input)
        return jsonify({"message": response})
    except Exception as e:
        logger.error(f"Error getting GPT fallback response: {str(e)}")
        return jsonify({"message": "I'm sorry, but I couldn't process your request at this time. Please try again later."}), 500

def get_gpt_response(user_input):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specializing in financial and stock market information. Provide a complete response within 250 tokens."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    if 'choices' not in result:
        raise ValueError(f"Unexpected API response: {result}")

    return result['choices'][0]['message']['content'].strip()

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    
    def safe_serialize(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.reset_index().to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.int64, np.float64)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_serialize(item) for item in obj]
        else:
            try:
                return json.dumps(obj)
            except:
                return str(obj)

    data = {
        "info": safe_serialize(stock.info),
        # "history": safe_serialize(stock.history(period="1mo")),
        # "actions": safe_serialize(stock.actions),
        # "dividends": safe_serialize(stock.dividends),
        # "splits": safe_serialize(stock.splits),
        # "financials": safe_serialize(stock.financials),
        # "major_holders": safe_serialize(stock.major_holders),
        # "institutional_holders": safe_serialize(stock.institutional_holders),
        # "recommendations": safe_serialize(stock.recommendations)
    }
    
    return json.dumps(data)

def get_stock_info(ticker, metric):
    stock = yf.Ticker(ticker)
    
    if metric in ['market cap', 'market capitalization']:
        value = stock.info.get('marketCap', 'N/A')
        if isinstance(value, (int, float)):
            return f"${value:,.0f}"
        return value
    elif metric in ['dividend yield', 'yield']:
        value = stock.info.get('dividendYield', 'N/A')
        if isinstance(value, float):
            return f"{value * 100:.2f}%"
        return value
    elif metric in ['pe ratio', 'price to earnings ratio']:
        value = stock.info.get('trailingPE', 'N/A')
        if isinstance(value, float):
            return f"{value:.2f}"
        return value
    elif metric in ['price', 'current price']:
        value = stock.history(period="1d")['Close'].iloc[-1]
        return f"${value:.2f}"
    elif metric in ['volume', 'trading volume']:
        value = stock.info.get('volume', 'N/A')
        if isinstance(value, (int, float)):
            return f"{value:,}"
        return value
    else:
        return f"Information about '{metric}' is not available in our stock data."

def get_stock_chart(ticker, start_date, end_date):
    def parse_date(date_str):
        if date_str.lower() == 'latest':
            return datetime.now()
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        logger.debug(f"Parsed date {date_str} to {parsed_date}")
        return parsed_date

    start = parse_date(start_date)
    end = parse_date(end_date)

    logger.debug(f"Fetching stock data for {ticker} from {start} to {end}")
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)

    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist['Close'])
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img

def get_comparative_stock_chart(tickers, start_date, end_date):
    def parse_date(date_str):
        if date_str.lower() == 'latest':
            return datetime.now()
        return datetime.strptime(date_str, '%Y-%m-%d')

    start = parse_date(start_date)
    end = parse_date(end_date)

    plt.figure(figsize=(12, 6))

    for ticker in tickers:
        logger.debug(f"Fetching stock data for {ticker} from {start} to {end}")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)
        
        # Index the prices to 100 at the start date
        indexed_prices = hist['Close'] / hist['Close'].iloc[0] * 100
        
        plt.plot(hist.index, indexed_prices, label=ticker)

    plt.title(f"Comparative Stock Performance (Indexed to 100)")
    plt.xlabel("Date")
    plt.ylabel("Indexed Price")
    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img

@app.route('/process-query', methods=['POST'])
def process_query():
    data = request.get_json()
    user_input = data['input']
    
    logger.debug(f"Received query: {user_input}")
    
    try:
        info = get_info_from_gpt(user_input)
        logger.debug(f"get_info_from_gpt returned: {info}")
        
        if info['query_type'] == 'chart':
            logger.debug("Recognized as a chart query")
            ticker = info['ticker']
            start_date = info['start_date']
            end_date = info['end_date']
            logger.debug(f"Generating chart for {ticker} from {start_date} to {end_date}")
            
            try:
                logger.debug(f"Generating chart for {ticker} from {start_date} to {end_date}")
                img = get_stock_chart(ticker, start_date, end_date)
                img_path = f"temp_{ticker}_{uuid.uuid4().hex[:8]}_chart.png"
                with open(img_path, 'wb') as f:
                    f.write(img.getvalue())
                return jsonify({
                    "message": f"Chart for {ticker} from {start_date} to {end_date} has been generated.",
                    "image_path": img_path
                })
            except Exception as e:
                logger.error(f"Error generating stock chart: {str(e)}")
                logger.error(traceback.format_exc())
                return get_gpt_fallback_response(user_input)
        
        elif info['query_type'] == 'comparative_chart':
            logger.debug("Recognized as a comparative chart query")
            tickers = info.get('tickers', info.get('ticker', []))  # Try both 'tickers' and 'ticker'
            start_date = info['start_date']
            end_date = info['end_date']
            logger.debug(f"Generating comparative chart for {tickers} from {start_date} to {end_date}")
            
            try:
                img = get_comparative_stock_chart(tickers, start_date, end_date)
                img_path = f"temp_comparative_{uuid.uuid4().hex[:8]}_chart.png"
                with open(img_path, 'wb') as f:
                    f.write(img.getvalue())
                return jsonify({
                    "message": f"Comparative chart for {', '.join(tickers)} from {start_date} to {end_date} has been generated.",
                    "image_path": img_path
                })
            except Exception as e:
                logger.error(f"Error generating comparative stock chart: {str(e)}")
                logger.error(traceback.format_exc())
                return get_gpt_fallback_response(user_input)
        
        elif info['query_type'] == 'info':
            return jsonify({"message": info['response']})
        
        else:
            return get_gpt_fallback_response(user_input)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return get_gpt_fallback_response(user_input)

def get_gpt_fallback_response(user_input):
    try:
        response = get_gpt_response(user_input)
        return jsonify({"message": response})
    except Exception as e:
        logger.error(f"Error getting GPT fallback response: {str(e)}")
        return jsonify({"message": "I'm sorry, but I couldn't process your request at this time. Please try again later."}), 500

def get_gpt_response(user_input):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specializing in financial and stock market information. Provide a complete response within 250 tokens."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    if 'choices' not in result:
        raise ValueError(f"Unexpected API response: {result}")

    return result['choices'][0]['message']['content'].strip()

def get_gpt_fallback_response(user_input):
    try:
        response = get_gpt_response(user_input)
        return jsonify({"message": response})
    except Exception as e:
        logger.error(f"Error getting GPT fallback response: {str(e)}")
        return jsonify({"message": "I'm sorry, but I couldn't process your request at this time. Please try again later."}), 500

def get_gpt_response(user_input):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    
    # Calculate available tokens for the response
    system_message = "You are a helpful assistant specializing in financial and stock market information. Provide a complete response within 250 tokens."
    estimated_prompt_tokens = len(system_message.split()) + len(user_input.split())
    max_response_tokens = 300  # Adjust this value as needed

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": max_response_tokens,
        "temperature": 0.7
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    if 'choices' not in result:
        raise ValueError(f"Unexpected API response: {result}")

    return result['choices'][0]['message']['content'].strip()

def process_stock_info(user_input, stock_data):
    # Parse the JSON string into a Python dictionary
    data = json.loads(stock_data)
    
    # Extract relevant information based on the user's query
    if 'pe ratio' in user_input.lower() or 'per ratio' in user_input.lower():
        pe_ratio = data['info'].get('trailingPE', 'N/A')
        return f"The P/E ratio for {data['info'].get('symbol')} is {pe_ratio}."
    elif 'dividend yield' in user_input.lower():
        div_yield = data['info'].get('dividendYield', 'N/A')
        if div_yield != 'N/A':
            try:
                div_yield = float(div_yield)
                div_yield = f"{div_yield * 100:.2f}%"
            except (ValueError, TypeError):
                # If conversion to float fails, keep the original value
                pass
        return f"The dividend yield for {data['info'].get('symbol')} is {div_yield}."
    elif 'market cap' in user_input.lower():
        market_cap = data['info'].get('marketCap', 'N/A')
        if market_cap != 'N/A':
            try:
                market_cap = float(market_cap)
                market_cap = f"${market_cap:,.0f}"
            except (ValueError, TypeError):
                # If conversion to float fails, keep the original value
                pass
        return f"The market capitalization for {data['info'].get('symbol')} is {market_cap}."
    else:
        # If we can't determine a specific metric, return a general overview
        symbol = data['info'].get('symbol', 'N/A')
        name = data['info'].get('longName', 'N/A')
        price = data['info'].get('currentPrice', 'N/A')
        if price != 'N/A':
            try:
                price = float(price)
                price = f"${price:.2f}"
            except (ValueError, TypeError):
                # If conversion to float fails, keep the original value
                pass
        return f"{name} ({symbol}) - Current Price: {price}"

@app.route('/get-chart/<path:img_path>')
def get_chart(img_path):
    try:
        return send_file(img_path, mimetype='image/png')
    finally:
        @after_this_request
        def remove_file(response):
            try:
                os.remove(img_path)
                logger.debug(f"Removed temporary file: {img_path}")
            except Exception as error:
                logger.error(f"Error removing file {img_path}: {error}")
            return response

if __name__ == '__main__':
    app.run(debug=True)