from flask import Flask, render_template, url_for, redirect, g, request, send_file, make_response, flash, session, jsonify, after_this_request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, Email, Optional
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from config import Config
from datetime import datetime, date
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.switch_backend('Agg')
import matplotlib
matplotlib.use('Agg')
import io
import base64
import pandas as pd
from yahoo_fin import stock_info as si
import pandas_datareader as pdr
import numpy as np
from datetime import datetime, timedelta
import pytz
import csv
from flask_caching import Cache
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.consumer.storage.sqla import SQLAlchemyStorage
from flask_dance.consumer import oauth_authorized
from sqlalchemy.orm.exc import NoResultFound
import os
from dotenv import load_dotenv
from translations import translations
import uuid
import traceback
import requests
import json
import functools
import unicodedata
import logging

load_dotenv()  # This loads the variables from .env
# Set up logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

app.config['GOOGLE_OAUTH_CLIENT_ID'] = os.environ.get('GOOGLE_OAUTH_CLIENT_ID')
app.config['GOOGLE_OAUTH_CLIENT_SECRET'] = os.environ.get('GOOGLE_OAUTH_CLIENT_SECRET')

google_bp = make_google_blueprint(
    client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
    scope=["profile", "email"]
)
app.register_blueprint(google_bp, url_prefix="/login")

@app.before_request
def before_request():
    g.language = session.get('language', 'en')  # Default to English

@app.context_processor
def inject_language():
    return dict(language=g.language, translations=translations)

import logging

@login_manager.user_loader
def load_user(user_id):
    logging.info(f"Attempting to load user with id: {user_id}")
    if user_id is not None:
        try:
            user = User.query.get(int(user_id))
            logging.info(f"Loaded user: {user}")
            return user
        except ValueError:
            logging.warning(f"Invalid user_id: {user_id}")
            return None
    logging.warning("user_id is None")
    return None

# Helper functions
@cache.memoize(timeout=900)
def fetch_latest_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            latest_price = history['Close'].iloc[-1]
            return round(float(latest_price), 2) if latest_price is not None else 0.0
        else:
            print(f"No data returned for {ticker}")
            return 0.0
    except Exception as e:
        print(f"Error fetching latest price for {ticker}: {e}")
        return 0.0

@cache.memoize(timeout=900)
def fetch_daily_return(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="5d")  # Fetch the last 5 days' price data
        if len(history) < 2:
            return 0.0  # Not enough data to calculate daily return
        today_close = history['Close'].iloc[-1]
        yesterday_close = history['Close'].iloc[-2]
        daily_return = ((today_close - yesterday_close) / yesterday_close) * 100
        return float(daily_return)  # Round to 2 decimal places
    except Exception as e:
        return 0.0

def fetch_forwardPE(ticker):
    stock = yf.Ticker(ticker)
    quote_table = stock.info
    forward_pe = quote_table.get('forwardPE')
    return float(forward_pe) if forward_pe is not None else 0.0  # Check for None

def fetch_divyiled(ticker):
    stock = yf.Ticker(ticker)
    quote_table = stock.info
    div_yield = quote_table.get('dividendYield')
    # Check if div_yield is None, and return "-" if it is
    if div_yield is None:
        return 0.0
    # If div_yield is a valid number, multiply by 100 to get the percentage
    return float(div_yield * 100)

def calculate_returns(purchase_price, latest_price):
    if purchase_price == 0:  # Prevent division by zero
        return 0.0
    return float(((latest_price - purchase_price) / purchase_price) * 100)  # Always return a float

def calculate_portfolio_return(stocks_data):
    total_investment = sum(stock.purchase_price * stock.shares for stock in stocks_data)
    total_current_value = sum(stock.latest_price * stock.shares for stock in stocks_data)
    portfolio_return = ((total_current_value - total_investment) / total_investment) * 100
    return float(portfolio_return) 

def calculate_portfolio_return_withdiv(stocks_data):
    total_investment = sum(stock.purchase_price * stock.shares for stock in stocks_data)
    total_current_value = sum(stock.latest_price * stock.shares for stock in stocks_data)
    total_div = sum(stock.latest_price * stock.div_yield for stock in stocks_data)
    portfolio_return_withdiv = ((total_current_value + total_div - total_investment) / total_investment) * 100
    return float(portfolio_return_withdiv)

def calculate_portfolio_metrics(stocks_data):
    # Get the list of tickers
    tickers = [stock.ticker for stock in stocks_data]
    
    # Calculate portfolio weights
    total_value = sum(stock.latest_price * stock.shares for stock in stocks_data)
    weights = np.array([stock.latest_price * stock.shares / total_value for stock in stocks_data])
    
    # Set timezone-aware end date in JST and make start date based on the last 3 years
    jst = pytz.timezone('Asia/Tokyo')
    end_date = datetime.now(jst).replace(tzinfo=None)  # Make end_date timezone-naive for consistency
    start_date = end_date - timedelta(days=5*365)
    
    # Fetch data for stocks and Nikkei 225 (market benchmark)
    tickers.append('^N225')  # Nikkei 225 ticker
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Make sure the stock data is timezone-naive
    data.index = data.index.tz_localize(None)
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Fetch Japanese 10-year government bond yield as risk-free rate
    try:
        jgb_data = pdr.get_data_fred('IRLTLT01JPM156N', start=start_date, end=end_date)
        jgb_data.index = jgb_data.index.tz_localize(None)  # Make the FRED data timezone-naive
        rf_rate = jgb_data['IRLTLT01JPM156N'] / 100 / 252  # Convert annual rate to daily rate
    except Exception as e:
        print(f"Error fetching Japanese risk-free rate: {e}")
        rf_rate = pd.Series(0.001 / 252, index=returns.index)  # Use 0.1% annual rate as fallback
    
    # Align indexes and fill missing values
    aligned_data = pd.concat([returns, rf_rate], axis=1).fillna(method='ffill')
    stock_returns = aligned_data.iloc[:, :-1]
    rf_rate = aligned_data.iloc[:, -1]
    
    # Calculate excess returns
    excess_returns = stock_returns.sub(rf_rate, axis=0)
    
    # Separate stock returns and market returns
    stock_excess_returns = excess_returns.iloc[:, :-1]
    market_excess_returns = excess_returns.iloc[:, -1]
    
    # Calculate portfolio excess returns
    portfolio_excess_returns = stock_excess_returns.dot(weights)
    
    # Calculate beta
    covariance = portfolio_excess_returns.cov(market_excess_returns)
    market_variance = market_excess_returns.var()
    beta = covariance / market_variance
    
    # Calculate alpha (Jensen's Alpha)
    expected_excess_return = beta * market_excess_returns.mean() * 252
    alpha = portfolio_excess_returns.mean() * 252 - expected_excess_return
    
    # Calculate Sharpe ratio
    sharpe_ratio = portfolio_excess_returns.mean() / portfolio_excess_returns.std() * np.sqrt(252)
    
    return {
        'beta': beta,
        'alpha': alpha,
        'sharpe_ratio': sharpe_ratio
    }

from datetime import datetime, date
@cache.memoize(timeout=3600)
def calculate_income_gain_pct(ticker, purchase_date, purchase_price):
    try:
        stock_data = yf.Ticker(ticker)
        
        # Get dividend history
        dividends = stock_data.dividends
        dividends.index = dividends.index.tz_localize(None)
        
        # Convert purchase_date to datetime if it's a date
        if isinstance(purchase_date, date):
            purchase_date = datetime.combine(purchase_date, datetime.min.time())
        
        # Filter dividends after the purchase date
        dividends_after_purchase = dividends[dividends.index >= purchase_date]
        
        # Calculate total dividends received (Income Gain)
        total_dividends_received = dividends_after_purchase.sum()
        
        # Income Gain Return % calculation
        income_gain_return_pct = (total_dividends_received / purchase_price) * 100
        
        return float(income_gain_return_pct)  # Convert to Python float
    except Exception as e:
        print(f"Error calculating income gain for {ticker}: {e}")
        return 0.0

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(120), unique=True, nullable=True)  # Add this line
    password = db.Column(db.String(128), nullable=False)
    
    def get_id(self):
        return str(self.id)

class Stock(db.Model):
    __table_args__ = (
        db.Index('idx_user_ticker', 'user_id', 'ticker'),
    )
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(100), nullable=True)
    purchase_price = db.Column(db.Float, nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    purchase_date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date())
    latest_price = db.Column(db.Float, nullable=True)
    daily_return = db.Column(db.Float, nullable=True, default=0.0)
    return_performance = db.Column(db.Float, nullable=True, default=0.0)
    forward_pe = db.Column(db.Float, nullable=True, default=0.0)
    div_yield = db.Column(db.Float, nullable=True, default=0.0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    income_gain_pct = db.Column(db.Float, nullable=True, default=0.0)

class StockMemo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    memo = db.Column(db.Text)

class OAuth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(50), nullable=False)
    provider_user_id = db.Column(db.String(256), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id), nullable=False)
    user = db.relationship(User)

google_bp.storage = SQLAlchemyStorage(OAuth, db.session, user=current_user)

@oauth_authorized.connect_via(google_bp)
def google_logged_in(blueprint, token):
    if not token:
        flash("Failed to log in with Google.", category="error")
        return False

    resp = blueprint.session.get("/oauth2/v1/userinfo")
    if not resp.ok:
        msg = "Failed to fetch user info from Google."
        flash(msg, category="error")
        return False

    google_info = resp.json()
    google_user_id = str(google_info["id"])

    oauth = OAuth.query.filter_by(
        provider=blueprint.name,
        provider_user_id=google_user_id
    ).first()

    if oauth is None:
        # Check if a user with this email already exists
        existing_user = User.query.filter_by(email=google_info["email"]).first()
        if existing_user:
            user = existing_user
        else:
            # Create a new user
            username = google_info["email"].split('@')[0]
            # Ensure username is unique
            base_username = username
            counter = 1
            while User.query.filter_by(username=username).first():
                username = f"{base_username}{counter}"
                counter += 1
            
            hashed_password = bcrypt.generate_password_hash('temporary_password').decode('utf-8')
            user = User(
                username=username,
                email=google_info["email"],
                password=hashed_password
            )
            db.session.add(user)
            db.session.flush()  # This assigns an id to the user

        oauth = OAuth(
            provider=blueprint.name,
            provider_user_id=google_user_id,
            user_id=user.id
        )
        db.session.add(oauth)
        db.session.commit()
    else:
        user = oauth.user

    login_user(user)
    flash("Successfully signed in with Google.")

    # Redirect to dashboard
    return redirect(url_for('dashboard'))

@app.route("/login/google")
def login_google():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v1/userinfo")
    assert resp.ok, resp.text
    return "You are {email} on Google".format(email=resp.json()["email"])

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    email = StringField('Email', validators=[Optional(), Email()], render_kw={"placeholder": "Email (optional)"})
    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

    def validate_email(self, email):
        if email.data:
            existing_user_email = User.query.filter_by(email=email.data).first()
            if existing_user_email:
                raise ValidationError('That email is already registered. Please use a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/set_language/<language>')
def set_language(language):
    if language in ['en', 'jp']:
        session['language'] = language
    return redirect(request.referrer or url_for('home'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter((User.username == form.username.data) | 
                                 (User.email == form.username.data)).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        if form.email.data:
            new_user.email = form.email.data
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        ticker = request.form['ticker']
        purchase_price = float(request.form['purchase_price'])
        shares = int(request.form['num_shares'])
        purchase_date = datetime.strptime(request.form['purchase_date'], '%Y-%m-%d').date()

        latest_price = fetch_latest_price(ticker)
        return_performance = calculate_returns(purchase_price, latest_price)
        forward_pe = fetch_forwardPE(ticker)
        div_yield = fetch_divyiled(ticker)
        daily_return = fetch_daily_return(ticker)

        # Fetch company name
        company_name = yf.Ticker(ticker).info.get('shortName', 'N/A')

        stock = Stock.query.filter_by(ticker=ticker, user_id=current_user.id).first()
        if stock:
            # Update existing stock
            total_investment = (stock.purchase_price * stock.shares) + (purchase_price * shares)
            total_shares = stock.shares + shares
            new_average_purchase_price = total_investment / total_shares
            
            # Calculate weighted average purchase date
            existing_weighted_date = (stock.purchase_date.toordinal() * stock.shares) + (purchase_date.toordinal() * shares)
            weighted_average_date = date.fromordinal(existing_weighted_date // total_shares)
            
            stock.purchase_price = new_average_purchase_price
            stock.shares = total_shares
            stock.latest_price = latest_price
            stock.daily_return = daily_return
            stock.return_performance = calculate_returns(new_average_purchase_price, latest_price)
            stock.forward_pe = forward_pe
            stock.div_yield = div_yield
            stock.company_name = company_name  # Update company name
            stock.purchase_date = weighted_average_date  # Update to weighted average date

        else:
            # Create a new stock entry
            new_stock = Stock(
                ticker=ticker,
                company_name=company_name,  # Add company name
                purchase_price=purchase_price,
                purchase_date=purchase_date,  # Add this line
                shares=shares,
                latest_price=latest_price,
                daily_return=daily_return,
                return_performance=return_performance,
                forward_pe=forward_pe,
                div_yield=div_yield,
                user_id=current_user.id
            )
            db.session.add(new_stock)

        db.session.commit()
        return redirect(url_for('dashboard'))
    
    stock_data = Stock.query.filter_by(user_id=current_user.id).all()
    
    # Batch fetch all stock data
    tickers = [stock.ticker for stock in stock_data]
    if tickers:
        batch_data = fetch_batch_stock_data(tickers)
        
        # Update stock objects with batch data
        for stock in stock_data:
            if stock.ticker in batch_data:
                stock_info = batch_data[stock.ticker]
                stock.latest_price = stock_info['latest_price']
                stock.daily_return = stock_info['daily_return']
                stock.return_performance = calculate_returns(stock.purchase_price, stock.latest_price)

    db.session.commit()

    portfolio_return = calculate_portfolio_return(stock_data) if stock_data else 0
    portfolio_return_withdiv = calculate_portfolio_return_withdiv(stock_data) if stock_data else 0
    total_cost = sum(stock.purchase_price * stock.shares for stock in stock_data)
    total_income_gain_pct = sum(stock.income_gain_pct * (stock.purchase_price * stock.shares) / total_cost for stock in stock_data) if total_cost > 0 else 0
    portfolio_return_withincome = portfolio_return + total_income_gain_pct
    
    return render_template('dashboard.html', 
                         stocks=stock_data,
                         portfolio_return=portfolio_return,
                         portfolio_return_withdiv=portfolio_return_withdiv,
                         portfolio_return_withincome=portfolio_return_withincome)

@app.route('/portfolio', methods=['GET', 'POST'])
@login_required
def portfolio():
    stock_data = Stock.query.filter_by(user_id=current_user.id).all()
    
    # Batch fetch all stock data at once
    tickers = [stock.ticker for stock in stock_data]
    if tickers:
        batch_data = fetch_batch_stock_data(tickers)
        batch_dividend_data = fetch_batch_dividend_data(tickers, min(stock.purchase_date for stock in stock_data))
        
        # Update stock objects with batch data
        for stock in stock_data:
            if stock.ticker in batch_data:
                stock_info = batch_data[stock.ticker]
                stock.latest_price = stock_info['latest_price']
                stock.daily_return = stock_info['daily_return']
                stock.return_performance = calculate_returns(stock.purchase_price, stock.latest_price)
                stock.income_gain_pct = float(calculate_income_gain_pct(stock.ticker, stock.purchase_date, stock.purchase_price))

    db.session.commit()

    # Calculate portfolio metrics
    total_cost = sum(stock.purchase_price * stock.shares for stock in stock_data)
    total_value = sum(stock.latest_price * stock.shares for stock in stock_data)
    portfolio_return = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # Calculate other metrics
    total_income_gain_pct = sum(stock.income_gain_pct * (stock.purchase_price * stock.shares) / total_cost for stock in stock_data) if total_cost > 0 else 0
    portfolio_return_withincome = portfolio_return + total_income_gain_pct
    total_income_gain = total_cost * total_income_gain_pct/100
    return_value = total_value - total_cost
    dividend_value = sum((stock.div_yield / 100) * stock.latest_price * stock.shares for stock in stock_data)
    
    # Calculate win rate
    num_stocks = len(stock_data)
    winning_stocks = sum(1 for stock in stock_data if stock.return_performance > 0)
    win_rate = (winning_stocks / num_stocks) * 100 if num_stocks > 0 else 0

    # Calculate holding periods
    today = date.today()
    holding_periods = [(stock, (today - stock.purchase_date).days) for stock in stock_data]
    
    if holding_periods:
        avg_days_held = sum(days for _, days in holding_periods) / len(holding_periods)
        longest_held = max(holding_periods, key=lambda x: x[1])
        longest_held_stock, longest_held_days = longest_held
        shortest_held = min(holding_periods, key=lambda x: x[1])
        shortest_held_stock, shortest_held_days = shortest_held
    else:
        avg_days_held = longest_held_days = shortest_held_days = 0
        longest_held_stock = shortest_held_stock = None

    # Create performance chart with original styling
    if stock_data:
        # Prepare performance data
        portfolio_performance_data = []
        for stock in stock_data:
            capital_gain = stock.return_performance
            income_gain = stock.income_gain_pct
            total_return = capital_gain + income_gain
            portfolio_performance_data.append((stock.company_name, capital_gain, income_gain, total_return))

        # Sort by total return
        portfolio_performance_data.sort(key=lambda x: x[3], reverse=True)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, max(6, len(portfolio_performance_data) * 0.4)))
        company_names, capital_gains, income_gains, total_returns = zip(*portfolio_performance_data)

        # Truncate long company names
        max_name_length = 20
        truncated_names = [name[:max_name_length] + '...' if len(name) > max_name_length else name for name in company_names]
        y_pos = range(len(company_names))
        
        # Plot bars with original styling
        bar_height = 0.8
        for i, (capital_gain, income_gain) in enumerate(zip(capital_gains, income_gains)):
            ax.barh(i, capital_gain, align='center', color='white', edgecolor='black', height=bar_height)
            end = max(0, capital_gain)
            ax.barh(i, income_gain, left=end, align='center', color='white', edgecolor='black', height=bar_height, hatch='///')

        # Style the chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(truncated_names)
        ax.invert_yaxis()
        ax.set_xlabel('Performance (%)')
        ax.set_title("Portfolio Performance (Capital Gain + Income Gain)")
        
        # Add legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Capital Gain'),
            Patch(facecolor='white', edgecolor='black', hatch='///', label='Income Gain')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        # Add total return labels
        for i, (capital_gain, income_gain) in enumerate(zip(capital_gains, income_gains)):
            total_return = capital_gain + income_gain
            ax.text(total_return + 0.5 if total_return >= 0 else total_return - 0.5, 
                    i, 
                    f'{total_return:.1f}%', 
                    va='center', 
                    ha='left' if total_return >= 0 else 'right', 
                    fontsize=12)

        # Final styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        plt.tight_layout()

        # Save chart
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        portfolio_performance_chart = base64.b64encode(buffer.getvalue()).decode()
        plt.close('all')
    else:
        portfolio_performance_chart = None

    return render_template('portfolio.html',
                         stocks=stock_data,
                         portfolio_return=portfolio_return,
                         portfolio_return_withincome=portfolio_return_withincome,
                         total_value=total_value,
                         return_value=return_value,
                         dividend_value=dividend_value,
                         num_stocks=num_stocks,
                         win_rate=win_rate,
                         avg_days_held=avg_days_held,
                         longest_held_stock=longest_held_stock,
                         longest_held_days=longest_held_days,
                         shortest_held_stock=shortest_held_stock,
                         shortest_held_days=shortest_held_days,
                         total_income_gain=total_income_gain,
                         portfolio_performance_chart=portfolio_performance_chart)

@app.route('/delete', methods=['POST'])
def delete():
    ticker = request.form['ticker']
    stock = Stock.query.filter_by(ticker=ticker, user_id=current_user.id).first()
    if stock:
        db.session.delete(stock)
        db.session.commit()
    return redirect(url_for('dashboard'))



# Update the stockan route
@app.route('/stockan', methods=['GET', 'POST'])
def stockan():
    ticker = request.args.get('ticker') or request.form.get('ticker')
    
    if ticker:
        stock_info = get_stock_info(ticker)
        
        # Load the memo if it exists
        memo = StockMemo.query.filter_by(ticker=ticker).first()
        if memo:
            stock_info['memo'] = memo.memo
        else:
            stock_info['memo'] = ""
        
        stock_info['ticker'] = ticker  # Ensure the ticker is included in stock_info
        
        return render_template('stockan.html', stock_info=stock_info, ticker=ticker)
    
    return render_template('stockan.html')

# Function to get the last trading day's performance
def get_last_trading_day_performance(ticker):
    try:
        # Set timezone to Japan Standard Time (JST)
        jst = pytz.timezone('Asia/Tokyo')
        end_date = datetime.now(jst)
        start_date = end_date - timedelta(days=5)  # Fetch 5 days to ensure we get the last trading day

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if len(hist) >= 2:
            last_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            last_date = hist.index[-1].strftime('%Y-%m-%d')
            return (last_close - prev_close) / prev_close * 100, last_date
        return 0, None
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return 0, None

@app.route('/today')
@login_required
# @cache.memoize(timeout=900)
def today():
    # Get user's stocks
    stock_data = Stock.query.filter_by(user_id=current_user.id).all()

    # Calculate portfolio performance
    portfolio_performance = 0
    total_value = sum(stock.latest_price * stock.shares for stock in stock_data)
    if total_value > 0:
        for stock in stock_data:
            stock_performance, _ = get_last_trading_day_performance(stock.ticker)
            stock_value = stock.latest_price * stock.shares
            portfolio_performance += stock_performance * (stock_value / total_value)

    # Get Nikkei 225 performance
    nikkei_performance, nikkei_date = get_last_trading_day_performance('^N225')
    sp_performance, sp_date = get_last_trading_day_performance('^GSPC')

    # Calculate individual stock performances
    stock_performances = []
    for stock in stock_data:
        performance, _ = get_last_trading_day_performance(stock.ticker)
        stock_performances.append((stock.company_name, performance))

    # Sort performances from best to worst
    stock_performances.sort(key=lambda x: x[1], reverse=True)

    # Create horizontal bar chart for stock performances
    fig, ax = plt.subplots(figsize=(8, max(4, len(stock_performances) * 0.4)))
    company_names, performances = zip(*stock_performances)

    # Truncate long company names
    max_name_length = 20
    truncated_names = [name[:max_name_length] + '...' if len(name) > max_name_length else name for name in company_names]

    y_pos = range(len(company_names))
    bars = ax.barh(y_pos, performances, align='center', color='white', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(truncated_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Performance (%)')
    ax.set_title("Today's Stock Performances")

    # Add performance labels to the end of each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        gap = 0.03
        ax.text(width + gap, bar.get_y() + bar.get_height()/2, f'{performances[i]:.2f}%', 
                ha='left', va='center', color='black', fontsize=12)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set background color to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    # Save plot to a base64 string for stock performance
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    performance_chart = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return render_template('today.html',
                           portfolio_performance=portfolio_performance,
                           nikkei_performance=nikkei_performance,
                           nikkei_date=nikkei_date,
                           sp_performance=sp_performance,
                           sp_date=sp_date,
                           performance_chart=performance_chart)  # Pass the new chart data

# Update the route for saving the memo
@app.route('/save_memo', methods=['POST'])
def save_memo():
    ticker = request.form['ticker']
    memo_text = request.form['memo']
    
    memo = StockMemo.query.filter_by(ticker=ticker).first()
    if memo:
        memo.memo = memo_text
    else:
        new_memo = StockMemo(ticker=ticker, memo=memo_text)
        db.session.add(new_memo)
    
    db.session.commit()
    
    # Redirect back to the stockan page with the ticker as a parameter
    return redirect(url_for('stockan', ticker=ticker))

def get_valuation_history(ticker):
    try:
        # Fetch valuation data
        valuation_data = si.get_stats_valuation(ticker)
        
        # Set the first column as index
        valuation_data = valuation_data.set_index('Unnamed: 0')
        
        # Convert the dataframe to a dictionary
        data_dict = valuation_data.to_dict()
        
        # Prepare the output dictionary
        output = {}
        for col, values in data_dict.items():
            output[col] = {k: v for k, v in values.items() if pd.notnull(v)}
        
        return output
    except Exception as e:
        print(f"Error fetching valuation history for {ticker}: {str(e)}")
        return None

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Fetch required information
        company_name = info.get('longName', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = f"{market_cap:,}"
        per = info.get('trailingPE', 'N/A')
        if per != 'N/A':
            per = f"{per:.2f}"
        pbr = info.get('priceToBook', 'N/A')
        if pbr != 'N/A':
            pbr = f"{pbr:.2f}"
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield = f"{dividend_yield*100:.2f}"
        targetMeanPrice = info.get('targetMeanPrice', 'N/A')
        targetMedianPrice = info.get('targetMedianPrice', 'N/A')       
        analystRating = info.get('recommendationKey', 'N/A')
        numberofAnalysts = info.get('numberOfAnalystOpinions', 'N/A')
        # Generate stock chart
        hist = stock.history(period="1mo")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(hist.index, hist['Close'])
        ax.set_title(f"{company_name} Stock Price - Last Month")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        
        # Save plot to a base64 string
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode()
        
        # Fetch valuation history
        valuation_history = get_valuation_history(ticker)
        
        # Clean up
        plt.close(fig)
        del fig, ax

        return {
            'company_name': company_name,
            'market_cap': market_cap,
            'per': per,
            'pbr': pbr,
            'dividend_yield': dividend_yield,
            'targetMeanPrice': targetMeanPrice,
            'targetMedianPrice': targetMedianPrice,
            'analystRating': analystRating,
            'numberofAnalysts': numberofAnalysts,
            'chart_image': f"data:image/png;base64,{chart_image}",
            'valuation_history': valuation_history
        }
    except Exception as e:
        print(f"Error fetching stock info for {ticker}: {str(e)}")
        return None

@app.route('/download_csv')
@login_required
def download_csv():
    stocks = Stock.query.filter_by(user_id=current_user.id).all()
    
    # Create a StringIO object to write CSV data
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Write the header
    cw.writerow(['Ticker', 'Company Name', 'Purchase Date', 'Purchase Price', 'Shares', 'Latest Price', 'Return Performance', 'Income Gain','Dividend Yield'])
    
    # Write the data
    for stock in stocks:
        cw.writerow([
            stock.ticker,
            stock.company_name,
            stock.purchase_date.strftime('%Y-%m-%d'),
            stock.purchase_price,
            stock.shares,
            stock.latest_price,
            stock.return_performance,
            stock.income_gain_pct,
            stock.div_yield
        ])
    
    # Create a response
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=stock_database.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output

@app.route('/edit_stock/<int:stock_id>', methods=['GET', 'POST'])
@login_required
def edit_stock(stock_id):
    stock = Stock.query.get_or_404(stock_id)
    if request.method == 'POST':
        stock.ticker = request.form['ticker']
        stock.purchase_price = float(request.form['purchase_price'])
        stock.shares = int(request.form['num_shares'])
        stock.purchase_date = datetime.strptime(request.form['purchase_date'], '%Y-%m-%d').date()
        db.session.commit()
        flash('Stock updated successfully', 'success')
        return redirect(url_for('dashboard'))
    return render_template('edit_stock.html', stock=stock)

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

    return result['choices'][0]['message']['content']

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return json.dumps({"info": info})

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

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def get_comparative_stock_chart(tickers, start_date, end_date):
    plt.figure(figsize=(10, 5))
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        if end_date == 'latest':
            end_date = datetime.now().strftime('%Y-%m-%d')
        data = stock.history(start=start_date, end=end_date)
        
        # Index the stock price to 100 at the beginning of the period
        indexed_price = data['Close'] / data['Close'].iloc[0] * 100
        plt.plot(data.index, indexed_price, label=ticker)
    
    plt.title("Comparative Stock Performance (Indexed to 100)")
    plt.xlabel("Date")
    plt.ylabel("Indexed Price")
    plt.legend()
    plt.grid(True)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

# Modify the existing /stockai route or add a new one if it doesn't exist
@app.route('/stockai', methods=['GET', 'POST'])
@login_required
def stockai():
    if request.method == 'POST':
        return process_query()
    return render_template('stockai.html')

@app.route('/process-query', methods=['POST'])
@login_required
def process_query():
    user_input = request.json['input']
    logger.debug(f"Received user input: {user_input}")
    
    try:
        info = get_info_from_gpt(user_input)
        logger.debug(f"Processed info: {info}")
        
        if info['query_type'] == 'chart':
            logger.debug("Recognized as a chart query")
            ticker = info['ticker']
            start_date = info['start_date']
            end_date = info['end_date']
            logger.debug(f"Generating chart for {ticker} from {start_date} to {end_date}")
            
            try:
                img = get_stock_chart(ticker, start_date, end_date)
                img_path = f"temp_{uuid.uuid4().hex[:8]}_chart.png"
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

@app.route('/get-chart/<path:img_path>')
@login_required
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

@cache.memoize(timeout=900)  # 15 minutes cache
def fetch_batch_stock_data(tickers):
    data = {}
    try:
        # Fetch data for all tickers in one call
        batch_data = yf.download(tickers, period="2d")
        
        for ticker in tickers:
            if len(batch_data) >= 2:
                latest_price = batch_data['Close'][ticker].iloc[-1]
                prev_price = batch_data['Close'][ticker].iloc[-2]
                daily_return = ((latest_price - prev_price) / prev_price) * 100
                data[ticker] = {
                    'latest_price': round(float(latest_price), 2),
                    'daily_return': round(float(daily_return), 2)
                }
    except Exception as e:
        print(f"Error in batch fetch: {e}")
    return data

@cache.memoize(timeout=3600)  # Cache for 1 hour
def fetch_batch_dividend_data(tickers, start_date):
    dividend_data = {}
    try:
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends
            if not dividends.empty:
                dividends.index = dividends.index.tz_localize(None)
                dividends_after_purchase = dividends[dividends.index >= start_date]
                dividend_data[ticker] = dividends_after_purchase.sum()
    except Exception as e:
        print(f"Error fetching dividend data: {e}")
    return dividend_data

@app.route('/api/stock-updates', methods=['POST'])
@login_required
def get_stock_updates():
    tickers = request.json['tickers']
    batch_data = fetch_batch_stock_data(tickers)
    return jsonify(batch_data)

if __name__ == "__main__":
    app.run(debug=True)

# Cleanup matplotlib
plt.close('all')
matplotlib.pyplot.close('all')