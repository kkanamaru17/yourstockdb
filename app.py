from flask import Flask, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from config import Config
import yfinance as yf
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import matplotlib
matplotlib.use('Agg')
import io
import base64
import pandas as pd
from yahoo_fin import stock_info as si

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper functions
# def fetch_latest_price(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         latest_price = stock.history(period="1d")['Close'].iloc[-1]
#         return float(latest_price) if latest_price is not None else 0.0  # Check for None
#     except Exception:
#         return 0


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

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(128), nullable=False)

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    latest_price = db.Column(db.Float, nullable=True)
    daily_return = db.Column(db.Float, nullable=True, default=0.0)
    return_performance = db.Column(db.Float, nullable=True, default=0.0)
    forward_pe = db.Column(db.Float, nullable=True, default=0.0)
    div_yield = db.Column(db.Float, nullable=True, default=0.0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class StockMemo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    memo = db.Column(db.Text)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        ticker = request.form['ticker']
        purchase_price = float(request.form['purchase_price'])
        shares = int(request.form['num_shares'])
        
        latest_price = fetch_latest_price(ticker)
        return_performance = calculate_returns(purchase_price, latest_price)
        forward_pe = fetch_forwardPE(ticker)
        div_yield = fetch_divyiled(ticker)
        daily_return = fetch_daily_return(ticker)

        stock = Stock.query.filter_by(ticker=ticker, user_id=current_user.id).first()
        if stock:
            # Aggregate shares and recalculate the average purchase price
            total_investment = (stock.purchase_price * stock.shares) + (purchase_price * shares)
            total_shares = stock.shares + shares
            new_average_purchase_price = total_investment / total_shares

            # Update stock details
            stock.purchase_price = new_average_purchase_price
            stock.shares = total_shares
            stock.latest_price = latest_price
            stock.daily_return = daily_return
            stock.return_performance = calculate_returns(new_average_purchase_price, latest_price)
            stock.forward_pe = forward_pe
            stock.div_yield = div_yield
        else:
            # Create a new stock entry
            new_stock = Stock(
                ticker=ticker,
                purchase_price=purchase_price,
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

    # Update latest prices and recalculate return for each stock
    for stock in stock_data:
        stock.latest_price = fetch_latest_price(stock.ticker)
        stock.daily_return = fetch_daily_return(stock.ticker)
        stock.return_performance = calculate_returns(stock.purchase_price, stock.latest_price)
    
    db.session.commit()

    portfolio_return = calculate_portfolio_return(stock_data) if stock_data else 0
    portfolio_return_withdiv = calculate_portfolio_return_withdiv(stock_data) if stock_data else 0

    # Calculate new metrics
    total_value = sum(stock.latest_price * stock.shares for stock in stock_data)
    total_cost = sum(stock.purchase_price * stock.shares for stock in stock_data)
    return_value = total_value - total_cost
    dividend_value = sum((stock.div_yield / 100) * stock.latest_price * stock.shares for stock in stock_data)
    num_stocks = len(stock_data)
    winning_stocks = sum(1 for stock in stock_data if stock.return_performance > 0)
    win_rate = (winning_stocks / num_stocks) * 100 if num_stocks > 0 else 0

    return render_template('dashboard.html', 
                           stocks=stock_data, 
                           portfolio_return=portfolio_return, 
                           portfolio_return_withdiv=portfolio_return_withdiv,
                           total_value=total_value,
                           return_value=return_value,
                           dividend_value=dividend_value,
                           num_stocks=num_stocks,
                           win_rate=win_rate)

@app.route('/delete', methods=['POST'])
def delete():
    ticker = request.form['ticker']
    stock = Stock.query.filter_by(ticker=ticker, user_id=current_user.id).first()
    if stock:
        db.session.delete(stock)
        db.session.commit()
    return redirect(url_for('dashboard'))

# @app.route('/stockan', methods=['GET', 'POST'])
# def stockan():
#     if request.method == 'POST':
#         ticker = request.form['ticker']
#         stock_info = get_stock_info(ticker)
#         return render_template('stockan.html', stock_info=stock_info)
#     return render_template('stockan.html')

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


if __name__ == "__main__":
    app.run(debug=True)

# Cleanup matplotlib
plt.close('all')
matplotlib.pyplot.close('all')