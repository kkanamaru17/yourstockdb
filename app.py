from flask import Flask, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from config import Config
import yfinance as yf

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SECRET_KEY'] = 'thisisasecretkey'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper functions
def fetch_latest_price(ticker):
    stock = yf.Ticker(ticker)
    latest_price = stock.history(period="1d")['Close'].iloc[-1]
    return latest_price

def fetch_forwardPE(ticker):
    stock = yf.Ticker(ticker)
    quote_table = stock.info
    forward_pe = quote_table.get('forwardPE')
    return forward_pe

def fetch_divyiled(ticker):
    stock = yf.Ticker(ticker)
    quote_table = stock.info
    div_yield = quote_table.get('dividendYield')
    # Check if div_yield is None, and return "-" if it is
    if div_yield is None:
        return "-"
    # If div_yield is a valid number, multiply by 100 to get the percentage
    return div_yield * 100

def calculate_returns(purchase_price, latest_price):
    return ((latest_price - purchase_price) / purchase_price) * 100

def calculate_portfolio_return(stocks_data):
    total_investment = sum(stock.purchase_price * stock.shares for stock in stocks_data)
    total_current_value = sum(stock.latest_price * stock.shares for stock in stocks_data)
    portfolio_return = ((total_current_value - total_investment) / total_investment) * 100
    return portfolio_return


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    latest_price = db.Column(db.Float, nullable=True)
    return_performance = db.Column(db.Float, nullable=True, default=0.0)
    forward_pe = db.Column(db.Float, nullable=True, default=0.0)
    div_yield = db.Column(db.Float, nullable=True, default=0.0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

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


# @app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
# def dashboard():
#     return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
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
        
        latest_price = fetch_latest_price(ticker)
        return_performance = calculate_returns(purchase_price, latest_price)
        forward_pe = fetch_forwardPE(ticker)
        div_yield = fetch_divyiled(ticker)
        
        stock = Stock.query.filter_by(ticker=ticker, user_id=current_user.id).first()
        if stock:
            stock.purchase_price = purchase_price
            stock.shares = shares
            stock.latest_price = latest_price
            stock.return_performance = return_performance
            stock.forward_pe = forward_pe
            stock.div_yield = div_yield
        else:
            new_stock = Stock(
                ticker=ticker,
                purchase_price=purchase_price,
                shares=shares,
                latest_price=latest_price,
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
        stock.return_performance = calculate_returns(stock.purchase_price, stock.latest_price)
    
    db.session.commit()

    portfolio_return = calculate_portfolio_return(stock_data) if stock_data else 0
    return render_template('dashboard.html', stocks=stock_data, portfolio_return=portfolio_return)

@app.route('/delete', methods=['POST'])
@login_required
def delete():
    ticker = request.form['ticker']
    stock = Stock.query.filter_by(ticker=ticker, user_id=current_user.id).first()
    if stock:
        db.session.delete(stock)
        db.session.commit()
    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    app.run(debug=True)