# from app import app, db, Stock
# import yfinance as yf

# def update_company_names():
#     with app.app_context():
#         stocks = Stock.query.all()
#         for stock in stocks:
#             if stock.company_name is None or stock.company_name == 'None':
#                 try:
#                     ticker = yf.Ticker(stock.ticker)
#                     company_name = ticker.info.get('shortName', 'N/A')
#                     stock.company_name = company_name
#                     print(f"Updated {stock.ticker} with company name: {company_name}")
#                 except Exception as e:
#                     print(f"Error updating {stock.ticker}: {str(e)}")
        
#         db.session.commit()
#         print("Company names update completed.")

# if __name__ == "__main__":
#     update_company_names()