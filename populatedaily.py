# from app import db, Stock, fetch_daily_return

# def populate_daily_return():
#     # Query all existing stocks
#     stocks = Stock.query.all()
#     # Iterate through each stock
#     for stock in stocks:
#         # Fetch the daily return value
#         daily_return = fetch_daily_return(stock.ticker)
#         # Update the daily_return column
#         stock.daily_return = daily_return
#     # Commit the changes to the database
#     db.session.commit()

# if __name__ == "__main__":
#     populate_daily_return()
