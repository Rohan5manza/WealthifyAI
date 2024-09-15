from pymongo import MongoClient

# Setup MongoDB client
client = MongoClient("mongodb+srv://rohanmarar5manza:sherlockholmes@cluster0.hby3g.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['trading_db']
portfolio_collection = db['portfolios']

# Get user portfolio
def get_user_portfolio(user_id: str):
    return portfolio_collection.find_one({"user_id": user_id})

# Save user portfolio
def save_user_portfolio(user_id: str, ticker: str, quantity: int):
    portfolio = get_user_portfolio(user_id)
    
    if portfolio:
        # Update existing portfolio
        portfolio['stocks'][ticker] = portfolio['stocks'].get(ticker, 0) + quantity
        portfolio_collection.update_one({'user_id': user_id}, {'$set': {'stocks': portfolio['stocks']}})
    else:
        # Create new portfolio
        portfolio = {"user_id": user_id, "stocks": {ticker: quantity}}
        portfolio_collection.insert_one(portfolio)
    
    return portfolio


