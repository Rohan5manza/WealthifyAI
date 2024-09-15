from fastapi import FastAPI, Query
from pydantic import BaseModel
from backend.stock_data import get_stock_data
from backend.strategy import calculate_factor_scores
from stock_data import get_stock_data, get_candlestick_data, create_candlestick_chart

from fastapi import FastAPI
from pydantic import BaseModel
from stock_data import get_stock_data, calculate_factor_scores

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str
    period: str = "1y"

@app.post("/stock_data/")
def stock_data(request: StockRequest):
    df = get_stock_data(request.ticker, request.period)
    result = df[['Close']].reset_index().to_dict('records')
    return result

@app.post("/factor_investing/")
def factor_investing(request: StockRequest):
    df = get_stock_data(request.ticker, request.period)
    df = calculate_factor_scores(df)
    result = df[['Close', 'Factor_Score', 'Decision']].reset_index().to_dict('records')
    return result
