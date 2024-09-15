import pandas as pd
import pandas_ta as ta

def calculate_indicators(data):
    df = pd.DataFrame(data)
    
    # Calculate indicators (e.g. RSI, MACD, SMA)
    df.ta.rsi(close='Close', length=14, append=True)
    df.ta.sma(close='Close', length=50, append=True)
    df.ta.macd(close='Close', append=True)
    
    return df[['RSI_14', 'SMA_50', 'MACD_12_26_9']].to_dict()
