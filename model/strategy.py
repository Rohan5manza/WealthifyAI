def calculate_factor_scores(df):
    """
    Assigns factor scores based on PE Ratio (Value), Momentum, and Volatility.
    A positive score indicates a buy signal, and a negative score indicates a sell signal.
    """

    # Score based on Value Factor (lower P/E is better)
    df['Value_Score'] = -1 * df['PE_Ratio']

    # Score based on Momentum (higher momentum is better)
    df['Momentum_Score'] = df['Momentum']

    # Score based on Volatility (lower volatility is better)
    df['Volatility_Score'] = -1 * df['Volatility']

    # Combine factor scores (you can assign weights to each factor)
    df['Factor_Score'] = 0.4 * df['Value_Score'] + 0.3 * df['Momentum_Score'] + 0.3 * df['Volatility_Score']

    # Generate trading signals based on the combined factor score
    df['Signal'] = 0
    df.loc[df['Factor_Score'] > 0, 'Signal'] = 1   # Buy signal if score is positive
    df.loc[df['Factor_Score'] < 0, 'Signal'] = -1  # Sell signal if score is negative

    return df
