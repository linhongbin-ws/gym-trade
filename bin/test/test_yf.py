import yfinance as yf


df = yf.download(['MSFT', 'AAPL', 'GOOG'], period='1mo')
print(df)