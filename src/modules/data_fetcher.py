import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    """Fetches real-time stock data from Yahoo Finance."""

    POPULAR_STOCKS = {
        "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.", "MSFT": "Microsoft Corp.",
        "TSLA": "Tesla Inc.", "AMZN": "Amazon.com Inc.", "META": "Meta Platforms",
        "NVDA": "NVIDIA Corp.", "NFLX": "Netflix Inc.", "JPM": "JPMorgan Chase",
        "BRK-B": "Berkshire Hathaway", "V": "Visa Inc.", "WMT": "Walmart Inc.",
    }

    def __init__(self):
        self._cache = {}

    def fetch_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        key = f"{symbol}_{period}"
        if key in self._cache:
            return self._cache[key]
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for symbol '{symbol}'.")
        df.index = pd.to_datetime(df.index)
        self._cache[key] = df
        return df

    def fetch_info(self, symbol: str) -> dict:
        try:
            t = yf.Ticker(symbol)
            info = t.info
            return {
                "name":        info.get("longName", symbol),
                "symbol":      symbol,
                "price":       info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "market_cap":  info.get("marketCap", 0),
                "pe_ratio":    info.get("trailingPE", 0),
                "52w_high":    info.get("fiftyTwoWeekHigh", 0),
                "52w_low":     info.get("fiftyTwoWeekLow", 0),
                "volume":      info.get("volume", 0),
                "sector":      info.get("sector", "N/A"),
                "description": info.get("longBusinessSummary", "")[:300],
            }
        except Exception:
            return {"name": symbol, "symbol": symbol, "price": 0}

    def get_current_price(self, symbol: str) -> float:
        try:
            df = self.fetch_history(symbol, "5d")
            return round(float(df['Close'].iloc[-1]), 2)
        except Exception:
            return 0.0

    def get_price_change(self, symbol: str) -> dict:
        try:
            df = self.fetch_history(symbol, "5d")
            prev  = float(df['Close'].iloc[-2])
            curr  = float(df['Close'].iloc[-1])
            chg   = curr - prev
            pct   = (chg / prev) * 100
            return {"current": round(curr,2), "change": round(chg,2),
                    "pct": round(pct,2), "up": chg >= 0}
        except Exception:
            return {"current":0,"change":0,"pct":0,"up":True}

    def get_multiple_prices(self, symbols: list) -> dict:
        result = {}
        for s in symbols:
            result[s] = self.get_price_change(s)
        return result

    def get_popular_stocks(self): return self.POPULAR_STOCKS
