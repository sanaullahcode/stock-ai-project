# =============================================================
# FILE: src/models/predictor.py
# PROJECT: AI Stock Market Prediction System
# AUTHORS: Sana Ullah (42051), Hasna Ishtiaq (42013), Jaweria Shakoor
# DEPT: Software Engineering
# DESC: ML Model — Linear Regression + Random Forest for prediction
# =============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """
    AI Stock Prediction Model.
    Uses Linear Regression + Random Forest ensemble.
    Demonstrates: ML Training, Feature Engineering, Model Evaluation
    """

    def __init__(self):
        self.lr_model   = LinearRegression()
        self.rf_model   = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler     = MinMaxScaler()
        self.is_trained = False
        self.symbol     = None
        self.history    = None

    # ── Data Fetch ────────────────────────────────────────────
    def fetch_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            self.symbol  = symbol
            self.history = df
            return df
        except Exception as e:
            raise ValueError(f"Failed to fetch {symbol}: {str(e)}")

    # ── Feature Engineering ───────────────────────────────────
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators as ML features."""
        data = df.copy()

        # Moving Averages
        data['MA_5']  = data['Close'].rolling(5).mean()
        data['MA_10'] = data['Close'].rolling(10).mean()
        data['MA_20'] = data['Close'].rolling(20).mean()
        data['MA_50'] = data['Close'].rolling(50).mean()

        # Price momentum
        data['Return_1d'] = data['Close'].pct_change(1)
        data['Return_5d'] = data['Close'].pct_change(5)
        data['Return_10d'] = data['Close'].pct_change(10)

        # Volatility
        data['Volatility'] = data['Close'].rolling(10).std()

        # Volume features
        data['Volume_MA'] = data['Volume'].rolling(5).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

        # RSI
        delta = data['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['BB_Mid']   = data['Close'].rolling(20).mean()
        data['BB_Upper'] = data['BB_Mid'] + 2 * data['Close'].rolling(20).std()
        data['BB_Lower'] = data['BB_Mid'] - 2 * data['Close'].rolling(20).std()
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']

        # Target — next day close
        data['Target'] = data['Close'].shift(-1)

        return data.dropna()

    # ── Train ─────────────────────────────────────────────────
    def train(self, symbol: str) -> dict:
        """Fetch data, engineer features, train models."""
        df = self.fetch_data(symbol)
        data = self._create_features(df)

        features = ['Close','Open','High','Low','Volume',
                    'MA_5','MA_10','MA_20','MA_50',
                    'Return_1d','Return_5d','Return_10d',
                    'Volatility','Volume_Ratio','RSI',
                    'BB_Width','BB_Upper','BB_Lower']

        X = data[features].values
        y = data['Target'].values

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False)

        self.lr_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)

        # Ensemble prediction
        lr_pred = self.lr_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)
        ensemble = 0.4 * lr_pred + 0.6 * rf_pred

        self.is_trained    = True
        self._features     = features
        self._last_data    = data
        self._last_X       = X_scaled

        mae  = mean_absolute_error(y_test, ensemble)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble))
        r2   = r2_score(y_test, ensemble)

        return {
            "symbol":   symbol,
            "mae":      round(mae, 4),
            "rmse":     round(rmse, 4),
            "r2":       round(r2 * 100, 2),
            "accuracy": round(max(0, r2 * 100), 2),
            "samples":  len(data),
        }

    # ── Predict ───────────────────────────────────────────────
    def predict_next_days(self, days: int = 7) -> list:
        """Predict next N days closing prices."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        data     = self._last_data.copy()
        features = self._features
        preds    = []

        for i in range(days):
            last_row = data[features].iloc[-1].values.reshape(1, -1)
            last_scaled = self.scaler.transform(last_row)

            lr_p = self.lr_model.predict(last_scaled)[0]
            rf_p = self.rf_model.predict(last_scaled)[0]
            pred = 0.4 * lr_p + 0.6 * rf_p

            date = (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            current = data['Close'].iloc[-1]
            change  = ((pred - current) / current) * 100

            preds.append({
                "date":      date,
                "predicted": round(float(pred), 2),
                "change":    round(float(change), 2),
                "trend":     "UP 📈" if change > 0 else "DOWN 📉",
            })

            # Append synthetic row for next iteration
            new_row = data.iloc[-1].copy()
            new_row['Close'] = pred
            data = pd.concat([data, new_row.to_frame().T], ignore_index=True)
            data = self._create_features(data)

        return preds

    # ── Stock Info ────────────────────────────────────────────
    def get_stock_info(self, symbol: str) -> dict:
        """Get current stock info."""
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info
            hist   = ticker.history(period="5d")
            current = float(hist['Close'].iloc[-1]) if not hist.empty else 0
            prev    = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
            change  = current - prev
            pct     = (change / prev * 100) if prev else 0

            return {
                "symbol":      symbol.upper(),
                "name":        info.get("longName", symbol),
                "price":       round(current, 2),
                "change":      round(change, 2),
                "change_pct":  round(pct, 2),
                "volume":      info.get("volume", 0),
                "market_cap":  info.get("marketCap", 0),
                "pe_ratio":    info.get("trailingPE", 0),
                "sector":      info.get("sector", "N/A"),
                "52w_high":    info.get("fiftyTwoWeekHigh", 0),
                "52w_low":     info.get("fiftyTwoWeekLow", 0),
                "trend":       "UP 📈" if change >= 0 else "DOWN 📉",
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    # ── Historical Chart Data ─────────────────────────────────
    def get_chart_data(self, symbol: str, period: str = "6mo") -> dict:
        """Get OHLCV data for charts."""
        try:
            df = yf.Ticker(symbol).history(period=period)
            return {
                "dates":   df.index.strftime("%Y-%m-%d").tolist(),
                "open":    [round(float(x), 2) for x in df['Open']],
                "high":    [round(float(x), 2) for x in df['High']],
                "low":     [round(float(x), 2) for x in df['Low']],
                "close":   [round(float(x), 2) for x in df['Close']],
                "volume":  [int(x) for x in df['Volume']],
                "ma20":    [round(float(x), 2) if not np.isnan(x) else None
                            for x in df['Close'].rolling(20).mean()],
                "ma50":    [round(float(x), 2) if not np.isnan(x) else None
                            for x in df['Close'].rolling(50).mean()],
            }
        except Exception as e:
            return {"error": str(e)}
