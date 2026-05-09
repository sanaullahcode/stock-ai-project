import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """Multi-model AI Stock Predictor — Linear Regression, Random Forest, Gradient Boosting."""

    def __init__(self, symbol: str):
        self.symbol      = symbol.upper()
        self._scaler     = MinMaxScaler()
        self._models     = {
            "Linear Regression":    LinearRegression(),
            "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        self._best_model = None
        self._best_name  = None
        self._metrics    = {}
        self._is_trained = False
        self._feature_cols = ['MA_5','MA_10','MA_20','RSI','Volatility',
                              'Price_Change','Volume_MA','High_Low_Diff',
                              'Open_Close','Lag_1','Lag_2','Lag_3']

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-9)))

    def _engineer_features(self, df):
        df = df.copy()
        df['MA_5']       = df['Close'].rolling(5).mean()
        df['MA_10']      = df['Close'].rolling(10).mean()
        df['MA_20']      = df['Close'].rolling(20).mean()
        df['RSI']        = self._compute_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(10).std()
        df['Price_Change']  = df['Close'].pct_change()
        df['Volume_MA']     = df['Volume'].rolling(5).mean()
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['Open_Close']    = df['Close'] - df['Open']
        df['Lag_1'] = df['Close'].shift(1)
        df['Lag_2'] = df['Close'].shift(2)
        df['Lag_3'] = df['Close'].shift(3)
        df.dropna(inplace=True)
        return df

    def train(self, df):
        df_feat = self._engineer_features(df)
        X = df_feat[self._feature_cols].values
        y = df_feat['Close'].values
        X_sc = self._scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
        best_r2 = -999
        for name, model in self._models.items():
            model.fit(X_tr, y_tr)
            p  = model.predict(X_te)
            r2 = r2_score(y_te, p)
            self._metrics[name] = {
                "r2":   round(r2, 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_te, p))), 4),
                "mae":  round(float(mean_absolute_error(y_te, p)), 4),
            }
            if r2 > best_r2:
                best_r2 = r2
                self._best_model = model
                self._best_name  = name
        self._is_trained = True
        return self._metrics

    def predict_next(self, df, days=7):
        if not self._is_trained:
            raise RuntimeError("Train model first.")
        df_feat  = self._engineer_features(df)
        last_row = df_feat[self._feature_cols].iloc[-1].values.copy().astype(float)
        preds = []
        for _ in range(days):
            sc   = self._scaler.transform([last_row])
            pred = float(self._best_model.predict(sc)[0])
            preds.append(round(pred, 2))
            last_row = np.roll(last_row, -1)
            last_row[-1] = pred
        return preds

    def get_summary(self):
        b = self._metrics.get(self._best_name, {})
        return {
            "symbol": self.symbol, "best_model": self._best_name,
            "r2": b.get("r2",0), "accuracy_pct": round(max(0,b.get("r2",0))*100,2),
            "rmse": b.get("rmse",0), "mae": b.get("mae",0),
            "all_models": self._metrics
        }

    def is_trained(self): return self._is_trained
