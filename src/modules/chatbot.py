import random
from datetime import datetime


class StockChatbot:
    """AI Stock Advisor Chatbot — answers stock questions with ML predictions."""

    def __init__(self, data_fetcher, predictor_factory):
        self._fetcher  = data_fetcher
        self._pf       = predictor_factory
        self._trained  = {}

    def _get_or_train(self, symbol):
        if symbol not in self._trained:
            df = self._fetcher.fetch_history(symbol, "1y")
            p  = self._pf(symbol)
            p.train(df)
            self._trained[symbol] = (p, df)
        return self._trained[symbol]

    def respond(self, message: str) -> str:
        msg = message.lower().strip()
        # Greeting
        if any(w in msg for w in ["hello","hi","hey","salam","assalam"]):
            return "Hello! I am your AI Stock Advisor 🤖. Ask me about any stock — predictions, analysis, buy/sell advice!"

        # Prediction
        if any(w in msg for w in ["predict","prediction","forecast","future","next week","tomorrow"]):
            symbol = self._extract_symbol(msg)
            if symbol:
                try:
                    p, df = self._get_or_train(symbol)
                    preds = p.predict_next(df, 5)
                    curr  = self._fetcher.get_current_price(symbol)
                    trend = "📈 BULLISH" if preds[-1] > curr else "📉 BEARISH"
                    return (f"🔮 **{symbol} 5-Day Prediction:**\n"
                           f"Current: ${curr}\n"
                           + "\n".join([f"Day {i+1}: ${v}" for i,v in enumerate(preds)])
                           + f"\n\n{trend} — Model: {p.get_summary()['best_model']}"
                           f" | Accuracy: {p.get_summary()['accuracy_pct']}%")
                except Exception as e:
                    return f"Sorry, could not predict {symbol}: {e}"
            return "Which stock should I predict? E.g. 'predict AAPL'"

        # Price
        if any(w in msg for w in ["price","cost","worth","value","trading at"]):
            symbol = self._extract_symbol(msg)
            if symbol:
                try:
                    d = self._fetcher.get_price_change(symbol)
                    arrow = "🟢 ▲" if d["up"] else "🔴 ▼"
                    return (f"📊 **{symbol}** — ${d['current']}\n"
                           f"{arrow} {d['change']:+.2f} ({d['pct']:+.2f}%) today")
                except Exception as e:
                    return f"Could not fetch price: {e}"
            return "Which stock price? E.g. 'price of TSLA'"

        # Buy advice
        if any(w in msg for w in ["buy","should i buy","good buy","invest"]):
            symbol = self._extract_symbol(msg)
            if symbol:
                try:
                    p, df = self._get_or_train(symbol)
                    preds = p.predict_next(df, 7)
                    curr  = self._fetcher.get_current_price(symbol)
                    upside = ((preds[-1] - curr) / curr) * 100
                    if upside > 2:
                        rec = f"✅ **BUY** — Model predicts {upside:.1f}% upside in 7 days."
                    elif upside < -2:
                        rec = f"❌ **AVOID** — Model predicts {abs(upside):.1f}% downside."
                    else:
                        rec = "⚠️ **HOLD** — Minimal movement predicted. Watch and wait."
                    return f"🤖 AI Analysis for {symbol}:\n{rec}\n\n⚠️ Disclaimer: Not financial advice."
                except Exception as e:
                    return f"Analysis failed: {e}"
            return "Which stock? E.g. 'should I buy AAPL'"

        # Market
        if any(w in msg for w in ["market","nasdaq","s&p","dow"]):
            return ("📈 **Market Overview:**\nI track individual stocks in real-time.\n"
                   "Ask me: 'predict AAPL', 'price TSLA', 'should I buy NVDA'")

        # Help
        if any(w in msg for w in ["help","what can","commands","how"]):
            return ("🤖 **AI Stock Advisor — Commands:**\n"
                   "• 'predict AAPL' — 5-day price forecast\n"
                   "• 'price of TSLA' — current price\n"
                   "• 'should I buy NVDA' — buy/sell advice\n"
                   "• 'analyze MSFT' — full analysis\n"
                   "• Any stock symbol works! (AAPL, TSLA, GOOGL...)")

        # Analyze
        if any(w in msg for w in ["analyze","analysis","tell me about","info"]):
            symbol = self._extract_symbol(msg)
            if symbol:
                try:
                    info = self._fetcher.fetch_info(symbol)
                    p, df = self._get_or_train(symbol)
                    s = p.get_summary()
                    return (f"📊 **{info.get('name',symbol)} ({symbol})**\n"
                           f"Price: ${info.get('price',0):.2f}\n"
                           f"Sector: {info.get('sector','N/A')}\n"
                           f"52W High: ${info.get('52w_high',0):.2f}\n"
                           f"52W Low:  ${info.get('52w_low',0):.2f}\n\n"
                           f"🤖 ML Model: {s['best_model']}\n"
                           f"Accuracy: {s['accuracy_pct']}%")
                except Exception as e:
                    return f"Could not analyze: {e}"

        return ("I'm not sure what you're asking. Try:\n"
               "• 'predict AAPL'\n• 'price TSLA'\n• 'should I buy NVDA'\n• 'help'")

    def _extract_symbol(self, msg: str) -> str:
        known = ["AAPL","GOOGL","MSFT","TSLA","AMZN","META","NVDA","NFLX",
                 "JPM","V","WMT","BRK-B","AMD","INTC","PYPL","UBER","LYFT"]
        for s in known:
            if s.lower() in msg:
                return s
        words = msg.upper().split()
        for w in words:
            if 1 < len(w) <= 5 and w.isalpha():
                return w
        return ""
