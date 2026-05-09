# =============================================================
# FILE: src/models/chatbot.py
# AUTHORS: Hasna Ishtiaq (42013), Jaweria Shakoor
# DESC: AI Stock Chatbot — NLP-based query handler
# =============================================================

import re
from datetime import datetime


class StockChatbot:
    """
    AI-powered Stock Market Chatbot.
    Handles natural language queries about stocks.
    """

    def __init__(self, predictor):
        self.predictor  = predictor
        self.trained    = {}   # {symbol: metrics}
        self.context    = {}   # conversation context
        self._greetings = ["hello","hi","hey","salam","assalam"]
        self._farewells = ["bye","goodbye","exit","quit","alvida"]

    # ── Intent Detection ──────────────────────────────────────
    def _detect_intent(self, msg: str) -> tuple:
        msg_lower = msg.lower()
        symbol    = self._extract_symbol(msg)

        if any(g in msg_lower for g in self._greetings):
            return "greet", None
        if any(f in msg_lower for f in self._farewells):
            return "farewell", None
        if any(w in msg_lower for w in ["predict","forecast","future","tomorrow","next week","will"]):
            return "predict", symbol
        if any(w in msg_lower for w in ["price","cost","worth","trading","current","now"]):
            return "price", symbol
        if any(w in msg_lower for w in ["train","learn","analyze","study","model"]):
            return "train", symbol
        if any(w in msg_lower for w in ["buy","sell","should i","recommend","advice","suggest"]):
            return "advice", symbol
        if any(w in msg_lower for w in ["news","update","latest","happening"]):
            return "news", symbol
        if any(w in msg_lower for w in ["help","what can","how to","guide"]):
            return "help", None
        if any(w in msg_lower for w in ["portfolio","my stocks","holdings"]):
            return "portfolio", None
        if any(w in msg_lower for w in ["market","overall","index","sp500","nasdaq","dow"]):
            return "market", symbol
        if symbol:
            return "stock_info", symbol
        return "unknown", None

    def _extract_symbol(self, msg: str) -> str:
        """Extract stock ticker from message."""
        # Common stock names mapping
        name_map = {
            "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL",
            "microsoft": "MSFT", "amazon": "AMZN", "tesla": "TSLA",
            "meta": "META", "facebook": "META", "nvidia": "NVDA",
            "netflix": "NFLX", "twitter": "X", "samsung": "005930.KS",
            "bitcoin": "BTC-USD", "ethereum": "ETH-USD",
            "sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI",
            "gold": "GC=F", "oil": "CL=F",
        }
        msg_lower = msg.lower()
        for name, sym in name_map.items():
            if name in msg_lower:
                return sym

        # Detect uppercase ticker (2-5 letters)
        tickers = re.findall(r'\b[A-Z]{2,5}\b', msg)
        if tickers:
            common = {"I","A","AN","THE","IS","ARE","FOR","AND","OR","IN","ON","AT","TO"}
            filtered = [t for t in tickers if t not in common]
            if filtered:
                return filtered[0]
        return None

    # ── Response Generator ────────────────────────────────────
    def respond(self, message: str) -> dict:
        """Main chatbot response function."""
        intent, symbol = self._detect_intent(message)
        response = ""
        data     = {}
        rtype    = "text"

        try:
            if intent == "greet":
                response = (
                    "👋 Hello! I'm **StockAI Bot** — your intelligent stock market assistant!\n\n"
                    "I can help you with:\n"
                    "• 📈 **Predict** stock prices using AI/ML\n"
                    "• 💰 **Current prices** of any stock\n"
                    "• 🧠 **Train** ML model on any stock\n"
                    "• 💡 **Buy/Sell recommendations**\n"
                    "• 📰 **Market insights**\n\n"
                    "Try asking: *'Predict AAPL for next 7 days'* or *'What is Tesla price?'*"
                )

            elif intent == "farewell":
                response = "👋 Goodbye! Happy investing! Remember: always do your own research! 📊"

            elif intent == "help":
                response = (
                    "🤖 **StockAI Bot — Help Guide**\n\n"
                    "**Commands you can use:**\n"
                    "• `Predict AAPL` → AI prediction for Apple\n"
                    "• `Price of TSLA` → Current Tesla price\n"
                    "• `Train model on MSFT` → Train ML model\n"
                    "• `Should I buy NVDA?` → AI recommendation\n"
                    "• `GOOGL news` → Latest updates\n"
                    "• `Market overview` → Index summary\n\n"
                    "**Supported:** AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA, META, NFLX + more!"
                )

            elif intent == "price":
                if not symbol:
                    response = "❓ Which stock? Try: *'Price of AAPL'* or *'TSLA price'*"
                else:
                    info = self.predictor.get_stock_info(symbol)
                    if "error" in info:
                        response = f"❌ Could not fetch {symbol}. Try a valid ticker like AAPL, TSLA."
                    else:
                        trend_emoji = "🟢" if info['change'] >= 0 else "🔴"
                        response = (
                            f"{trend_emoji} **{info['name']} ({info['symbol']})**\n\n"
                            f"💵 Current Price: **${info['price']}**\n"
                            f"📊 Change: {'+' if info['change']>=0 else ''}{info['change']} "
                            f"({'+' if info['change_pct']>=0 else ''}{info['change_pct']}%)\n"
                            f"📦 Sector: {info['sector']}\n"
                            f"📈 52W High: ${info['52w_high']}\n"
                            f"📉 52W Low: ${info['52w_low']}\n"
                            f"Trend: {info['trend']}"
                        )
                        data  = info
                        rtype = "stock_info"

            elif intent == "train":
                if not symbol:
                    response = "❓ Which stock to train on? Try: *'Train model on AAPL'*"
                else:
                    response = f"🧠 Training AI model on **{symbol}**... Please wait!"
                    rtype    = "train_trigger"
                    data     = {"symbol": symbol}

            elif intent == "predict":
                if not symbol:
                    response = "❓ Which stock to predict? Try: *'Predict TSLA for next 7 days'*"
                else:
                    days_match = re.search(r'(\d+)\s*day', message.lower())
                    days = int(days_match.group(1)) if days_match else 7
                    days = min(days, 30)
                    rtype    = "predict_trigger"
                    data     = {"symbol": symbol, "days": days}
                    response = f"🔮 Generating AI prediction for **{symbol}** — next {days} days..."

            elif intent == "advice":
                if not symbol:
                    response = "❓ Which stock? Try: *'Should I buy AAPL?'*"
                else:
                    info = self.predictor.get_stock_info(symbol)
                    if "error" not in info:
                        pct = info.get('change_pct', 0)
                        pe  = info.get('pe_ratio', 0)
                        if pct > 1.5:
                            rec = "📈 **BULLISH** — Stock is trending UP today. Momentum is positive."
                            action = "Consider **holding or buying** if fundamentals are strong."
                        elif pct < -1.5:
                            rec = "📉 **BEARISH** — Stock is trending DOWN today. Caution advised."
                            action = "Consider **waiting** for a better entry point."
                        else:
                            rec = "➡️ **NEUTRAL** — Stock is relatively flat today."
                            action = "Consider **researching further** before making a decision."
                        response = (
                            f"💡 **AI Recommendation for {symbol}**\n\n"
                            f"{rec}\n"
                            f"{action}\n\n"
                            f"📊 Today's Change: {pct}%\n"
                            f"⚠️ *Disclaimer: This is AI-generated analysis, not financial advice.*"
                        )
                    else:
                        response = f"❌ Could not analyze {symbol}."

            elif intent == "market":
                response = (
                    "📊 **Market Overview**\n\n"
                    "Fetching live index data... Use the **Market** tab for detailed charts!\n\n"
                    "**Major Indices:**\n"
                    "• S&P 500 (^GSPC)\n• NASDAQ (^IXIC)\n• Dow Jones (^DJI)\n\n"
                    "Try: *'Price of ^GSPC'* for live S&P 500 data!"
                )
                rtype = "market_trigger"

            elif intent == "stock_info":
                info = self.predictor.get_stock_info(symbol)
                if "error" in info:
                    response = f"❌ Could not find **{symbol}**. Check the ticker symbol."
                else:
                    response = (
                        f"📈 **{info['name']}** ({symbol})\n"
                        f"Price: ${info['price']} | Change: {info['change_pct']}%\n"
                        f"Trend: {info['trend']}"
                    )
                    data  = info
                    rtype = "stock_info"

            else:
                response = (
                    "🤔 I didn't quite understand that. Try:\n"
                    "• *'Predict AAPL'*\n• *'Price of TSLA'*\n"
                    "• *'Should I buy MSFT?'*\n• *'Help'*"
                )

        except Exception as e:
            response = f"⚠️ Something went wrong: {str(e)}. Please try again!"

        return {
            "response": response,
            "type":     rtype,
            "data":     data,
            "time":     datetime.now().strftime("%H:%M"),
        }
