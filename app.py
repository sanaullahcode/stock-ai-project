# =============================================================
# FILE: app.py
# PROJECT: AI Stock Market Prediction System
# AUTHORS: Sana Ullah (42051), Hasna Ishtiaq (42013), Jaweria Shakoor
# DEPT: Software Engineering
# DESC: Flask Backend — REST API
# =============================================================

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import traceback

from src.models.predictor import StockPredictor
from src.models.chatbot   import StockChatbot
from src.models.portfolio import Portfolio

app       = Flask(__name__)
CORS(app)

predictor = StockPredictor()
chatbot   = StockChatbot(predictor)
portfolio = Portfolio()

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<symbol>')
def get_stock(symbol):
    try:
        info = predictor.get_stock_info(symbol.upper())
        return jsonify({"success": True, "data": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/chart/<symbol>')
def get_chart(symbol):
    try:
        period = request.args.get('period', '6mo')
        data   = predictor.get_chart_data(symbol.upper(), period)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/train/<symbol>', methods=['POST'])
def train_model(symbol):
    try:
        metrics = predictor.train(symbol.upper())
        return jsonify({"success": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/predict/<symbol>')
def predict(symbol):
    try:
        days = int(request.args.get('days', 7))
        if not predictor.is_trained or predictor.symbol != symbol.upper():
            predictor.train(symbol.upper())
        predictions = predictor.predict_next_days(days)
        return jsonify({"success": True, "predictions": predictions, "symbol": symbol.upper()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data    = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({"success": False, "error": "Empty message"}), 400
        result = chatbot.respond(message)
        # Handle train/predict triggers
        if result['type'] == 'train_trigger':
            sym     = result['data']['symbol']
            metrics = predictor.train(sym)
            result['metrics'] = metrics
            result['response'] = (
                f"✅ **Model trained on {sym}!**\n\n"
                f"📊 Accuracy: **{metrics['r2']}%**\n"
                f"📉 MAE: {metrics['mae']}\n"
                f"📐 RMSE: {metrics['rmse']}\n"
                f"📅 Training samples: {metrics['samples']}\n\n"
                f"Now ask me to *predict {sym}* for future prices!"
            )
        elif result['type'] == 'predict_trigger':
            sym  = result['data']['symbol']
            days = result['data']['days']
            if not predictor.is_trained or predictor.symbol != sym:
                predictor.train(sym)
            preds = predictor.predict_next_days(days)
            result['predictions'] = preds
            lines = "\n".join(
                [f"• {p['date']}: **${p['predicted']}** ({'+' if p['change']>=0 else ''}{p['change']}%) {p['trend']}"
                 for p in preds])
            result['response'] = (
                f"🔮 **AI Prediction for {sym} — Next {days} Days:**\n\n{lines}\n\n"
                f"⚠️ *ML predictions — not financial advice.*"
            )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        holdings = portfolio.get_holdings()
        prices   = {}
        for sym in holdings:
            try:
                info = predictor.get_stock_info(sym)
                prices[sym] = info.get('price', 0)
            except:
                prices[sym] = holdings[sym]['avg_price']
        pnl   = portfolio.calculate_pnl(prices)
        total = portfolio.get_total_value(prices)
        return jsonify({"success": True, "holdings": pnl, "summary": total})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/portfolio/add', methods=['POST'])
def add_to_portfolio():
    try:
        data      = request.get_json()
        symbol    = data['symbol'].upper()
        shares    = float(data['shares'])
        buy_price = float(data['buy_price'])
        result    = portfolio.add_stock(symbol, shares, buy_price)
        return jsonify({"success": True, "holding": result, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/portfolio/remove/<symbol>', methods=['DELETE'])
def remove_from_portfolio(symbol):
    try:
        portfolio.remove_stock(symbol.upper())
        return jsonify({"success": True, "message": f"{symbol} removed."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/watchlist')
def get_watchlist():
    symbols = ['AAPL','TSLA','MSFT','GOOGL','AMZN','NVDA','META','NFLX']
    result  = []
    for sym in symbols:
        try:
            info = predictor.get_stock_info(sym)
            result.append(info)
        except:
            pass
    return jsonify({"success": True, "watchlist": result})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AI STOCK MARKET PREDICTION SYSTEM")
    print("  Sana Ullah (42051) | Hasna Ishtiaq (42013) | Jaweria Shakoor")
    print("  Dept: Software Engineering")
    print("="*60)
    print("  Open browser: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
