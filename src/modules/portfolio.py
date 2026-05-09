from datetime import datetime


class Portfolio:
    """User portfolio tracker — buy/sell stocks, track P&L."""

    def __init__(self, user_name: str, initial_balance: float = 100000.0):
        self.user_name = user_name
        self._cash     = initial_balance
        self._holdings = {}   # {symbol: {qty, avg_price}}
        self._history  = []

    def buy(self, symbol: str, qty: int, price: float) -> dict:
        cost = qty * price
        if cost > self._cash:
            raise ValueError(f"Insufficient cash. Need ${cost:.2f}, have ${self._cash:.2f}")
        self._cash -= cost
        if symbol in self._holdings:
            h = self._holdings[symbol]
            total_qty   = h["qty"] + qty
            h["avg_price"] = (h["qty"]*h["avg_price"] + qty*price) / total_qty
            h["qty"]    = total_qty
        else:
            self._holdings[symbol] = {"qty": qty, "avg_price": price}
        rec = {"action":"BUY","symbol":symbol,"qty":qty,"price":price,
               "total":cost,"date":datetime.now().strftime("%Y-%m-%d %H:%M")}
        self._history.append(rec)
        return rec

    def sell(self, symbol: str, qty: int, price: float) -> dict:
        if symbol not in self._holdings or self._holdings[symbol]["qty"] < qty:
            raise ValueError(f"Not enough shares of {symbol} to sell.")
        revenue = qty * price
        avg     = self._holdings[symbol]["avg_price"]
        pl      = (price - avg) * qty
        self._cash += revenue
        self._holdings[symbol]["qty"] -= qty
        if self._holdings[symbol]["qty"] == 0:
            del self._holdings[symbol]
        rec = {"action":"SELL","symbol":symbol,"qty":qty,"price":price,
               "total":revenue,"pl":round(pl,2),
               "date":datetime.now().strftime("%Y-%m-%d %H:%M")}
        self._history.append(rec)
        return rec

    def get_value(self, current_prices: dict) -> dict:
        holdings_val = 0
        positions = []
        for sym, h in self._holdings.items():
            curr  = current_prices.get(sym, h["avg_price"])
            val   = curr * h["qty"]
            pl    = (curr - h["avg_price"]) * h["qty"]
            pl_pct= ((curr - h["avg_price"]) / h["avg_price"]) * 100
            holdings_val += val
            positions.append({"symbol":sym,"qty":h["qty"],"avg":round(h["avg_price"],2),
                              "current":round(curr,2),"value":round(val,2),
                              "pl":round(pl,2),"pl_pct":round(pl_pct,2)})
        total = self._cash + holdings_val
        return {"cash":round(self._cash,2),"holdings_value":round(holdings_val,2),
                "total":round(total,2),"positions":positions,"history":self._history}

    def get_cash(self): return round(self._cash, 2)
    def get_holdings(self): return dict(self._holdings)
    def get_history(self): return list(self._history)
