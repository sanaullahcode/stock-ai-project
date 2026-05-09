# =============================================================
# FILE: src/models/portfolio.py
# AUTHOR: Jaweria Shakoor
# DESC: Portfolio Tracker — add stocks, track P&L
# =============================================================

from datetime import datetime


class Portfolio:
    """Personal stock portfolio tracker."""

    def __init__(self):
        self._holdings = {}   # {symbol: {shares, avg_price, added}}

    def add_stock(self, symbol: str, shares: float, buy_price: float) -> dict:
        sym = symbol.upper()
        if sym in self._holdings:
            old = self._holdings[sym]
            total_shares = old['shares'] + shares
            avg = ((old['shares'] * old['avg_price']) + (shares * buy_price)) / total_shares
            self._holdings[sym] = {'shares': total_shares, 'avg_price': round(avg,2),
                                   'added': old['added']}
        else:
            self._holdings[sym] = {'shares': shares, 'avg_price': buy_price,
                                   'added': datetime.now().strftime("%Y-%m-%d")}
        return self._holdings[sym]

    def remove_stock(self, symbol: str):
        sym = symbol.upper()
        if sym not in self._holdings:
            raise KeyError(f"{sym} not in portfolio.")
        del self._holdings[sym]

    def get_holdings(self) -> dict:
        return dict(self._holdings)

    def calculate_pnl(self, current_prices: dict) -> list:
        result = []
        for sym, data in self._holdings.items():
            current = current_prices.get(sym, data['avg_price'])
            invested = data['shares'] * data['avg_price']
            current_val = data['shares'] * current
            pnl = current_val - invested
            pnl_pct = (pnl / invested) * 100 if invested else 0
            result.append({
                'symbol':      sym,
                'shares':      data['shares'],
                'avg_price':   data['avg_price'],
                'current':     round(current, 2),
                'invested':    round(invested, 2),
                'current_val': round(current_val, 2),
                'pnl':         round(pnl, 2),
                'pnl_pct':     round(pnl_pct, 2),
                'status':      'PROFIT 🟢' if pnl >= 0 else 'LOSS 🔴',
            })
        return result

    def get_total_value(self, current_prices: dict) -> dict:
        items = self.calculate_pnl(current_prices)
        total_inv = sum(i['invested'] for i in items)
        total_cur = sum(i['current_val'] for i in items)
        total_pnl = total_cur - total_inv
        return {
            'total_invested': round(total_inv, 2),
            'total_current':  round(total_cur, 2),
            'total_pnl':      round(total_pnl, 2),
            'total_pnl_pct':  round((total_pnl/total_inv*100) if total_inv else 0, 2),
            'holdings_count': len(self._holdings),
        }
