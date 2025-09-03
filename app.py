#!/usr/bin/env python3
"""
Intraday ML Trader Orchestrator (verbose)
- Uses settings.py, auth.py, storage.py
- WebSocket + REST fallback with robust parsing (aliases + nested fields)
- Online SGDRegressor forecaster
- Paper trader with TP/SL
- Backfill CLI
- Clear heartbeats & basic token auto-refresh
"""
import os
import sys
import time
import signal
import argparse
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from settings import (
    APP_ID, LOG_PATH, WS_LITE_MODE, SYMBOLS,
    PREDICT_HORIZON_SECS, MIN_TRAIN_SNAPSHOTS,
    ENTRY_THRESHOLD_PCT, TAKE_PROFIT_PCT, STOP_LOSS_PCT,
    POSITION_QTY, ALLOW_SHORT, DRY_RUN, MAX_OPEN_POSITIONS,
    QUOTE_POLL_SECS, MARKET_TZ_OFFSET_MIN, MARKET_OPEN, MARKET_CLOSE,
    SESSION, DATA_BASE, DB_PATH,
)
from auth import ensure_tokens, access_token_valid, refresh_access_token, load_tokens, save_tokens
from storage import (
    init_db, ensure_schema, insert_tick_row, log_prediction,
    open_trade, close_trade, price_stats, price_at_or_after,
    backfill_ticks,
)
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws


# ---------------- Helpers ----------------
def now_ts() -> int:
    return int(time.time())


def is_market_open(ts: Optional[int] = None) -> bool:
    ts = ts or now_ts()
    dt = datetime.utcfromtimestamp(ts) + timedelta(minutes=MARKET_TZ_OFFSET_MIN)
    if dt.weekday() >= 5:
        return False
    after_open = (dt.hour, dt.minute) >= MARKET_OPEN
    before_close = (dt.hour, dt.minute) <= MARKET_CLOSE
    return after_open and before_close


def chunked(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ---------------- Online Forecaster ----------------
class OnlinePriceForecaster:
    def __init__(self, symbols: List[str], horizon_secs: int = PREDICT_HORIZON_SECS):
        self.horizon = horizon_secs
        self.models = {
            s: SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=1e-5,
                learning_rate="optimal",
                random_state=42,
            )
            for s in symbols
        }
        self.scalers = {s: StandardScaler() for s in symbols}
        self.trained_counts = {s: 0 for s in symbols}
        self.buffers = {s: deque(maxlen=10000) for s in symbols}
        self.pending = {s: deque() for s in symbols}
        self.lock = threading.Lock()

    def features(self, sym: str, now_ts_val: int) -> Optional[np.ndarray]:
        buf = self.buffers[sym]
        if not buf:
            return None
        times, prices = zip(*buf)
        times = np.array(times)
        prices = np.array(prices, dtype=float)
        price_now = prices[-1]

        feats: List[float] = []
        # returns over multiple windows
        for W in [5, 15, 30, 60, 120, 300]:
            cutoff = now_ts_val - W
            idx = np.searchsorted(times, cutoff, side="left")
            p0 = prices[idx] if idx < len(prices) else prices[-1]
            ret = (price_now - p0) / p0 * 100.0 if p0 > 0 else 0.0
            feats.append(ret)

        # micro-momentum
        last5 = prices[-5:] if len(prices) >= 5 else prices
        if len(last5) > 1:
            feats.append((last5[-1] - last5[0]) / last5[0] * 100.0)
        else:
            feats.append(0.0)
        return np.array(feats)

    def on_tick(self, sym: str, ts: int, price: float) -> Optional[Tuple[float, float]]:
        with self.lock:
            self.buffers[sym].append((ts, price))
            X = self.features(sym, ts)
            if X is None:
                return None
            # schedule label
            self.pending[sym].append((ts, price, X, ts + self.horizon))

            if self.trained_counts[sym] >= MIN_TRAIN_SNAPSHOTS:
                scaler, model = self.scalers[sym], self.models[sym]
                x = scaler.transform([X])
                delta = float(model.predict(x)[0])  # % move prediction
                pred_price = price * (1 + delta / 100.0)
                return delta, pred_price
            return None

    def training_step(self, sym: str, now_val: int):
        with self.lock:
            while self.pending[sym] and self.pending[sym][0][3] <= now_val:
                ts0, price0, X, t_target = self.pending[sym].popleft()
                future = price_at_or_after(sym, t_target)
                if future is None:
                    continue
                y = (future - price0) / price0 * 100.0
                sc, model = self.scalers[sym], self.models[sym]
                sc.partial_fit([X])
                model.partial_fit(sc.transform([X]), [y])
                self.trained_counts[sym] += 1


# ---------------- Trader ----------------
class Trader:
    def __init__(self, fy: Optional[fyersModel.FyersModel], allow_short: bool = ALLOW_SHORT):
        self.fy = fy
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.allow_short = allow_short

    def place_market(self, sym: str, side: str, qty: int) -> bool:
        if DRY_RUN or self.fy is None:
            print(f"[PAPER-ORDER] {side} {qty} {sym}")
            return True
        try:
            data = {
                "symbol": sym,
                "qty": qty,
                "type": 2,  # MARKET
                "side": 1 if side == "BUY" else -1,
                "productType": "INTRADAY",
                "validity": "DAY",
                "offlineOrder": False,
            }
            r = self.fy.place_order(data=data)
            print(f"[ORDER RESP] {r}")
            return isinstance(r, dict) and r.get("s") == "ok"
        except Exception as e:
            print(f"[ORDER ERR] {e}")
            return False

    def maybe_enter(self, sym: str, ts: int, price: float, delta: float, pred_price: float):
        with self.lock:
            if sym in self.positions:
                return
            move = (pred_price - price) / price * 100.0
            if move >= ENTRY_THRESHOLD_PCT:
                side = "BUY"
            elif move <= -ENTRY_THRESHOLD_PCT and self.allow_short:
                side = "SELL"
            else:
                return
            tp = price * (1 + TAKE_PROFIT_PCT / 100.0) if side == "BUY" else price * (1 - TAKE_PROFIT_PCT / 100.0)
            sl = price * (1 - STOP_LOSS_PCT / 100.0) if side == "BUY" else price * (1 + STOP_LOSS_PCT / 100.0)
            if self.place_market(sym, side, POSITION_QTY):
                tid = open_trade(sym, side, POSITION_QTY, ts, price, tp, sl)
                self.positions[sym] = {"id": tid, "side": side, "entry": price, "tp": tp, "sl": sl}
                print(f"[ENTER {side}] {sym} @ {price:.2f} TP={tp:.2f} SL={sl:.2f}")

    def on_tick_manage(self, sym: str, ts: int, price: float):
        with self.lock:
            pos = self.positions.get(sym)
            if not pos:
                return
            if pos["side"] == "BUY":
                if price >= pos["tp"]:
                    print(f"[EXIT TP] {sym} @ {price:.2f}")
                    close_trade(pos["id"], ts, price)
                    self.positions.pop(sym, None)
                elif price <= pos["sl"]:
                    print(f"[EXIT SL] {sym} @ {price:.2f}")
                    close_trade(pos["id"], ts, price)
                    self.positions.pop(sym, None)
            else:  # SELL
                if price <= pos["tp"]:
                    print(f"[EXIT TP] {sym} @ {price:.2f}")
                    close_trade(pos["id"], ts, price)
                    self.positions.pop(sym, None)
                elif price >= pos["sl"]:
                    print(f"[EXIT SL] {sym} @ {price:.2f}")
                    close_trade(pos["id"], ts, price)
                    self.positions.pop(sym, None)

    def flatten_all(self, ts: int):
        with self.lock:
            for sym, pos in list(self.positions.items()):
                px = price_at_or_after(sym, ts) or pos["entry"]
                print(f"[EOD FLATTEN] {sym} @ {px:.2f}")
                close_trade(pos["id"], ts, px)
                self.positions.pop(sym, None)


# ---------------- WS Runner ----------------
class WSRunner:
    def __init__(self, access_token: str, symbols: List[str], on_tick_cb):
        self._at = f"{APP_ID}:{access_token}"
        self.symbols = symbols
        self.fy = None
        self._connected = threading.Event()
        self.cb = on_tick_cb

    def on_message(self, msg):
        now = now_ts()
        try:
            if isinstance(msg, list):
                for m in msg:
                    sym = m.get("symbol") or m.get("symbolName") or m.get("n")
                    v = m.get("v") or {}
                    ltp = m.get("ltp") or v.get("lp") or m.get("price") or m.get("last_traded_price")
                    if sym and ltp is not None:
                        insert_tick_row(sym, now, m)
                        self.cb(sym, now, float(ltp))
            elif isinstance(msg, dict):
                if msg.get("type") in {"cn", "ful", "sub"}:
                    print("[WS]", msg)
                    return
                sym = msg.get("symbol") or msg.get("symbolName") or msg.get("n")
                v = msg.get("v") or {}
                ltp = msg.get("ltp") or v.get("lp") or msg.get("price") or msg.get("last_traded_price")
                if sym and ltp is not None:
                    insert_tick_row(sym, now, msg)
                    self.cb(sym, now, float(ltp))
        except Exception as e:
            print(f"[WS parse warn] {e}; raw={msg}")

    def on_error(self, m): print("[WS error]", m)
    def on_close(self, m): print("[WS close]", m)

    def on_open(self):
        print("[WS] Connected. Subscribing…")
        self.fy.subscribe(symbols=self.symbols, data_type="SymbolUpdate")
        self._connected.set()
        self.fy.keep_running()

    def start(self):
        print("[WS] Starting socket…")
        self.fy = data_ws.FyersDataSocket(
            access_token=self._at,
            log_path=LOG_PATH,
            litemode=WS_LITE_MODE,
            write_to_file=False,
            reconnect=True,
            on_connect=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        self.fy.connect()


# ---------------- Main ----------------
_stop = threading.Event()


def _handle_sig(s, f):
    print("[INFO] Shutdown requested…")
    _stop.set()


signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)


def main():
    # DB + schema
    init_db()
    ensure_schema()
    print(f"[INIT] DB at {os.path.abspath(DB_PATH)}")
    print(f"[INIT] Symbols: {SYMBOLS}")

    # Auth
    print("[AUTH] Ensuring tokens…")
    tokens = ensure_tokens()
    at = tokens["access_token"]
    print("[AUTH] OK.")

    # FYERS client (only used if DRY_RUN=False)
    fy = fyersModel.FyersModel(client_id=APP_ID, token=f"{APP_ID}:{at}", is_async=False, log_path=LOG_PATH)

    # ML + Trader
    forecaster = OnlinePriceForecaster(SYMBOLS)
    trader = Trader(fy if not DRY_RUN else None)
    print(f"[RUN] DRY_RUN={DRY_RUN}  POLL={QUOTE_POLL_SECS}s  HORIZON={PREDICT_HORIZON_SECS}s")

    # Tick callback
    def on_tick(sym: str, ts: int, price: float):
        pred = forecaster.on_tick(sym, ts, price)
        trader.on_tick_manage(sym, ts, price)
        if pred and is_market_open(ts):
            delta, pred_p = pred
            direction = "UP" if pred_p >= price else "DOWN"
            _ = log_prediction(sym, ts, PREDICT_HORIZON_SECS, price, pred_p, direction, ENTRY_THRESHOLD_PCT, ts + PREDICT_HORIZON_SECS)
            trader.maybe_enter(sym, ts, price, delta, pred_p)

    # WebSocket (best-effort; REST is also active)
    ws = WSRunner(at, SYMBOLS, on_tick)
    threading.Thread(target=ws.start, daemon=True).start()
    waited = 0.0
    while not ws._connected.is_set() and waited < 10.0 and not _stop.is_set():
        time.sleep(0.5)
        waited += 0.5
    if ws._connected.is_set():
        print("[WS] Subscribed.")
    else:
        print("[WS] Not connected yet (will keep trying via SDK).")

    # Background trainer thread
    def bg_trainer():
        while not _stop.is_set():
            now_val = now_ts()
            for s in SYMBOLS:
                forecaster.training_step(s, now_val)
            time.sleep(0.5)

    threading.Thread(target=bg_trainer, daemon=True).start()

    # Main loop: REST fallback + token refresh + EOD flatten + heartbeat
    loop_no = 0
    last_token_check = 0
    while not _stop.is_set():
        loop_no += 1

        # Token refresh (every ~60s)
        if now_ts() - last_token_check > 60:
            if not access_token_valid(at):
                try:
                    t = load_tokens()
                    rt = t.get("refresh_token")
                    if rt:
                        print("[AUTH] Refreshing access token…")
                        newt = refresh_access_token(rt)
                        if "access_token" not in newt and "accessToken" in newt:
                            newt["access_token"] = newt["accessToken"]
                        if "refresh_token" not in newt and "refreshToken" in newt:
                            newt["refresh_token"] = newt["refreshToken"]
                        if "refresh_token" not in newt:
                            newt["refresh_token"] = rt
                        save_tokens(newt)
                        at = newt["access_token"]
                        # Update SDK token (WS SDK needs reconnect to use new token; REST will work)
                        fy.token = f"{APP_ID}:{at}"
                        print("[AUTH] Token refreshed.")
                except Exception as e:
                    print(f"[AUTH warn] refresh failed: {e}")
            last_token_check = now_ts()

        # REST fallback
        try:
            headers = {"Authorization": f"{APP_ID}:{at}"}
            inserted = 0
            now_val = now_ts()
            for chunk in chunked(SYMBOLS, 50):
                url = f"{DATA_BASE}/data/quotes"
                r = SESSION.get(url, headers=headers, params={"symbols": ",".join(chunk)}, timeout=8)
                j = r.json()
                if j.get("s") != "ok":
                    print("[REST warn] non-ok response:", j)
                    continue
                for row in j.get("d", []):
                    sym = row.get("symbol") or row.get("n") or row.get("symbolName")
                    v = row.get("v") or {}
                    ltp = row.get("ltp") or v.get("lp") or row.get("price") or row.get("last_traded_price")
                    if sym and ltp is not None:
                        insert_tick_row(sym, now_val, row)
                        on_tick(sym, now_val, float(ltp))
                        inserted += 1
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[REST {stamp}] loop={loop_no} inserted={inserted} symbols={len(SYMBOLS)}")
        except Exception as e:
            print("[REST warn] exception:", e)

        # EOD flatten (only once after close — here we check every loop; harmless)
        if not is_market_open():
            trader.flatten_all(now_ts())

        # Sleep between REST polls
        for _ in range(QUOTE_POLL_SECS):
            if _stop.is_set():
                break
            time.sleep(1)

    print("[INFO] Bye.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backfill", action="store_true", help="Re-parse raw_json into structured columns.")
    p.add_argument("--start", help="Backfill start date (YYYY-MM-DD)")
    p.add_argument("--end", help="Backfill end date (YYYY-MM-DD)")
    p.add_argument("--symbols", help="Comma-separated symbols to restrict backfill")
    a = p.parse_args()
    if a.backfill:
        syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else None
        backfill_ticks(a.start, a.end, syms)
        sys.exit(0)
    main()
