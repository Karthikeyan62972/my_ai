#!/usr/bin/env python3
"""SQLite storage, schema, alias-aware tick insert, metrics & backfill."""
import sqlite3, json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from settings import DB_PATH

# ---------- Schema ----------
DDL = {
    "ticks": (
        "CREATE TABLE IF NOT EXISTS ticks ("
        "symbol TEXT NOT NULL, ts INTEGER NOT NULL, ltp REAL, "
        "bid REAL, ask REAL, bid_qty REAL, ask_qty REAL, "
        "open REAL, high REAL, low REAL, prev_close REAL, vwap REAL, "
        "vol_traded_today REAL, last_traded_qty REAL, raw_json TEXT, "
        "PRIMARY KEY(symbol, ts))"
    ),
    "predictions": (
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, ts INTEGER, horizon INTEGER, "
        "price_now REAL, pred_price REAL, direction TEXT, threshold_pct REAL, "
        "expiry_ts INTEGER, hit INTEGER DEFAULT NULL)"
    ),
    "trades": (
        "CREATE TABLE IF NOT EXISTS trades ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, "
        "entry_ts INTEGER, entry_price REAL, target_price REAL, stop_price REAL, "
        "exit_ts INTEGER, exit_price REAL, pnl REAL, status TEXT)"
    ),
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, ts)",
    "CREATE INDEX IF NOT EXISTS idx_preds_symbol_expiry ON predictions(symbol, expiry_ts)",
]

@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db_conn() as c:
        for ddl in DDL.values():
            c.execute(ddl)
        for idx in INDEXES:
            c.execute(idx)

def ensure_schema():
    required_text = {"raw_json"}
    required_real = {
        "bid","ask","bid_qty","ask_qty","open","high","low",
        "prev_close","vwap","vol_traded_today","last_traded_qty"
    }
    with db_conn() as c:
        cur = c.execute("PRAGMA table_info(ticks)")
        cols = {row[1] for row in cur.fetchall()} if cur else set()
        for col in sorted(required_real - cols):
            c.execute(f"ALTER TABLE ticks ADD COLUMN {col} REAL")
        for col in sorted(required_text - cols):
            c.execute(f"ALTER TABLE ticks ADD COLUMN {col} TEXT")

# ---------- alias-aware mapping ----------
def _pick(row: Dict[str, Any], *aliases, default=None, cast=float):
    v = row.get("v") or {}
    for a in aliases:
        if a in row and row[a] is not None:
            return cast(row[a]) if (cast and row[a] is not None) else row[a]
        if a in v and v[a] is not None:
            return cast(v[a]) if (cast and v[a] is not None) else v[a]
    return default

def insert_tick_row(symbol: str, ts: int, row: Dict[str, Any]):
    ltp = _pick(row, "ltp", "price", "last_traded_price", "lp")
    data = {
        "symbol": symbol,
        "ts": ts,
        "ltp": float(ltp) if ltp is not None else None,
        "bid": _pick(row, "bid", "bp"),
        "ask": _pick(row, "ask", "ap"),
        "bid_qty": _pick(row, "bid_qty", "bq"),
        "ask_qty": _pick(row, "ask_qty", "aq"),
        "open": _pick(row, "open", "open_price", "o"),
        "high": _pick(row, "high", "high_price", "h"),
        "low": _pick(row, "low", "low_price", "l"),
        "prev_close": _pick(row, "prev_close", "prev_close_price", "pc"),
        "vwap": _pick(row, "vwap", "atp"),
        "vol_traded_today": _pick(row, "vol_traded_today", "volume"),
        "last_traded_qty": _pick(row, "last_traded_qty", "ltq"),
        "raw_json": json.dumps(row, separators=(",", ":"), ensure_ascii=False),
    }
    with db_conn() as c:
        c.execute(
            """
            INSERT INTO ticks(symbol, ts, ltp, bid, ask, bid_qty, ask_qty, open, high, low, prev_close,
                              vwap, vol_traded_today, last_traded_qty, raw_json)
            VALUES (:symbol, :ts, :ltp, :bid, :ask, :bid_qty, :ask_qty, :open, :high, :low, :prev_close,
                    :vwap, :vol_traded_today, :last_traded_qty, :raw_json)
            ON CONFLICT(symbol, ts) DO UPDATE SET
              ltp=excluded.ltp,
              bid=excluded.bid, ask=excluded.ask,
              bid_qty=excluded.bid_qty, ask_qty=excluded.ask_qty,
              open=excluded.open, high=excluded.high, low=excluded.low, prev_close=excluded.prev_close,
              vwap=excluded.vwap, vol_traded_today=excluded.vol_traded_today, last_traded_qty=excluded.last_traded_qty,
              raw_json=excluded.raw_json
            """,
            data,
        )

# ---------- query helpers ----------
def price_stats(symbol: str, start_ts: int, end_ts: int) -> Tuple[Optional[float], Optional[float]]:
    with db_conn() as c:
        cur = c.execute(
            "SELECT MIN(ltp), MAX(ltp) FROM ticks WHERE symbol=? AND ts BETWEEN ? AND ?",
            (symbol, start_ts, end_ts),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (None, None)

def price_at_or_after(symbol: str, ts: int) -> Optional[float]:
    with db_conn() as c:
        cur = c.execute(
            "SELECT ltp FROM ticks WHERE symbol=? AND ts>=? ORDER BY ts ASC LIMIT 1",
            (symbol, ts),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None

def log_prediction(symbol: str, ts: int, horizon: int, price_now: float, pred_price: float,
                   direction: str, threshold_pct: float, expiry_ts: int) -> int:
    with db_conn() as c:
        cur = c.execute(
            "INSERT INTO predictions(symbol, ts, horizon, price_now, pred_price, direction, threshold_pct, expiry_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, ts, horizon, price_now, pred_price, direction, threshold_pct, expiry_ts),
        )
        return cur.lastrowid

def mark_prediction_hit(pred_id: int, hit: int):
    with db_conn() as c:
        c.execute("UPDATE predictions SET hit=? WHERE id=?", (hit, pred_id))

def open_trade(symbol: str, side: str, qty: int, entry_ts: int, entry_price: float,
               target_price: float, stop_price: float) -> int:
    with db_conn() as c:
        cur = c.execute(
            "INSERT INTO trades(symbol, side, qty, entry_ts, entry_price, target_price, stop_price, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')",
            (symbol, side, qty, entry_ts, entry_price, target_price, stop_price),
        )
        return cur.lastrowid

def close_trade(trade_id: int, exit_ts: int, exit_price: float):
    with db_conn() as c:
        cur = c.execute("SELECT side, qty, entry_price FROM trades WHERE id=?", (trade_id,))
        row = cur.fetchone()
        if not row: return
        side, qty, entry_price = row
        pnl_per_share = (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
        pnl = pnl_per_share * qty
        c.execute(
            "UPDATE trades SET exit_ts=?, exit_price=?, pnl=?, status='CLOSED' WHERE id=?",
            (exit_ts, exit_price, pnl, trade_id),
        )

# ---------- backfill ----------
def _reparse_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ltp": _pick(row, "ltp", "price", "last_traded_price", "lp"),
        "bid": _pick(row, "bid", "bp"),
        "ask": _pick(row, "ask", "ap"),
        "bid_qty": _pick(row, "bid_qty", "bq"),
        "ask_qty": _pick(row, "ask_qty", "aq"),
        "open": _pick(row, "open", "open_price", "o"),
        "high": _pick(row, "high", "high_price", "h"),
        "low": _pick(row, "low", "low_price", "l"),
        "prev_close": _pick(row, "prev_close", "prev_close_price", "pc"),
        "vwap": _pick(row, "vwap", "atp"),
        "vol_traded_today": _pick(row, "vol_traded_today", "volume"),
        "last_traded_qty": _pick(row, "last_traded_qty", "ltq"),
    }

def backfill_ticks(start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   symbols: Optional[List[str]] = None,
                   batch_size: int = 2000) -> int:
    """
    Re-parse raw_json for each tick and fill any NULL columns (ltp/bid/ask/open/high/low/pc/vwap/volume/ltq).
    You can optionally filter by date range (YYYY-MM-DD) and/or symbols.
    Returns the number of rows updated.
    """
    where = ["raw_json IS NOT NULL"]
    params: List[Any] = []

    # Date filters (inclusive)
    if start_date:
        start_ts = int(datetime.fromisoformat(start_date).timestamp())
        where.append("ts >= ?")
        params.append(start_ts)
    if end_date:
        dt_end = datetime.fromisoformat(end_date)
        end_ts = int((dt_end + timedelta(days=1)).timestamp()) - 1
        where.append("ts <= ?")
        params.append(end_ts)

    # Symbol filter
    if symbols:
        placeholders = ",".join(["?"] * len(symbols))
        where.append(f"symbol IN ({placeholders})")
        params.extend(symbols)

    where_sql = " AND ".join(where)
    updated = 0

    with db_conn() as c:
        sel_cur = c.execute(
            f"SELECT symbol, ts, raw_json FROM ticks WHERE {where_sql} ORDER BY ts ASC",
            tuple(params),
        )
        while True:
            rows = sel_cur.fetchmany(batch_size)
            if not rows:
                break

            for sym, ts, raw in rows:
                try:
                    row = json.loads(raw)
                except Exception:
                    continue

                cols = _reparse_columns(row)
                upd_cur = c.execute(
                    """
                    UPDATE ticks SET
                        ltp               = COALESCE(:ltp, ltp),
                        bid               = COALESCE(:bid, bid),
                        ask               = COALESCE(:ask, ask),
                        bid_qty           = COALESCE(:bid_qty, bid_qty),
                        ask_qty           = COALESCE(:ask_qty, ask_qty),
                        open              = COALESCE(:open, open),
                        high              = COALESCE(:high, high),
                        low               = COALESCE(:low, low),
                        prev_close        = COALESCE(:prev_close, prev_close),
                        vwap              = COALESCE(:vwap, vwap),
                        vol_traded_today  = COALESCE(:vol_traded_today, vol_traded_today),
                        last_traded_qty   = COALESCE(:last_traded_qty, last_traded_qty)
                    WHERE symbol = :symbol AND ts = :ts
                    """,
                    {**cols, "symbol": sym, "ts": ts},
                )
                updated += (upd_cur.rowcount or 0)

    print(f"[BACKFILL] Updated rows: {updated}")
    return updated
