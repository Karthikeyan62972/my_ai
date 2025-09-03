#!/usr/bin/env python3
"""
FastAPI backend for Market DB (market.db)
Serves JSON for: symbols, candles, latest tick, predictions, trades, daily metrics.
"""

import os, sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = os.environ.get("FYERS_DB_PATH", "market.db")
IST = timezone(timedelta(hours=5, minutes=30))

app = FastAPI(title="Intraday ML Trader UI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

def conn():
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def to_ms(ts: int) -> int:
    # SQLite stores seconds (int); UI wants ms
    return int(ts) * 1000

# ---------- Models ----------
class Candle(BaseModel):
    t: int   # epoch ms
    o: float
    h: float
    l: float
    c: float
    v: float

class Tick(BaseModel):
    t: int
    ltp: float
    bid: float | None = None
    ask: float | None = None
    v: float | None = None

class DailyMetrics(BaseModel):
    date: str
    predictions: int
    hits: int
    hit_rate_pct: float

# ---------- Endpoints ----------

@app.get("/api/symbols", response_model=List[str])
def get_symbols():
    with conn() as c:
        rows = c.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol").fetchall()
    return [r["symbol"] for r in rows]

@app.get("/api/last", response_model=Tick)
def latest_tick(symbol: str = Query(..., description="e.g., NSE:SBIN-EQ")):
    with conn() as c:
        r = c.execute(
            "SELECT symbol, ts, ltp, bid, ask, vol_traded_today AS v "
            "FROM ticks WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    if not r:
        raise HTTPException(404, f"No ticks for {symbol}")
    return Tick(t=to_ms(r["ts"]), ltp=r["ltp"], bid=r["bid"], ask=r["ask"], v=r["v"])

@app.get("/api/candles", response_model=List[Candle])
def get_candles(
    symbol: str = Query(...),
    interval: str = Query("1m", pattern=r"^(1m|5m|15m|1h)$"),
    lookback: int = Query(180, ge=10, le=3000, description="number of candles"),
):
    # Bucket ts into intervals and compute OHLC from per-second ticks.
    # Volume approximated from diff(vol_traded_today) sum per bucket (if present).
    bucket_sec = {"1m":60, "5m":300, "15m":900, "1h":3600}[interval]
    with conn() as c:
        # Pull enough rows for the requested lookback with a small buffer
        rows = c.execute(
            "SELECT ts, ltp, vol_traded_today FROM ticks WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, lookback * 120),  # heuristic buffer
        ).fetchall()
    if not rows:
        return []
    # Build buckets
    from collections import defaultdict
    buckets: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"prices": [], "ts": None, "vols": []})
    for r in rows:
        ts = int(r["ts"])
        bucket = (ts // bucket_sec) * bucket_sec
        d = buckets[bucket]
        d["ts"] = bucket
        if r["ltp"] is not None:
            d["prices"].append(float(r["ltp"]))
        d["vols"].append(r["vol_traded_today"] if r["vol_traded_today"] is not None else None)
    # Turn into candles
    out: List[Candle] = []
    for bts in sorted(buckets.keys()):
        d = buckets[bts]
        p = d["prices"]
        if not p:  # skip empty
            continue
        o = p[-1]      # because we selected DESC; last seen in this bucket is earliest
        c_ = p[0]      # first in this list is latest
        h = max(p)
        l = min(p)
        # volume via diffs
        vlist = [x for x in d["vols"] if x is not None]
        v = 0.0
        if len(vlist) >= 2:
            v = sum(max(0.0, float(vlist[i] - vlist[i+1])) for i in range(len(vlist)-1))
        out.append(Candle(t=to_ms(d["ts"]), o=o, h=h, l=l, c=c_, v=round(v, 2)))
    # Keep the most recent N
    return out[-lookback:]

@app.get("/api/predictions")
def get_predictions(symbol: str, limit: int = 200):
    with conn() as c:
        rows = c.execute(
            "SELECT id, ts, price_now, pred_price, direction, expiry_ts, hit "
            "FROM predictions WHERE symbol=? ORDER BY id DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "t": to_ms(r["ts"]),
            "price_now": r["price_now"],
            "pred_price": r["pred_price"],
            "direction": r["direction"],
            "expiry": to_ms(r["expiry_ts"]),
            "hit": r["hit"],
        }
        for r in rows
    ]

@app.get("/api/trades")
def get_trades(status: Optional[str] = Query(None, pattern=r"^(OPEN|CLOSED)$")):
    sql = "SELECT id, symbol, side, qty, entry_ts, entry_price, target_price, stop_price, exit_ts, exit_price, pnl, status FROM trades"
    params: List[Any] = []
    if status:
        sql += " WHERE status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT 500"
    with conn() as c:
        rows = c.execute(sql, tuple(params)).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "symbol": r["symbol"], "side": r["side"], "qty": r["qty"],
            "entry_t": to_ms(r["entry_ts"]),
            "entry_price": r["entry_price"], "tp": r["target_price"], "sl": r["stop_price"],
            "exit_t": to_ms(r["exit_ts"]) if r["exit_ts"] else None,
            "exit_price": r["exit_price"], "pnl": r["pnl"], "status": r["status"]
        })
    return out

@app.get("/api/metrics/daily", response_model=DailyMetrics)
def daily_metrics(date: Optional[str] = None):
    if date:
        day = datetime.fromisoformat(date).date()
    else:
        day = datetime.now(IST).date()
    start = int(datetime(day.year, day.month, day.day, tzinfo=IST).timestamp())
    end = start + 86400 - 1
    with conn() as c:
        r = c.execute(
            "SELECT count(*) AS total, sum(CASE WHEN hit=1 THEN 1 ELSE 0 END) AS hits "
            "FROM predictions WHERE ts BETWEEN ? AND ?",
            (start, end),
        ).fetchone()
    total = int(r["total"] or 0)
    hits = int(r["hits"] or 0)
    hr = round((hits / total * 100.0) if total else 0.0, 2)
    return DailyMetrics(date=str(day), predictions=total, hits=hits, hit_rate_pct=hr)


# ===== Ingestion Watchdog =====
from datetime import timezone

DEFAULT_STALE_THRESHOLD = 30  # seconds; tweak as you like

def _now_sec() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def _last_tick_row(symbol: str):
    with conn() as c:
        return c.execute(
            "SELECT ts FROM ticks WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,),
        ).fetchone()

def _count_since(symbol: str, since_ts: int) -> int:
    with conn() as c:
        r = c.execute(
            "SELECT COUNT(*) AS n FROM ticks WHERE symbol=? AND ts>=?",
            (symbol, since_ts),
        ).fetchone()
    return int(r["n"] or 0)

@app.get("/api/ingestion/status")
def ingestion_status(symbol: str, threshold: int = DEFAULT_STALE_THRESHOLD):
    """
    Returns last tick time and whether data is stale for the symbol.
    """
    row = _last_tick_row(symbol)
    if not row:
        return {
            "symbol": symbol,
            "last_ts": None,
            "last_iso": None,
            "age_sec": None,
            "stale": True,
            "count_1m": 0,
            "count_5m": 0,
            "threshold": threshold,
        }
    last_ts = int(row["ts"])
    now = _now_sec()
    age = max(0, now - last_ts)
    return {
        "symbol": symbol,
        "last_ts": last_ts * 1000,  # ms for convenience
        "last_iso": datetime.fromtimestamp(last_ts, IST).isoformat(),
        "age_sec": age,
        "stale": age > threshold,
        "count_1m": _count_since(symbol, now - 60),
        "count_5m": _count_since(symbol, now - 300),
        "threshold": threshold,
    }

@app.get("/api/ingestion/summary")
def ingestion_summary(threshold: int = DEFAULT_STALE_THRESHOLD):
    """
    Returns status for all symbols that exist in ticks.
    """
    with conn() as c:
        rows = c.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol").fetchall()
    syms = [r["symbol"] for r in rows]
    out = []
    for s in syms:
        row = _last_tick_row(s)
        if not row:
            out.append({"symbol": s, "last_ts": None, "age_sec": None, "stale": True})
            continue
        last_ts = int(row["ts"])
        now = _now_sec()
        age = max(0, now - last_ts)
        out.append({
            "symbol": s,
            "last_ts": last_ts * 1000,
            "last_iso": datetime.fromtimestamp(last_ts, IST).isoformat(),
            "age_sec": age,
            "stale": age > threshold,
            "count_1m": _count_since(s, now - 60),
            "count_5m": _count_since(s, now - 300),
        })
    return {"threshold": threshold, "symbols": out}