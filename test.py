#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust 15-minute ahead price predictor with dense resampling, numeric-only features,
and safe fallbacks (EWMA, naive) if ML can't train.

Usage:
  python3 test.py --db /home/karthik/market.db --symbol "NSE:DRREDDY-EQ" --horizon-min 15

Stdout (exact):
  ltp , predicted traded price for next 15 min , Risk signal , threshold(ret)

Diagnostics go to stderr.
"""

import argparse
import sqlite3
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None

QUERY = """
SELECT symbol, ts, ltp, bid, ask, bid_qty, ask_qty, vwap, vol_traded_today,
       last_traded_qty, exch_feed_time, bid_size, ask_size, bid_price, ask_price,
       tot_buy_qty, tot_sell_qty, avg_trade_price, ch, chp, last_traded_time
FROM ticks
WHERE symbol = ?
ORDER BY ts ASC
"""

# -------------- Utilities --------------

def _parse_any_ts(series: pd.Series) -> pd.Series:
    """Parse timestamps that might be strings or unix s/ms/us/ns; return UTC-aware."""
    s = series.copy()
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, utc=True, errors="coerce")

    if s.dtype == object:
        out = pd.to_datetime(s, utc=True, errors="coerce", infer_datetime_format=True)
        if out.notna().any():
            return out

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() == 0:
        return pd.to_datetime(s, utc=True, errors="coerce")

    maxabs = np.nanmax(np.abs(s_num.values.astype("float64")))
    if maxabs < 1e11:   # seconds
        return pd.to_datetime(s_num, unit="s", utc=True, errors="coerce")
    elif maxabs < 1e14: # ms
        return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
    elif maxabs < 1e17: # us
        return pd.to_datetime(s_num, unit="us", utc=True, errors="coerce")
    else:               # ns
        return pd.to_datetime(s_num, unit="ns", utc=True, errors="coerce")


def _safe_pct(x: float) -> float:
    return 100.0 * (np.expm1(x) if np.isfinite(x) else 0.0)


# -------------- Data IO --------------

def read_ticks(db_path: str, symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path, timeout=30)  # wait up to 30s for lock
    try:
        con.execute("PRAGMA journal_mode=WAL;")  # better concurrency
        df = pd.read_sql_query(QUERY, con, params=[symbol])
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No data for symbol={symbol}")

    # Parse timestamps safely
    for c in ["ts", "exch_feed_time", "last_traded_time"]:
        if c in df.columns:
            df[c] = _parse_any_ts(df[c])

    # Deduplicate/clean
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # Force numerics where applicable
    numeric_cols = [
        "ltp","bid","ask","bid_qty","ask_qty","vwap","vol_traded_today","last_traded_qty",
        "bid_size","ask_size","bid_price","ask_price","tot_buy_qty","tot_sell_qty",
        "avg_trade_price","ch","chp"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["ts"].notna() & df["ltp"].notna()]
    if df.empty:
        raise ValueError("No valid ts/ltp rows after parsing.")

    return df


# -------------- Feature Engineering (ticks -> features -> minute bars) --------------

def add_tick_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["mid_price"] = np.where(
        d[["bid", "ask"]].notna().all(axis=1),
        (d["bid"] + d["ask"]) / 2.0,
        d["ltp"]
    )
    d["spread"] = d["ask"] - d["bid"]
    d["rel_spread_bps"] = 1e4 * (d["spread"] / d["mid_price"])

    d["depth_imbalance"] = np.where(
        d[["tot_buy_qty", "tot_sell_qty"]].notna().all(axis=1),
        (d["tot_buy_qty"] - d["tot_sell_qty"]) / (d["tot_buy_qty"] + d["tot_sell_qty"] + 1e-9),
        np.nan
    )
    d["micro_price"] = np.where(
        d[["bid_price", "ask_price", "bid_size", "ask_size"]].notna().all(axis=1),
        (d["bid_price"] * d["ask_size"] + d["ask_price"] * d["bid_size"]) / (d["bid_size"] + d["ask_size"] + 1e-9),
        d["mid_price"]
    )

    d["ltp_log"] = np.log(d["ltp"].replace(0, np.nan))
    d["ret_1"] = d["ltp_log"].diff().fillna(0)
    d["ret_5"] = d["ltp_log"].diff(5).fillna(0)
    d["ret_10"] = d["ltp_log"].diff(10).fillna(0)

    # Time-of-day features (UTC)
    seconds = (d["ts"].dt.hour * 3600 + d["ts"].dt.minute * 60 + d["ts"].dt.second).astype(float)
    day = 24 * 3600.0
    d["tod_sin"] = np.sin(2 * np.pi * seconds / day)
    d["tod_cos"] = np.cos(2 * np.pi * seconds / day)

    # Activity
    if "vol_traded_today" in d.columns:
        d["d_vol"] = d["vol_traded_today"].diff().clip(lower=0)
    if "last_traded_qty" in d.columns:
        d["d_qty"] = d["last_traded_qty"].fillna(0).astype(float)
    d["trade_intensity_30"] = pd.Series(d.get("d_qty", np.nan)).rolling(30, min_periods=5).mean()

    return d.replace([np.inf, -np.inf], np.nan)


def to_minute_bars(d: pd.DataFrame) -> pd.DataFrame:
    """
    Downsample ticks to 1-minute bars.
    Keep **only numeric columns** during aggregation to prevent datetime leakage.
    """
    d = d.set_index("ts").sort_index()

    # Keep only numeric columns to avoid datetimes/objects getting into features
    d_num = d.select_dtypes(include=[np.number])

    # Build minute index
    minute_index = pd.date_range(
        start=d.index.min().floor("T"),
        end=d.index.max().ceil("T"),
        freq="T",
        tz="UTC"
    )

    # Sum columns for counts/qty; last for price-like
    sum_cols = [c for c in ["d_qty"] if c in d_num.columns]
    sum_map = {c: "sum" for c in sum_cols}
    last_cols = [c for c in d_num.columns if c not in sum_cols]
    last_map = {c: "last" for c in last_cols}
    agg_map = {**last_map, **sum_map}

    bars = (
        d_num.resample("T")
            .agg(agg_map)
            .reindex(minute_index)
    )

    # Forward-fill for price-like series
    for c in ["ltp","bid","ask","mid_price","micro_price","vwap","avg_trade_price",
              "bid_price","ask_price","tot_buy_qty","tot_sell_qty","bid_size","ask_size",
              "spread","rel_spread_bps","depth_imbalance","ret_1","ret_5","ret_10",
              "tod_sin","tod_cos","trade_intensity_30","ch","chp"]:
        if c in bars.columns:
            bars[c] = bars[c].ffill()

    bars["ts"] = bars.index
    return bars.reset_index(drop=True)


def make_supervised(df_min: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Create supervised rows using dense minute index: target = ltp shifted -h."""
    d = df_min.copy().sort_values("ts").reset_index(drop=True)

    # Rolling features (on minute bars)
    for w in [5, 15, 30, 60, 120]:
        d[f"roll_mean_{w}"] = d["ltp"].rolling(w, min_periods=max(3, w//2)).mean()
        d[f"roll_std_{w}"] = d["ltp"].rolling(w, min_periods=max(3, w//2)).std()
        if "vwap" in d.columns:
            d[f"roll_vwap_dev_{w}"] = (d["vwap"] - d["ltp"]).rolling(w, min_periods=max(3, w//2)).mean()

    # Target price after horizon minutes
    d["y_price"] = d["ltp"].shift(-horizon_min)
    d["y_ret_log"] = np.log(d["y_price"] / d["ltp"])
    d = d.dropna(subset=["y_price"])  # drops last 'horizon' rows

    # Safety: ensure only numeric features + keep ts and targets
    numeric = d.select_dtypes(include=[np.number]).copy()
    numeric["ts"] = d["ts"].values  # keep ts for ordering/metadata
    return numeric.replace([np.inf, -np.inf], np.nan)


# -------------- Modeling --------------

def build_pipeline(feature_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), feature_cols)
    ])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    return Pipeline([("prep", pre), ("model", model)])


def fit_predict(df_sup: pd.DataFrame, horizon_min: int, cv_splits: int = 5):
    # Use only numeric columns; drop targets & metadata from features
    drop_cols = {"y_price","y_ret_log"}
    # df_sup is already numeric-only except 'ts', which is numeric dtype after selection (posix ns)
    feature_cols = [c for c in df_sup.columns if c not in drop_cols | {"ts"} and np.issubdtype(df_sup[c].dtype, np.number)]

    # Keep latest row for inference
    df_sup = df_sup.sort_values("ts").reset_index(drop=True)

    if len(df_sup) < max(60, horizon_min * 4):
        raise RuntimeError(f"Too few supervised rows ({len(df_sup)}). Need at least {max(60, horizon_min*4)}.")

    X = df_sup[feature_cols]
    y = df_sup["y_price"]

    pipe = build_pipeline(feature_cols)

    # CV
    n_splits = max(2, min(cv_splits, len(df_sup)//50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr, va in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict(X.iloc[va])
        maes.append(mean_absolute_error(y.iloc[va], p))
    cv_mae = float(np.mean(maes)) if maes else np.nan

    # Fit on all and predict for the **last** available minute
    pipe.fit(X, y)
    X_now = X.iloc[[-1]]
    y_pred_price = float(pipe.predict(X_now)[0])
    current_ltp = float(df_sup["ltp"].iloc[-1])

    # Risk thresholds from historical |y_ret|
    abs_rets = np.abs(df_sup["y_ret_log"].dropna().values)
    thr_ret = float(np.nanpercentile(abs_rets, 75)) if abs_rets.size >= 20 else float(np.nanmedian(abs_rets) if abs_rets.size else 0.0)
    pred_ret = float(np.log(y_pred_price / current_ltp))

    signal = "BUY" if pred_ret > thr_ret else ("SELL" if pred_ret < -thr_ret else "NEUTRAL")
    return current_ltp, y_pred_price, signal, thr_ret, cv_mae


# -------------- Fallbacks --------------

def fallback_ewma(minute_df: pd.DataFrame, horizon_min: int) -> Optional[float]:
    """EWMA of per-minute log-returns; returns predicted future price."""
    d = minute_df.sort_values("ts")
    if d.shape[0] < max(10, horizon_min + 5):
        return None
    ltp = d["ltp"].astype(float)
    lret = np.log(ltp / ltp.shift(1)).fillna(0.0)
    ew = lret.ewm(halflife=15, min_periods=5).mean()  # ~15-min half-life
    mu = float(ew.iloc[-1])
    pred = float(ltp.iloc[-1] * np.exp(mu * horizon_min))
    return pred


# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--symbol", default="NSE:DRREDDY-EQ")
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--cv-splits", type=int, default=5)
    args = ap.parse_args()

    try:
        ticks = read_ticks(args.db, args.symbol)
    except Exception as e:
        print(f"ERROR: Failed to read data: {e}", file=sys.stderr)
        sys.exit(2)

    if ticks["ts"].max() - ticks["ts"].min() < pd.Timedelta(minutes=max(2*args.horizon_min, 30)):
        print("# warn: dataset span is short; predictions may be noisy.", file=sys.stderr)

    ticks_feat = add_tick_features(ticks)
    minute_df = to_minute_bars(ticks_feat)

    sup = make_supervised(minute_df, args.horizon_min)

    # Try ML path
    try:
        ltp_now, pred_price, signal, thr_ret, cv_mae = fit_predict(sup, args.horizon_min, args.cv_splits)
        print(f"{ltp_now:.2f} , {pred_price:.2f} , {signal} , {_safe_pct(thr_ret):.3f}%")
        print(f"# info: CV_MAE≈{cv_mae:.4f} ; rows={len(sup)} ; last_ts={pd.to_datetime(minute_df['ts'].iloc[-1], utc=True)}", file=sys.stderr)
        return
    except Exception as e:
        print(f"# warn: ML pipeline fallback due to: {e}", file=sys.stderr)

    # Fallback 1: EWMA drift
    pred = fallback_ewma(minute_df, args.horizon_min)
    if pred is not None and np.isfinite(pred):
        ltp_now = float(minute_df["ltp"].iloc[-1])
        pred_ret = float(np.log(pred / ltp_now))
        lret = np.log(minute_df["ltp"] / minute_df["ltp"].shift(1)).dropna()
        thr_ret = float(np.nanpercentile(np.abs(lret.tail(240)), 75)) if len(lret) >= 20 else 0.0
        signal = "BUY" if pred_ret > thr_ret else ("SELL" if pred_ret < -thr_ret else "NEUTRAL")
        print(f"{ltp_now:.2f} , {pred:.2f} , {signal} , {_safe_pct(thr_ret):d.3f}%")
        print("# info: used EWMA fallback", file=sys.stderr)
        return

    # Fallback 2: naive hold
    ltp_now = float(minute_df["ltp"].iloc[-1])
    print(f"{ltp_now:.2f} , {ltp_now:.2f} , NEUTRAL , {_safe_pct(0.0):.3f}%")
    print("# info: used naive hold fallback", file=sys.stderr)


if __name__ == "__main__":
    main()
