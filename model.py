#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecast +H minutes price for symbols from /home/karthik/new/symbols.txt
using SQLite DB /home/karthik/market.db.

Now uses full microstructure features (bid/ask, qty, VWAP, volumes, etc.)
for the ML algorithms.

Algorithms:
  1) Exponential Smoothing / Holt-Winters (price-only; statsmodels if available; else EWMA)
  2) Random Forest (rich features; sklearn; fallback LinearRegression)
  3) Gradient Boosting (rich features; sklearn; fallback Ridge)

Robust timestamp parsing, resampling, feature engineering, and iterative H-step
forecast with exogenous features carried-forward where needed.
"""

import argparse
import datetime as dt
import math
import os
import sqlite3
import sys
import warnings
from typing import Optional, Tuple, List, Dict

warnings.filterwarnings("ignore")

# --- Required deps ---
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("This script requires pandas and numpy. Install with: pip install pandas numpy", file=sys.stderr)
    raise

# Optional deps
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as SMExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ---------------- IO ----------------

def safe_read_symbols(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"[WARN] Symbols file not found at {path}. Using default: NSE:DRREDDY-EQ")
        return ["NSE:DRREDDY-EQ"]
    syms = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                syms.append(s)
    if not syms:
        syms = ["NSE:DRREDDY-EQ"]
    return syms


def fetch_ticks(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    sql = """
    SELECT
        symbol,
        ts,
        ltp,
        bid,
        ask,
        bid_qty,
        ask_qty,
        open,
        high,
        low,
        prev_close,
        vwap,
        vol_traded_today,
        last_traded_qty,
        raw_json,
        last_traded_time,
        exch_feed_time,
        bid_size,
        ask_size,
        bid_price,
        ask_price,
        tot_buy_qty,
        tot_sell_qty,
        avg_trade_price,
        low_price,
        high_price,
        lower_ckt,
        upper_ckt,
        open_price,
        prev_close_price,
        ch,
        chp,
        type
    FROM ticks
    WHERE symbol = ?
    ORDER BY ts ASC
    """
    try:
        df = pd.read_sql_query(sql, conn, params=(symbol,))
    except Exception as e:
        print(f"[ERROR] SQL read failed for {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()
    return df


# ---------------- Time Parsing ----------------

def _parse_any_datetime_col(series: pd.Series) -> Optional[pd.DatetimeIndex]:
    if series is None:
        return None
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().sum() == 0:
            return None
        median_val = float(vals.dropna().median())
        try:
            if median_val >= 1e11:  # ms
                idx = pd.to_datetime(vals, unit="ms", errors="coerce", utc=False)
            else:
                idx = pd.to_datetime(vals, unit="s", errors="coerce", utc=False)
        except Exception:
            idx = None
    else:
        try:
            idx = pd.to_datetime(s, errors="coerce", utc=False)
        except Exception:
            idx = None

    if idx is None or idx.isna().all():
        return None
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
    try:
        idx = pd.DatetimeIndex(idx)
    except Exception:
        return None
    return idx


def coerce_datetime_index(df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
    for col in ["ts", "exch_feed_time", "last_traded_time"]:
        if col in df.columns:
            idx = _parse_any_datetime_col(df[col])
            if idx is not None and idx.notna().sum() > 0:
                return idx
    return None


# ---------------- Resampling & Features ----------------

NUMERIC_COLS_PRIORITIES = [
    # Primary
    "ltp", "bid", "ask", "vwap",
    "bid_qty", "ask_qty", "bid_size", "ask_size",
    "bid_price", "ask_price",
    "tot_buy_qty", "tot_sell_qty",
    "last_traded_qty", "vol_traded_today",
    "avg_trade_price",
    # OHLC-ish
    "open", "high", "low",
    "open_price", "prev_close_price", "prev_close",
    "low_price", "high_price",
    # Misc deltas/percents
    "ch", "chp",
    # Circuit limits
    "lower_ckt", "upper_ckt",
]

def to_minutely_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample all numeric columns to 1-minute using 'last' aggregation.
    Forward-fill short gaps (limit=5) and drop leading NA.
    Returns a DataFrame indexed by minute with a consistent numeric schema.
    """
    if df.empty:
        return pd.DataFrame()

    idx = coerce_datetime_index(df)
    if idx is None:
        return pd.DataFrame()

    # Keep only numeric/castable cols of interest
    cols_present = [c for c in NUMERIC_COLS_PRIORITIES if c in df.columns]
    if "ltp" not in cols_present:
        # ltp is mandatory target
        return pd.DataFrame()

    work = df[cols_present].apply(pd.to_numeric, errors="coerce")
    work.index = idx
    work = work.sort_index()

    # drop all-NA rows to help resample
    work = work.dropna(how="all")
    if work.empty:
        return pd.DataFrame()

    # 1-minute resample: last known in the minute
    res = work.resample("1T").last()

    # forward-fill up to 5 mins; keep initial NaNs to drop later
    res = res.ffill(limit=5).dropna(subset=["ltp"])
    # Ensure strictly increasing index & dedup
    res = res[~res.index.duplicated(keep="last")].sort_index()
    return res


def add_engineered_features(dfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add microstructure & statistical features to the minutely frame.
    Assumes dfm has at least ltp; other cols optional.
    """
    X = dfm.copy()

    # --- Core derived book features ---
    def safe_div(a, b):
        return np.where(np.abs(b) > 1e-12, a / b, 0.0)

    # Mid & Spreads
    if {"bid", "ask"}.issubset(X.columns):
        X["mid"] = (X["bid"] + X["ask"]) / 2.0
        X["spread"] = X["ask"] - X["bid"]
        X["rel_spread"] = safe_div(X["spread"], X["mid"])
    elif {"bid_price", "ask_price"}.issubset(X.columns):
        X["mid"] = (X["bid_price"] + X["ask_price"]) / 2.0
        X["spread"] = X["ask_price"] - X["bid_price"]
        X["rel_spread"] = safe_div(X["spread"], X["mid"])
    else:
        X["mid"] = X["ltp"]
        X["spread"] = 0.0
        X["rel_spread"] = 0.0

    # Depth / Imbalance
    bqty = None
    aqty = None
    if "bid_qty" in X.columns:
        bqty = X["bid_qty"]
    elif "tot_buy_qty" in X.columns:
        bqty = X["tot_buy_qty"]
    if "ask_qty" in X.columns:
        aqty = X["ask_qty"]
    elif "tot_sell_qty" in X.columns:
        aqty = X["tot_sell_qty"]

    if bqty is not None and aqty is not None:
        X["depth_total"] = (bqty.fillna(0) + aqty.fillna(0))
        X["imbalance"] = safe_div(bqty.fillna(0) - aqty.fillna(0), X["depth_total"])
    else:
        X["depth_total"] = 0.0
        X["imbalance"] = 0.0

    # VWAP deviation
    if "vwap" in X.columns:
        X["vwap_dev"] = X["ltp"] - X["vwap"]
        X["vwap_rel_dev"] = safe_div(X["ltp"] - X["vwap"], X["vwap"].replace(0, np.nan)).fillna(0)
    else:
        X["vwap_dev"] = 0.0
        X["vwap_rel_dev"] = 0.0

    # Price returns & volatility
    X["ret_1"] = X["ltp"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    for w in [3, 5, 10, 20]:
        X[f"ret_{w}"] = X["ltp"].pct_change(w).replace([np.inf, -np.inf], 0).fillna(0)
        X[f"roll_mean_{w}"] = X["ltp"].rolling(w).mean()
        X[f"roll_std_{w}"] = X["ltp"].rolling(w).std()

    # Volume/qty dynamics
    if "vol_traded_today" in X.columns:
        # Minute volume (diff of cumulative)
        X["vol_minute"] = X["vol_traded_today"].diff().clip(lower=0).fillna(0)
        for w in [5, 10, 20]:
            X[f"vol_roll_sum_{w}"] = X["vol_minute"].rolling(w).sum()
            X[f"vol_roll_mean_{w}"] = X["vol_minute"].rolling(w).mean()
    else:
        X["vol_minute"] = 0.0
        for w in [5, 10, 20]:
            X[f"vol_roll_sum_{w}"] = 0.0
            X[f"vol_roll_mean_{w}"] = 0.0

    if "last_traded_qty" in X.columns:
        X["ltq_roll_mean_5"] = X["last_traded_qty"].rolling(5).mean()
        X["ltq_roll_std_5"] = X["last_traded_qty"].rolling(5).std()
    else:
        X["ltq_roll_mean_5"] = 0.0
        X["ltq_roll_std_5"] = 0.0

    # Deviation from mid
    X["mid_dev"] = X["ltp"] - X["mid"]
    X["mid_rel_dev"] = safe_div(X["ltp"] - X["mid"], X["mid"].replace(0, np.nan)).fillna(0)

    # Calendar features
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.DatetimeIndex(X.index)
    X["minute"] = X.index.minute
    X["hour"] = X.index.hour
    X["dow"] = X.index.dayofweek

    # Lags for price and key exogenous vars
    lag_vars = ["ltp", "mid", "spread", "rel_spread", "imbalance", "vwap_dev", "vol_minute"]
    for L in [1, 2, 3, 5, 8, 13, 21]:
        for v in lag_vars:
            if v in X.columns:
                X[f"{v}_lag_{L}"] = X[v].shift(L)

    # Targets for supervised learning (1-step ahead)
    X["target"] = X["ltp"].shift(-1)

    # Basic NA handling
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X


# ---------------- ML Helpers ----------------

def _fit_model_with_pipeline(X: pd.DataFrame, y: pd.Series, model_type: str):
    """
    Standardize numeric features for tree/linear models (trees don't need it but harmless).
    Provide robust fallbacks if sklearn is missing.
    """
    if not _HAS_SKLEARN:
        # numpy-based linear fallback
        class _NPLinear:
            def __init__(self):
                self.coef_ = None
                self.intercept_ = 0.0
                self.cols_ = None
            def fit(self, X, y):
                Xv = X.values.astype(float)
                yv = y.values.astype(float)
                X1 = np.hstack([np.ones((Xv.shape[0], 1)), Xv])
                beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
                self.intercept_ = float(beta[0])
                self.coef_ = beta[1:]
                self.cols_ = X.columns
                return self
            def predict(self, X):
                X = X[self.cols_]
                Xv = X.values.astype(float)
                X1 = np.hstack([np.ones((Xv.shape[0], 1)), Xv])
                return X1 @ np.hstack([[self.intercept_], self.coef_])
        return _NPLinear().fit(X, y)

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=400, min_samples_split=4, n_jobs=-1, random_state=42
        )
    elif model_type == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, random_state=42
        )
    elif model_type == "lin":
        model = LinearRegression()
    else:
        model = Ridge(alpha=1.0)

    # For robustness, scale inputs for linear models; for trees it's okay too.
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model),
    ])
    return pipe.fit(X, y)


def _latest_feature_row(feature_df: pd.DataFrame) -> pd.Series:
    return feature_df.iloc[-1]


def _advance_one_minute_state(raw_minutely: pd.DataFrame,
                              y_series: pd.Series,
                              current_time: pd.Timestamp) -> Dict[str, float]:
    """
    Build the exogenous, minute-ahead state for the next step by:
      - carrying forward last known exogenous values (bid/ask/qty/vwap/etc.)
      - recomputing derived features that depend on price (returns, deviations)
    Returns a dict of the values needed to rebuild the next feature row.
    """
    last_row = raw_minutely.iloc[-1].copy()

    # Carry-forward exogenous book stats
    state = last_row.to_dict()

    # Update values that derive from price using y_series (which includes the new prediction outside)
    # Compute new returns will be handled by feature builder when we rebuild the frame.
    return state


def build_features_for_training(minutely_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feat_df = add_engineered_features(minutely_df)
    X = feat_df.drop(columns=["target"])
    y = feat_df["target"]
    return X, y


def rebuild_feature_frame_for_forecast(minutely_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute engineered features from the current minutely_df state.
    """
    return add_engineered_features(minutely_df)


def iterative_forecast_exog(model, minutely_df: pd.DataFrame, steps: int) -> float:
    """
    Iteratively forecast H steps ahead using rich features.
    - We add one minute at a time.
    - For exogenous variables not known in the future, we carry-forward last values.
    - Recompute engineered features at each step, then use the last row to predict 1-step ahead.
    """
    if minutely_df.empty or "ltp" not in minutely_df.columns:
        return np.nan

    df_state = minutely_df.copy()
    for _ in range(steps):
        # Build features from current state to get the row at 'now' (for predicting next min)
        feat_df = rebuild_feature_frame_for_forecast(df_state)
        if feat_df.empty:
            return float(df_state["ltp"].iloc[-1])

        X_now = feat_df.drop(columns=["target"])
        # Need the most recent row’s features
        x_last = X_now.iloc[[-1]]
        try:
            y_hat = float(model.predict(x_last)[0])
        except Exception:
            y_hat = float(df_state["ltp"].iloc[-1])

        # Advance one minute: append new row with carried-forward exog + predicted ltp
        next_idx = df_state.index[-1] + pd.Timedelta(minutes=1)
        new_row = df_state.iloc[-1].copy()
        # carry-forward all exogenous; overwrite ltp with prediction
        new_row["ltp"] = y_hat
        # vwap/vol_traded_today typically unknown; keep last (neutral assumption)
        df_state.loc[next_idx] = new_row.values

    return float(df_state["ltp"].iloc[-1])


# ---------------- Algorithms ----------------

def algo1_expsmooth(series: pd.Series, horizon_minutes: int) -> float:
    if len(series) < 10:
        return float(series.iloc[-1])

    if _HAS_STATSMODELS:
        try:
            model = SMExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            fc = fit.forecast(horizon_minutes)
            return float(fc.iloc[-1])
        except Exception:
            pass

    # EWMA fallback
    s = series.dropna()
    if len(s) < 5:
        return float(s.iloc[-1])
    # simple holdout to choose alpha
    n_test = min(30, max(5, len(s)//5))
    train, test = s.iloc[:-n_test], s.iloc[-n_test:]
    best_alpha, best_mae = None, float("inf")
    for alpha in [i/20 for i in range(1, 20)]:
        ewma = train.ewm(alpha=alpha, adjust=False).mean()
        last = ewma.iloc[-1]
        preds = np.full(shape=len(test), fill_value=last, dtype=float)
        mae = float(np.mean(np.abs(preds - test.values)))
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha
    ewma_full = s.ewm(alpha=best_alpha, adjust=False).mean()
    return float(ewma_full.iloc[-1])


def algo2_random_forest(minutely_df: pd.DataFrame, horizon_minutes: int) -> float:
    X, y = build_features_for_training(minutely_df)
    if X.empty:
        return float(minutely_df["ltp"].iloc[-1])
    model = _fit_model_with_pipeline(X, y, model_type="rf" if _HAS_SKLEARN else "lin")
    return iterative_forecast_exog(model, minutely_df, steps=horizon_minutes)


def algo3_gradient_boost(minutely_df: pd.DataFrame, horizon_minutes: int) -> float:
    X, y = build_features_for_training(minutely_df)
    if X.empty:
        return float(minutely_df["ltp"].iloc[-1])
    model = _fit_model_with_pipeline(X, y, model_type="gbr" if _HAS_SKLEARN else "ridge")
    return iterative_forecast_exog(model, minutely_df, steps=horizon_minutes)


# ---------------- Orchestration ----------------

def predict_for_symbol(conn: sqlite3.Connection, symbol: str, horizon_minutes: int) -> Dict[str, Optional[float]]:
    raw = fetch_ticks(conn, symbol)
    if raw.empty:
        print(f"[WARN] No data for {symbol}")
        return {"algo1_expsmooth": None, "algo2_random_forest": None, "algo3_gradient_boost": None}

    minutely = to_minutely_frame(raw)
    if minutely.empty:
        print(f"[WARN] No valid timestamps or prices for {symbol} (cannot form 1-minute series).")
        return {"algo1_expsmooth": None, "algo2_random_forest": None, "algo3_gradient_boost": None}

    # Soft outlier clamp on price
    q_low, q_high = minutely["ltp"].quantile(0.005), minutely["ltp"].quantile(0.995)
    minutely["ltp"] = minutely["ltp"].clip(lower=q_low, upper=q_high)

    # Algo 1 uses price-only series
    series = minutely["ltp"]

    try:
        p1 = algo1_expsmooth(series, horizon_minutes)
    except Exception as e:
        print(f"[WARN] Algo1 failed for {symbol}: {e}")
        p1 = float(series.iloc[-1])

    try:
        p2 = algo2_random_forest(minutely, horizon_minutes)
    except Exception as e:
        print(f"[WARN] Algo2 failed for {symbol}: {e}")
        p2 = float(series.iloc[-1])

    try:
        p3 = algo3_gradient_boost(minutely, horizon_minutes)
    except Exception as e:
        print(f"[WARN] Algo3 failed for {symbol}: {e}")
        p3 = float(series.iloc[-1])

    return {"algo1_expsmooth": p1, "algo2_random_forest": p2, "algo3_gradient_boost": p3}


def main():
    parser = argparse.ArgumentParser(description="Predict H-minute ahead prices with rich microstructure features.")
    parser.add_argument("--db-path", default="/home/karthik/market.db", help="Path to SQLite DB file")
    parser.add_argument("--symbols-path", default="/home/karthik/new/symbols.txt", help="Path to symbols.txt (one symbol per line)")
    parser.add_argument("--horizon-minutes", type=int, default=15, help="Forecast horizon in minutes (default: 15)")
    # Optional: restrict recent window (e.g., last N minutes) if desired
    parser.add_argument("--train-window-minutes", type=int, default=0,
                        help="If >0, restrict training to the last N minutes (freshness control).")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"[ERROR] Database not found at {args.db_path}", file=sys.stderr)
        sys.exit(1)

    try:
        conn = sqlite3.connect(args.db_path)
    except Exception as e:
        print(f"[ERROR] Could not open DB: {e}", file=sys.stderr)
        sys.exit(1)

    symbols = safe_read_symbols(args.symbols_path)
    if not symbols:
        print("[ERROR] No symbols found to process.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Forecast horizon: +{args.horizon_minutes} minutes ===")
    print(f"Run time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for sym in symbols:
        try:
            # Predict
            res_full = fetch_ticks(conn, sym)
            if res_full.empty:
                print(f"Symbol: {sym}\n  No prediction (no data)\n")
                continue

            minutely = to_minutely_frame(res_full)
            if minutely.empty:
                print(f"Symbol: {sym}\n  No prediction (invalid timestamps)\n")
                continue

            # Optional freshness restriction
            if args.train_window_minutes and args.train_window_minutes > 0:
                cutoff = minutely.index.max() - pd.Timedelta(minutes=args.train_window_minutes)
                minutely = minutely[minutely.index >= cutoff]
                if len(minutely) < 50:
                    print(f"[WARN] {sym}: very little data after window restriction; expanding to all data.")
                    minutely = to_minutely_frame(res_full)

            # Produce predictions
            series = minutely["ltp"]
            q_low, q_high = series.quantile(0.005), series.quantile(0.995)
            series = series.clip(lower=q_low, upper=q_high)
            minutely["ltp"] = series

            try:
                p1 = algo1_expsmooth(series, args.horizon_minutes)
            except Exception as e:
                print(f"[WARN] Algo1 failed for {sym}: {e}")
                p1 = float(series.iloc[-1])

            try:
                p2 = algo2_random_forest(minutely, args.horizon_minutes)
            except Exception as e:
                print(f"[WARN] Algo2 failed for {sym}: {e}")
                p2 = float(series.iloc[-1])

            try:
                p3 = algo3_gradient_boost(minutely, args.horizon_minutes)
            except Exception as e:
                print(f"[WARN] Algo3 failed for {sym}: {e}")
                p3 = float(series.iloc[-1])

            def fmt(v):
                return "NA" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{float(v):.4f}"

            print(f"Symbol: {sym}")
            print(f"  Algo1 (ExpSmooth/Holt-Winters): {fmt(p1)}")
            print(f"  Algo2 (RandomForest+Microfeat): {fmt(p2)}")
            print(f"  Algo3 (GradBoost+Microfeat)   : {fmt(p3)}\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"[ERROR] Failed for {sym}: {e}", file=sys.stderr)

    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
