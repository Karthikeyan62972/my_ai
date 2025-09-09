#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
intraday_pro.py  (optimized, same logic & prints)

- Keep logic & user-facing prints identical
- Faster SQLite reads via safe PRAGMAs
- Minor pandas efficiencies (no behavioral change)
- Backtest warm-up bugfix: compare +H prediction to +H future price
"""

import argparse, os, sys, math, sqlite3, warnings, datetime as dt, json, pickle
from typing import Optional, List, Tuple, Dict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

# Optional ML deps
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import SGDRegressor, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Parallel runtime
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
import contextlib

# ===================== Paths & Defaults =====================
DEFAULT_DB = "/home/karthik/market.db"
DEFAULT_SYMBOLS = "/home/karthik/new/symbols.txt"
DEFAULT_CHECKPOINT_DIR = "/home/karthik/new/checkpoints"
DEFAULT_CONFIG_FILE = "/home/karthik/new/config.json"
IST_TZ = "Asia/Kolkata"

# ===================== SQLite tuning (safe) =====================
def _tune_sqlite_connection(conn: sqlite3.Connection) -> None:
    """
    Best-effort, read-friendly PRAGMAs. All wrapped in try/except so they never
    change behavior or crash in read-only environments.
    """
    try:
        # read-only friendly toggles
        conn.execute("PRAGMA query_only=ON;")
    except Exception:
        pass
    try:
        # store temp in RAM and enlarge cache (negative = KiB units)
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-200000;")  # ~200MB cache
    except Exception:
        pass
    try:
        # enable memory mapping if supported (set to ~512MB)
        conn.execute("PRAGMA mmap_size=536870912;")
    except Exception:
        pass
    # These may no-op under read-only; harmless if ignored
    for p in ("PRAGMA synchronous=OFF;", "PRAGMA journal_mode=OFF;"):
        try:
            conn.execute(p)
        except Exception:
            pass

# ===================== IO Helpers =====================
def read_symbols(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"[WARN] Symbols file not found at {path}. Using default NSE:DRREDDY-EQ")
        return ["NSE:DRREDDY-EQ"]
    with open(path, "r") as fh:
        syms = [ln.strip() for ln in fh if ln.strip()]
    return syms or ["NSE:DRREDDY-EQ"]

def fetch_ticks(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    sql = """
    SELECT symbol, ts, ltp, bid, ask, bid_qty, ask_qty, vwap, vol_traded_today,
           last_traded_qty, exch_feed_time, bid_size, ask_size, bid_price, ask_price,
           tot_buy_qty, tot_sell_qty, avg_trade_price, ch, chp, last_traded_time
    FROM ticks
    WHERE symbol = ?
    ORDER BY ts DESC
    """
    try:
        t0 = time.time()
        # PRAGMAs (no-ops if unsupported)
        _tune_sqlite_connection(conn)
        df = pd.read_sql_query(sql, conn, params=(symbol,))
        dt_sec = time.time() - t0
        print(f"[INFO] Query for {symbol} returned {len(df)} rows in {dt_sec:.2f} sec")
        return df
    except Exception as e:
        print(f"[ERROR] SQL read failed for {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()

# ===================== Time Parsing (Hardened) =====================
def _parse_any_datetime_col(series: pd.Series) -> Optional[pd.DatetimeIndex]:
    """
    Convert a column to a tz-naive DatetimeIndex, robust to:
      - numeric epoch in ns / µs / ms / s
      - strings with/without timezone; 'IST' -> '+05:30'
      - mixed types; stray bad values
    """
    if series is None:
        return None

    s = series

    # Normalize potential string artifacts
    if s.dtype == object:
        s = s.astype(str).str.strip()
        s = s.str.replace(r"\bIST\b", "+05:30", regex=True)
        s = s.replace({"": np.nan})

    # Try numeric epochs (also catches numeric-looking strings)
    s_num = pd.to_numeric(s, errors="coerce")
    use_num = s_num.notna().sum()
    if use_num > 0 and use_num >= (0.5 * len(s)):
        med = float(np.nanmedian(s_num.values))
        try:
            if med >= 1e16:     # ns
                idx = pd.to_datetime(s_num, unit="ns", errors="coerce", utc=True)
            elif med >= 1e13:   # µs
                idx = pd.to_datetime(s_num, unit="us", errors="coerce", utc=True)
            elif med >= 1e10:   # ms
                idx = pd.to_datetime(s_num, unit="ms", errors="coerce", utc=True)
            else:               # s
                idx = pd.to_datetime(s_num, unit="s", errors="coerce", utc=True)
        except Exception:
            idx = None
    else:
        try:
            idx = pd.to_datetime(s, errors="coerce", utc=True)
        except Exception:
            idx = None

    if idx is None or idx.isna().all():
        return None

    # drop timezone (back to naive UTC)
    try:
        idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

    try:
        return pd.DatetimeIndex(idx)
    except Exception:
        return None

def coerce_index(df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
    """Try, in order: ts -> exch_feed_time -> last_traded_time."""
    for col in ("ts", "exch_feed_time", "last_traded_time"):
        if col in df.columns:
            idx = _parse_any_datetime_col(df[col])
            if idx is not None and idx.notna().sum() > 0:
                return pd.DatetimeIndex(idx)
    return None

# ===================== Resample & Market Hours =====================
NUMERIC_COLS = [
    "ltp","bid","ask","bid_qty","ask_qty","vwap","vol_traded_today","last_traded_qty",
    "bid_size","ask_size","bid_price","ask_price","tot_buy_qty","tot_sell_qty","avg_trade_price","ch","chp"
]

def to_minutely(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 1-minute last, ffill short gaps; keep only NSE mkt hours 09:15–15:30 IST."""
    if df.empty:
        return pd.DataFrame()

    idx = coerce_index(df)
    if idx is None:
        samples = []
        for c in ("ts", "exch_feed_time", "last_traded_time"):
            if c in df.columns:
                samples = df[c].dropna().astype(str).head(3).tolist()
                break
        print(f"[WARN] Could not parse timestamps (samples: {samples})")
        return pd.DataFrame()

    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if "ltp" not in cols:
        print("[WARN] Missing 'ltp' column after selection.")
        return pd.DataFrame()

    df_num = df[cols].apply(pd.to_numeric, errors="coerce")
    df_num.index = pd.DatetimeIndex(idx)

    df_num = df_num.sort_index().dropna(how="all")
    if df_num.empty:
        return pd.DataFrame()

    try:
        m1 = df_num.resample("1T").last()
    except Exception as e:
        print(f"[WARN] Resample failed: {e}")
        return pd.DataFrame()

    m1 = m1.ffill(limit=5).dropna(subset=["ltp"])

    # Convert to IST and filter trading hours
    try:
        # Convert naive UTC -> aware IST -> back to naive for downstream
        m1_aware = pd.to_datetime(m1.index, utc=True).tz_convert(IST_TZ)
        m1_idx = pd.DatetimeIndex(m1_aware.tz_localize(None))
        m1.index = m1_idx
        m1 = m1[(m1.index.time >= dt.time(9, 15)) & (m1.index.time <= dt.time(15, 30))]
    except Exception:
        pass

    return m1[~m1.index.duplicated(keep="last")].sort_index()

# ===================== Features =====================
def _safe_div(a, b):
    b = np.where(np.abs(b) > 1e-12, b, np.nan)
    return a / b

def build_features(minutely: pd.DataFrame, horizon_min: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build feature matrix and targets:
      - Regression target: y_reg = log(ltp_{t+H}) - log(ltp_t)
      - Classification target: y_cls = 1 if y_reg > 0 else 0
      - Also return last_price series aligned with X
    """
    X = minutely.copy()

    # Mid/spread
    if {"bid","ask"}.issubset(X.columns):
        X["mid"] = (X["bid"] + X["ask"]) / 2.0
        X["spread"] = X["ask"] - X["bid"]
        X["rel_spread"] = pd.Series(_safe_div(X["spread"].values, X["mid"].values), index=X.index).fillna(0)
    elif {"bid_price","ask_price"}.issubset(X.columns):
        X["mid"] = (X["bid_price"] + X["ask_price"]) / 2.0
        X["spread"] = X["ask_price"] - X["bid_price"]
        X["rel_spread"] = pd.Series(_safe_div(X["spread"].values, X["mid"].values), index=X.index).fillna(0)
    else:
        X["mid"], X["spread"], X["rel_spread"] = X["ltp"], 0.0, 0.0

    # Depth/imbalance
    bq = X["bid_qty"] if "bid_qty" in X.columns else X["tot_buy_qty"] if "tot_buy_qty" in X.columns else None
    aq = X["ask_qty"] if "ask_qty" in X.columns else X["tot_sell_qty"] if "tot_sell_qty" in X.columns else None
    if bq is not None and aq is not None:
        bt, at = bq.fillna(0), aq.fillna(0)
        depth = (bt + at)
        X["depth_total"] = depth
        X["imbalance"] = pd.Series(_safe_div((bt - at).values, depth.replace(0, np.nan).values), index=X.index).fillna(0)
    else:
        X["depth_total"], X["imbalance"] = 0.0, 0.0

    # VWAP deviation
    if "vwap" in X.columns:
        X["vwap_dev"] = X["ltp"] - X["vwap"]
        X["vwap_rel_dev"] = pd.Series(_safe_div((X["ltp"] - X["vwap"]).values, X["vwap"].replace(0, np.nan).values), index=X.index).fillna(0)
    else:
        X["vwap_dev"], X["vwap_rel_dev"] = 0.0, 0.0

    # Returns & volatility
    X["ret_1"] = X["ltp"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    for w in (3, 5, 10, 20):
        X[f"ret_{w}"] = X["ltp"].pct_change(w).replace([np.inf, -np.inf], 0).fillna(0)
        X[f"roll_mean_{w}"] = X["ltp"].rolling(w).mean()
        X[f"roll_std_{w}"] = X["ltp"].rolling(w).std()

    # Volume/qty dynamics
    if "vol_traded_today" in X.columns:
        X["vol_minute"] = X["vol_traded_today"].diff().clip(lower=0).fillna(0)
        for w in (5, 10, 20):
            X[f"vol_roll_sum_{w}"] = X["vol_minute"].rolling(w).sum()
            X[f"vol_roll_mean_{w}"] = X["vol_minute"].rolling(w).mean()
    else:
        X["vol_minute"] = 0.0
        for w in (5, 10, 20):
            X[f"vol_roll_sum_{w}"] = 0.0
            X[f"vol_roll_mean_{w}"] = 0.0

    # Calendar
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.DatetimeIndex(X.index)
    X["minute"], X["hour"], X["dow"] = X.index.minute, X.index.hour, X.index.dayofweek

    # Lags
    lag_vars = ["ltp","mid","spread","rel_spread","imbalance","vwap_dev","vol_minute"]
    for L in (1, 2, 3, 5, 8, 13, 21):
        for v in lag_vars:
            if v in X.columns:
                X[f"{v}_lag_{L}"] = X[v].shift(L)

    # Targets
    y_reg = np.log(X["ltp"].shift(-horizon_min)) - np.log(X["ltp"])
    y_cls = (y_reg > 0).astype(int)
    last_price = X["ltp"].copy()

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y_reg = y_reg.loc[X.index]
    y_cls = y_cls.loc[X.index]
    last_price = last_price.loc[X.index]
    return X, y_reg, y_cls, last_price

# ===================== Drift Detection =====================
def compute_drift_score(series: pd.Series, recent_window: int = 120, ref_window: int = 600) -> float:
    """Heuristic drift score on 1-min returns: |mean_recent - mean_ref| / (std_ref + 1e-6)."""
    ret = series.pct_change().dropna()
    if len(ret) < recent_window + ref_window + 10:
        return 0.0
    recent = ret.iloc[-recent_window:]
    ref = ret.iloc[-(recent_window+ref_window):-recent_window]
    mean_diff = abs(recent.mean() - ref.mean())
    std_ref = ref.std() + 1e-6
    return float(mean_diff / std_ref)

# ===================== Models & Utilities =====================
def fit_gbr(X, y, params: dict = None):
    if not _HAS_SKLEARN: return None
    p = {"n_estimators": 600, "learning_rate": 0.02, "max_depth": 3, "subsample": 0.8, "random_state": 42}
    if params: p.update(params)
    return GradientBoostingRegressor(**p).fit(X, y)

def fit_rf(X, y, params: dict = None):
    if not _HAS_SKLEARN: return None
    p = {"n_estimators": 400, "min_samples_split": 4, "n_jobs": -1, "random_state": 42}
    if params: p.update(params)
    return RandomForestRegressor(**p).fit(X, y)

def fit_sgd(X, y, params: dict = None):
    if not _HAS_SKLEARN: return None
    base = {"loss": "huber", "alpha": 1e-4, "max_iter": 2000, "tol": 1e-3, "random_state": 42}
    if params: base.update(params)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("sgd", SGDRegressor(**base))
    ])
    return pipe.fit(X, y)

def fit_classifier(X, y_cls, params: dict = None):
    if not _HAS_SKLEARN: return None
    p = {"solver": "lbfgs", "max_iter": 200}
    if params: p.update(params)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logit", LogisticRegression(**p))
    ])
    return pipe.fit(X, y_cls)

def quantile_gbr(X, y, alpha: float, params: dict = None):
    if not _HAS_SKLEARN: return None
    p = {"loss": "quantile", "alpha": alpha, "n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "random_state": 42}
    if params: p.update(params)
    return GradientBoostingRegressor(**p).fit(X, y)

def ret_to_price(ret_pred: float, last_price: float) -> float:
    return float(np.exp(ret_pred) * last_price)

# ===================== Config & Checkpoints =====================
def load_config(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(path: str, cfg: dict):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

def ckpt_path(symbol: str, base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    safe = symbol.replace(":", "_").replace("/", "_")
    return os.path.join(base_dir, f"{safe}_sgd.pkl")

def save_checkpoint(symbol: str, model, base_dir: str):
    try:
        with open(ckpt_path(symbol, base_dir), "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"[WARN] Could not save checkpoint for {symbol}: {e}")

def load_checkpoint(symbol: str, base_dir: str):
    path = ckpt_path(symbol, base_dir)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# ===================== Risk Signal =====================
def risk_signal(pred_ret: float, recent_rets: pd.Series, k: float = 0.75) -> Tuple[str, float]:
    """
    If |pred_ret| > k * recent_vol, issue signal:
      - 'LONG' if pred_ret>0
      - 'SHORT' if pred_ret<0
    Returns (signal, threshold_used)
    """
    vol = recent_rets.rolling(20).std().iloc[-1] if len(recent_rets) >= 20 else recent_rets.std()
    vol = float(vol) if vol and not np.isnan(vol) else 0.0
    thresh = k * vol
    sig = "FLAT"
    if abs(pred_ret) > thresh:
        sig = "LONG" if pred_ret > 0 else "SHORT"
    return sig, thresh

# ===================== Core Prediction =====================
def run_predict_for_symbol(conn,
                           symbol: str,
                           horizon: int,
                           train_window: int,
                           cfg: dict,
                           cfg_path: str,
                           ckpt_dir: str,
                           fmt: str = "text"):
    raw = fetch_ticks(conn, symbol)
    if raw.empty:
        print(f"Symbol: {symbol}\n  No data.\n"); return

    m1 = to_minutely(raw)
    if m1.empty:
        print(f"Symbol: {symbol}\n  No valid intraday series.\n"); return

    # Restrict to recent window
    if train_window and train_window > 0:
        cutoff = m1.index.max() - pd.Timedelta(minutes=train_window)
        m1 = m1[m1.index >= cutoff]
    if len(m1) < 120:
        print(f"Symbol: {symbol}\n  Too little data in window.\n"); return

    # Drift detection → shrink window if needed
    drift = compute_drift_score(m1["ltp"])
    if drift > 1.5 and train_window > 300:
        cutoff = m1.index.max() - pd.Timedelta(minutes=max(300, train_window // 2))
        m1 = m1[m1.index >= cutoff]

    X, y_reg, y_cls, last_price_series = build_features(m1, horizon)
    if X.empty or y_reg.empty:
        print(f"Symbol: {symbol}\n  Not enough feature/target rows.\n"); return

    # ---- drop rows with NaN target; keep last row for inference only ----
    target_mask = y_reg.notna()
    X_fit = X.loc[target_mask]
    y_fit = y_reg.loc[target_mask]
    y_cls_fit = y_cls.loc[target_mask] if not y_cls.empty else y_cls

    # Require that we still have enough rows to train
    if len(X_fit) < max(40, horizon + 10):
        print(f"Symbol: {symbol}\n  Not enough clean train rows after target alignment.\n"); return

    # Last row (for prediction)
    x_last = X.iloc[[-1]]
    last_price = float(last_price_series.iloc[0])
    last_rets = m1["ltp"].pct_change().dropna()

    # Load tuned params from config if present
    sym_cfg   = cfg.get(symbol, {})
    gbr_params = sym_cfg.get("gbr_params", {})
    rf_params  = sym_cfg.get("rf_params", {})
    sgd_params = sym_cfg.get("sgd_params", {})
    clf_params = sym_cfg.get("clf_params", {})

    # Fit models on CLEAN data
    gbr = fit_gbr(X_fit, y_fit, gbr_params)
    rf  = fit_rf(X_fit, y_fit, rf_params)
    sgd = fit_sgd(X_fit, y_fit, sgd_params)
    clf = fit_classifier(X_fit, y_cls_fit, clf_params) if not y_cls_fit.empty else None

    # Quantile models use the same clean slice
    gbr_q20 = quantile_gbr(X_fit, y_fit, alpha=0.20)
    gbr_q80 = quantile_gbr(X_fit, y_fit, alpha=0.80)

    def safe_pred(model, x, fallback=0.0):
        try:
            return float(model.predict(x)[0])
        except Exception:
            return float(fallback)

    # Regression predictions (log-return → price)
    ret_gbr = safe_pred(gbr, x_last, 0.0) if gbr is not None else 0.0
    ret_rf  = safe_pred(rf,  x_last, 0.0) if rf  is not None else 0.0
    ret_sgd = safe_pred(sgd, x_last, 0.0) if sgd is not None else 0.0

    price_gbr = ret_to_price(ret_gbr, last_price)
    price_rf  = ret_to_price(ret_rf,  last_price)
    price_sgd = ret_to_price(ret_sgd, last_price)

    # Direction probability
    prob_up = None
    if clf is not None:
        try:
            prob_up = float(clf.predict_proba(x_last)[0, 1])
        except Exception:
            prob_up = None

    # Quantile price bands
    q20 = ret_to_price(safe_pred(gbr_q20, x_last, 0.0), last_price) if gbr_q20 is not None else None
    q80 = ret_to_price(safe_pred(gbr_q80, x_last, 0.0), last_price) if gbr_q80 is not None else None

    # Risk-aware signal (use GBR return)
    signal, threshold = risk_signal(ret_gbr, last_rets, k=0.75)

    # Feature importances → pretty strings
    feat_imp_gbr = getattr(gbr, "feature_importances_", None) if gbr is not None else None
    feat_imp_rf  = getattr(rf,  "feature_importances_", None) if rf  is not None else None

    def top_feats_str(imp) -> str:
        if imp is None: return None
        idx = np.argsort(imp)[::-1][:10]
        cols = list(X_fit.columns)
        return ", ".join([f"{cols[i]}:{imp[i]:.3f}" for i in idx])

    topg = top_feats_str(feat_imp_gbr)
    topr = top_feats_str(feat_imp_rf)

    # --- prints (unchanged) ---
    print(f"Symbol: {symbol}")
    print(f"  Last price: {last_price:.4f}")
    print(f"  Pred (+{horizon}m) price:")
    print(f"    Algo1 (GradientBoost): {price_gbr:.4f}  (ret={ret_gbr:+.5f})")
    print(f"    Algo2 (RandomForest) : {price_rf:.4f}   (ret={ret_rf:+.5f})")
    print(f"    Algo3 (OnlineSGD)    : {price_sgd:.4f}  (ret={ret_sgd:+.5f})")
    if prob_up is not None:
        print(f"  Direction prob (UP): {prob_up:.3f}")
    if q20 is not None and q80 is not None:
        print(f"  Quantile band (P20, P80): ({q20:.4f}, {q80:.4f})")
    print(f"  Risk signal (k=0.75): {signal}  | threshold(ret)≈{threshold:.5f}")
    if drift > 0:
        print(f"  Drift score: {drift:.3f}  ({'shrunk window' if drift > 1.5 else 'normal'})")
    if topg: print(f"  Top features (GBR): {topg}")
    if topr: print(f"  Top features (RF):  {topr}")
    print()

    if sgd is not None:
        save_checkpoint(symbol, sgd, ckpt_dir)

    if fmt == "json":
        payload = {
            "symbol": symbol,
            "ts": int(time.time()),
            "horizon_min": int(horizon),
            "last_price": float(last_price),
            "price_gbr": float(price_gbr),
            "price_rf":  float(price_rf),
            "price_sgd": float(price_sgd),
            "ret_gbr": float(ret_gbr),
            "ret_rf":  float(ret_rf),
            "ret_sgd": float(ret_sgd),
            "prob_up": None if prob_up is None else float(prob_up),
            "q20": None if q20 is None else float(q20),
            "q80": None if q80 is None else float(q80),
            "risk_signal": signal,
            "threshold_ret": float(threshold),
            "drift_score": float(drift),
            "top_feats_gbr": topg,
            "top_feats_rf": topr,
        }
        print(json.dumps(payload, separators=(",", ":")))

# ===================== Backtest (learn + tune) =====================
def run_backtest_for_symbol(conn, symbol: str, horizon: int, window_minutes: int,
                            cfg: dict, cfg_path: str, ckpt_dir: str):
    raw = fetch_ticks(conn, symbol)
    m1 = to_minutely(raw)
    if m1.empty or len(m1) < (window_minutes + horizon + 120):
        print(f"{symbol}: insufficient data.\n"); return

    eval_len = min(600, max(240, len(m1)//5))
    start_idx = len(m1) - eval_len
    idxs = m1.index

    actuals, preds_g, preds_r, preds_s, probs_up = [], [], [], [], []

    gbr_grid = [
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3},
        {"n_estimators": 600, "learning_rate": 0.02, "max_depth": 3},
    ]
    rf_grid = [
        {"n_estimators": 300, "min_samples_split": 4},
        {"n_estimators": 500, "min_samples_split": 4},
    ]

    if _HAS_SKLEARN:
        sgd_pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("sgd", SGDRegressor(loss="huber", alpha=1e-4, random_state=42, max_iter=1, warm_start=True))
        ])
        sgd_inited = False
    else:
        sgd_pipe, sgd_inited = None, False

    # Warm-up param pick
    warm_end = start_idx - 1
    warm_start = max(0, warm_end - max(300, window_minutes//2))
    warm_slice = m1.iloc[warm_start:warm_end]
    best_gbr, best_rf = gbr_grid[0], rf_grid[0]
    best_gbr_mae, best_rf_mae = float("inf"), float("inf")

    if len(warm_slice) > 200:
        Xw, yw, _, lpw = build_features(warm_slice, horizon)
        if not Xw.empty:
            x_last_w = Xw.iloc[[-1]]
            # Align true future price at +horizon from the last feature row
            try:
                # Find the index of the last feature row within warm_slice and add horizon minutes
                last_idx = Xw.index[-1]
                future_idx = last_idx + pd.Timedelta(minutes=horizon)
                if future_idx in warm_slice.index:
                    true_future = float(warm_slice.loc[future_idx, "ltp"])
                else:
                    true_future = float(warm_slice["ltp"].shift(-horizon).reindex(Xw.index).iloc[-1])
            except Exception:
                true_future = np.nan

            for p in gbr_grid:
                g = fit_gbr(Xw, yw, p)
                if g is None: continue
                pw = ret_to_price(float(g.predict(x_last_w)[0]), float(lpw.iloc[-1]))
                if np.isfinite(true_future):
                    mae = abs(pw - true_future)
                    if mae < best_gbr_mae: best_gbr_mae, best_gbr = mae, p
            for p in rf_grid:
                r = fit_rf(Xw, yw, p)
                if r is None: continue
                pw = ret_to_price(float(r.predict(x_last_w)[0]), float(lpw.iloc[-1]))
                if np.isfinite(true_future):
                    mae = abs(pw - true_future)
                    if mae < best_rf_mae: best_rf_mae, best_rf = mae, p

    # Walk-forward
    for i in range(start_idx, len(m1) - horizon):
        now_ts = idxs[i]
        cutoff = now_ts - pd.Timedelta(minutes=window_minutes)
        train_slice = m1[(m1.index > cutoff) & (m1.index <= now_ts)]
        if len(train_slice) < 120: continue

        X, y_reg, y_cls, last_price_series = build_features(train_slice, horizon)
        if X.empty: continue
        x_last = X.iloc[[-1]]
        last_price = float(last_price_series.iloc[-1])

        true_price = float(m1["ltp"].iloc[i + horizon])

        g = fit_gbr(X, y_reg, best_gbr)
        r = fit_rf(X, y_reg, best_rf)
        if not sgd_inited and _HAS_SKLEARN:
            sgd_pipe.fit(X, y_reg); sgd_inited = True
        elif _HAS_SKLEARN:
            sgd_pipe.partial_fit(X, y_reg)

        ret_g = float(g.predict(x_last)[0]) if g is not None else 0.0
        ret_r = float(r.predict(x_last)[0]) if r is not None else 0.0
        ret_s = float(sgd_pipe.predict(x_last)[0]) if sgd_inited else 0.0

        preds_g.append(ret_to_price(ret_g, last_price))
        preds_r.append(ret_to_price(ret_r, last_price))
        preds_s.append(ret_to_price(ret_s, last_price))
        actuals.append(true_price)

        prob = None
        if _HAS_SKLEARN:
            try:
                clf = fit_classifier(X, y_cls)
                prob = float(clf.predict_proba(x_last)[0,1])
            except Exception:
                prob = None
        probs_up.append(prob if prob is not None else np.nan)

    if not actuals:
        print(f"{symbol}: not enough evaluation points.\n"); return

    def _metrics(y_true, y_pred):
        a, p = np.asarray(y_true,float), np.asarray(y_pred,float)
        mae = float(np.mean(np.abs(a-p)))
        rmse = math.sqrt(float(np.mean((a-p)**2)))
        mape = float(np.mean(np.abs((p-a)/np.where(a!=0,a,np.nan))))*100.0
        return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

    m_g = _metrics(actuals, preds_g)
    m_r = _metrics(actuals, preds_r)
    m_s = _metrics(actuals, preds_s)

    def fmt(d): return ", ".join([f"{k}={v:.4f}" for k,v in d.items()])
    print(f"{symbol}")
    print(f"  Algo1 (GradientBoost): {fmt(m_g)}  | best_params={best_gbr}")
    print(f"  Algo2 (RandomForest) : {fmt(m_r)}  | best_params={best_rf}")
    print(f"  Algo3 (OnlineSGD)    : {fmt(m_s)}")
    if np.isfinite(np.nanmean(probs_up)):
        print(f"  Avg P(up): {np.nanmean(probs_up):.3f}")
    print()

    # Save tuned params to config
    cfg.setdefault(symbol, {})
    cfg[symbol]["gbr_params"] = best_gbr
    cfg[symbol]["rf_params"] = best_rf
    save_config(cfg_path, cfg)

# ===================== Parallel Workers =====================
def _predict_worker(sym: str, db_path: str, horizon: int, train_window: int,
                    cfg: dict, cfg_path: str, ckpt_dir: str) -> str:
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        _tune_sqlite_connection(conn)
    except Exception as e:
        return f"[ERROR] {sym}: DB open failed: {e}"
    buf = StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_predict_for_symbol(conn, sym, horizon, train_window, cfg, cfg_path, ckpt_dir)
    except Exception as e:
        return f"[ERROR] {sym}: {e}"
    finally:
        try: conn.close()
        except Exception: pass
    return buf.getvalue().rstrip("\n")

def _backtest_worker(sym: str, db_path: str, horizon: int, train_window: int,
                     cfg_path: str, ckpt_dir: str) -> Tuple[str, dict]:
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        _tune_sqlite_connection(conn)
    except Exception as e:
        return (f"[ERROR] {sym}: DB open failed: {e}", {})
    buf = StringIO()
    cfg_delta = {}
    original_save_config = save_config
    def _capture_save_config(path, cfg):
        nonlocal cfg_delta
        if sym in cfg:
            cfg_delta = {sym: cfg[sym]}
    try:
        with contextlib.redirect_stdout(buf):
            globals()['save_config'] = _capture_save_config
            run_backtest_for_symbol(conn, sym, horizon, train_window, {}, cfg_path, ckpt_dir)
    except Exception as e:
        return (f"[ERROR] {sym}: {e}", {})
    finally:
        globals()['save_config'] = original_save_config
        try: conn.close()
        except Exception: pass
    return (buf.getvalue().rstrip("\n"), cfg_delta)

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser("Intraday forecasting & backtesting (single-file, parallel)")
    ap.add_argument("--mode", choices=["predict","backtest"], default="predict",
                    help="predict (default) or backtest")
    ap.add_argument("--db-path", default=DEFAULT_DB)
    ap.add_argument("--symbols-path", default=DEFAULT_SYMBOLS)
    ap.add_argument("--horizon-minutes", type=int, default=15)
    ap.add_argument("--train-window-minutes", type=int, default=750, help="~2 trading days by default")
    ap.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (processes). 1 = sequential")
    ap.add_argument("--symbol", help="Run for a single symbol instead of reading from file")
    args = ap.parse_args()

    if not os.path.exists(args.db_path):
        print(f"[ERROR] DB not found at {args.db_path}", file=sys.stderr); sys.exit(1)

    # Shared connection for sequential path
    try:
        conn = sqlite3.connect(args.db_path)
        _tune_sqlite_connection(conn)
    except Exception as e:
        print(f"[ERROR] Could not open DB: {e}", file=sys.stderr); sys.exit(1)

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = read_symbols(args.symbols_path)
    cfg = load_config(args.config_file)
    print(f"\n=== Mode: {args.mode} | Horizon: +{args.horizon_minutes}m | Train window: {args.train_window_minutes}m ===")
    print(f"Run time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.mode == "predict":
        if args.workers <= 1:
            for sym in symbols:
                try:
                    run_predict_for_symbol(conn, sym, args.horizon_minutes, args.train_window_minutes,
                                           cfg, args.config_file, args.checkpoint_dir)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[ERROR] {sym}: {e}", file=sys.stderr)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_predict_worker, sym, args.db_path, args.horizon_minutes,
                                  args.train_window_minutes, cfg, args.config_file, args.checkpoint_dir): sym
                        for sym in symbols}
                # Print results as they complete
                for fut in as_completed(futs):
                    out = fut.result()
                    if out: print(out)
    else:  # backtest
        if args.workers <= 1:
            for sym in symbols:
                try:
                    run_backtest_for_symbol(conn, sym, args.horizon_minutes, args.train_window_minutes,
                                            cfg, args.config_file, args.checkpoint_dir)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[ERROR] {sym}: {e}", file=sys.stderr)
            # Config saved inside function
        else:
            merged_cfg = load_config(args.config_file)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_backtest_worker, sym, args.db_path, args.horizon_minutes,
                                  args.train_window_minutes, args.config_file, args.checkpoint_dir): sym
                        for sym in symbols}
                for fut in as_completed(futs):
                    out, delta = fut.result()
                    if out: print(out)
                    if delta:
                        merged_cfg.update(delta)
            save_config(args.config_file, merged_cfg)

    try: conn.close()
    except Exception: pass

if __name__ == "__main__":
    main()
