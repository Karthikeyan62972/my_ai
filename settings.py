#!/usr/bin/env python3
"""Global settings, symbols, and common session."""
import os, re, requests
from typing import List

# ========= Credentials via ENV =========
# ================== YOUR CREDS (unchanged) ==================
FYERS_ID      = 'XG06506'
TOTP_SECRET   = '37SDULRJPPZLRMPSNJEFENX3NTQM7JEA'
PIN           = '4321'
APP_ID        = 'WVSXR99MI6-100'  # includes "-100" app type suffix
APP_SECRET    = 'ULX2XE1ITS'
REDIRECT_URI  = 'http://127.0.0.1:8080/ai_Trade'
# ===========================================================

# ========= Paths =========
DB_PATH      = os.environ.get("FYERS_DB_PATH", "/home/karthik/market.db")
TOKENS_PATH  = os.environ.get("FYERS_TOKENS_PATH", "tokens.json")
LOG_PATH     = os.environ.get("FYERS_LOG_PATH", "")

# ========= Market & Strategy =========
MARKET_TZ_OFFSET_MIN = 330  # IST
MARKET_OPEN  = (9, 15)
MARKET_CLOSE = (15, 30)

PREDICT_HORIZON_SECS = 60
FEATURE_WINDOWS_SECS = [5, 15, 30, 60, 120, 300]
MIN_TRAIN_SNAPSHOTS  = 300

ENTRY_THRESHOLD_PCT = 0.15
TAKE_PROFIT_PCT     = 0.30
STOP_LOSS_PCT       = 0.20
POSITION_QTY        = 1
ALLOW_SHORT         = False
DRY_RUN             = True
MAX_OPEN_POSITIONS  = 1

QUOTE_POLL_SECS = 5

# ========= APIs =========
VAGATOR_BASE = "https://api-t2.fyers.in/vagator/v2"
AUTH_BASES   = ["https://api.fyers.in", "https://api-t1.fyers.in", "https://myapi.fyers.in"]
DATA_BASE    = "https://api-t1.fyers.in"
REQ_TIMEOUT  = 20
WS_LITE_MODE = False  # keep False to receive richer fields

# ========= Symbols =========
DEFAULT_SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:SBIN-EQ",
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
]

def load_symbols() -> List[str]:
    env = os.environ.get("TRADE_SYMBOLS", "").strip()
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    path = os.environ.get("TRADE_SYMBOLS_FILE", "symbols.txt")
    if os.path.exists(path):
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            if lines:
                return lines
    return list(DEFAULT_SYMBOLS)

SYMBOLS = load_symbols()

# ========= Common HTTP session =========
SESSION = requests.Session()
SESSION.headers.update({"accept": "application/json", "user-agent": "Mozilla/5.0"})

# ========= Helpers =========
def parse_app_id(app_id_env: str):
    m = re.fullmatch(r"([A-Za-z0-9]+)(?:-([0-9]{2,3}))?", (app_id_env or "").strip())
    if not m:
        raise ValueError(f"APP_ID invalid: {app_id_env!r}")
    return m.group(1), (m.group(2) or "100")
