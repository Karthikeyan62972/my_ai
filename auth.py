#!/usr/bin/env python3
"""Headless FYERS v3 auth + token management."""
import base64, json, time, os, requests
from urllib.parse import parse_qs, urlencode, urlparse
import pyotp
from typing import Any, Dict, Optional

from settings import (
    FYERS_ID, TOTP_SECRET, PIN, APP_ID, APP_SECRET, REDIRECT_URI,
    VAGATOR_BASE, AUTH_BASES, DATA_BASE, TOKENS_PATH, REQ_TIMEOUT, SESSION,
    parse_app_id,
)

def b64(s: str) -> str: return base64.b64encode(s.encode()).decode()
def post_json(url, payload, timeout=REQ_TIMEOUT): return SESSION.post(url, json=payload, timeout=timeout)
def now_ts() -> int: return int(time.time())

def jwt_exp_epoch(jwt_token: str) -> Optional[int]:
    try:
        parts = jwt_token.split(".")
        if len(parts) != 3: return None
        import json as _json, base64 as _b64
        pad = "=" * (-len(parts[1]) % 4)
        payload = _b64.urlsafe_b64decode(parts[1] + pad)
        data = _json.loads(payload.decode())
        return int(data.get("exp")) if "exp" in data else None
    except Exception: return None

def app_id_hash(app_id: str, secret: str) -> str:
    import hashlib; return hashlib.sha256(f"{app_id}:{secret}".encode()).hexdigest()

def _try_get_auth_url(temp_token: str) -> Optional[str]:
    app_prefix, app_type = parse_app_id(APP_ID)
    SESSION.headers.update({
        "authorization": f"Bearer {temp_token}",
        "content-type": "application/json; charset=UTF-8",
        "accept": "application/json",
    })
    payload = {
        "fyers_id": FYERS_ID, "app_id": app_prefix, "redirect_uri": REDIRECT_URI,
        "appType": app_type, "code_challenge": "", "state": "state",
        "scope": "", "nonce": "", "response_type": "code", "create_cookie": True,
    }
    for base in AUTH_BASES:
        try:
            r = post_json(f"{base}/api/v3/token", payload)
            j = r.json() if "json" in r.headers.get("content-type","") else {}
            if j.get("Url"): return j["Url"]
        except requests.RequestException: pass
    params = {"client_id": APP_ID, "redirect_uri": REDIRECT_URI, "response_type": "code", "state": "state"}
    for base in AUTH_BASES:
        try:
            r = SESSION.get(f"{base}/api/v3/generate-authcode?" + urlencode(params), timeout=REQ_TIMEOUT, allow_redirects=False)
            loc = r.headers.get("Location")
            if loc: return loc
            j = r.json();  return j.get("Url") if j.get("Url") else None
        except Exception: pass
    return None

def get_auth_code_headless() -> str:
    r1 = post_json(f"{VAGATOR_BASE}/send_login_otp_v2", {"fy_id": b64(FYERS_ID), "app_id": "2"})
    r1.raise_for_status(); req_key = r1.json()["request_key"]
    for _ in range(2):
        totp = pyotp.TOTP(TOTP_SECRET).now()
        r2 = post_json(f"{VAGATOR_BASE}/verify_otp", {"request_key": req_key, "otp": totp})
        if r2.ok and r2.json().get("request_key"): req_key = r2.json()["request_key"]; break
        time.sleep(1)
    else: raise RuntimeError(f"TOTP verify failed: {r2.text}")
    r3 = post_json(f"{VAGATOR_BASE}/verify_pin_v2", {"request_key": req_key, "identity_type": "pin", "identifier": b64(PIN)})
    r3.raise_for_status(); temp_token = r3.json()["data"]["access_token"]
    backoff=1.0
    for _ in range(1,5):
        url = _try_get_auth_url(temp_token)
        if url:
            qs = parse_qs(urlparse(url).query); ac = (qs.get("auth_code") or [None])[0]
            if ac: return ac
        time.sleep(backoff); backoff = min(backoff*2,8.0)
    raise RuntimeError("Failed to obtain auth_code after retries.")

def exchange_auth_code(auth_code: str) -> Dict[str, Any]:
    from fyers_apiv3 import fyersModel
    session = fyersModel.SessionModel(client_id=APP_ID, secret_key=APP_SECRET,
                                      redirect_uri=REDIRECT_URI, response_type="code", grant_type="authorization_code")
    session.set_token(auth_code)
    tokens = session.generate_token()
    if "access_token" not in tokens: raise RuntimeError(f"Token exchange failed: {tokens}")
    return tokens

def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    url = f"{AUTH_BASES[1]}/api/v3/validate-refresh-token"
    payload = {"grant_type": "refresh_token","appIdHash": app_id_hash(APP_ID, APP_SECRET),"refresh_token": refresh_token,"pin": PIN}
    r = post_json(url,payload)
    if r.status_code!=200: raise RuntimeError(f"Refresh failed [{r.status_code}]: {r.text}")
    return r.json()

def access_token_valid(access_token: str, skew_secs: int = 120) -> bool:
    if not access_token: return False
    exp = jwt_exp_epoch(access_token)
    if exp: return (now_ts()+skew_secs)<exp
    try:
        headers={"Authorization":f"{APP_ID}:{access_token}"}
        r=SESSION.get(f"{DATA_BASE}/data/quotes?symbols=NSE:SBIN-EQ",headers=headers,timeout=5)
        return r.status_code==200
    except Exception: return False

def load_tokens()->Dict[str,Any]:
    if os.path.exists(TOKENS_PATH):
        try: return json.load(open(TOKENS_PATH))
        except Exception: pass
    return {}

def save_tokens(d:Dict[str,Any])->None:
    tmp=TOKENS_PATH+".tmp"; json.dump(d,open(tmp,"w"),indent=2,sort_keys=True); os.replace(tmp,TOKENS_PATH)

def ensure_tokens()->Dict[str,Any]:
    t=load_tokens(); at=t.get("access_token",""); rt=t.get("refresh_token","")
    if at and access_token_valid(at): return t
    if rt:
        try:
            newt=refresh_access_token(rt)
            if "access_token" not in newt and "accessToken" in newt: newt["access_token"]=newt["accessToken"]
            if "refresh_token" not in newt and "refreshToken" in newt: newt["refresh_token"]=newt["refreshToken"]
            if "refresh_token" not in newt: newt["refresh_token"]=rt
            save_tokens(newt); return newt
        except Exception as e: print(f"[WARN] Refresh failed, full login: {e}")
    code=get_auth_code_headless(); tokens=exchange_auth_code(code); save_tokens(tokens); return tokens
