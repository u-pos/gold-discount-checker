#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, math, time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import urllib.parse

JST = timezone(timedelta(hours=9))
TD_KEY = os.environ.get("TWELVE_API_KEY", "").strip()

BASE_TD = "https://api.twelvedata.com"

def now_jst_str():
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

def http_get(url, params=None, timeout=15):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8")

def td_price(symbol):
    """
    Twelve Data: quote で最新値と時刻
    返り値: (price(float|None), iso_time(str|None))
    """
    if not TD_KEY:
        return None, None
    try:
        txt = http_get(f"{BASE_TD}/quote", {"symbol": symbol, "apikey": TD_KEY})
        j = json.loads(txt)
        p = float(j.get("price"))
        t = j.get("datetime")  # ISO
        return p, t
    except Exception:
        return None, None

def td_series(symbol, interval="5min", outputsize=120):
    """
    Twelve Data: time_series で分足取得（XAU/USD, USD/JPY 用）
    返り値: pandas.Series(index=datetime, values=float)
    """
    if not TD_KEY:
        return None
    try:
        txt = http_get(f"{BASE_TD}/time_series", {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "orderby": "asc",
            "apikey": TD_KEY
        })
        j = json.loads(txt)
        if "values" not in j:  # エラー時は "message" 等
            return None
        rows = []
        for v in j["values"]:
            dt = datetime.fromisoformat(v["datetime"].replace("Z","+00:00")).astimezone(JST)
            rows.append((dt, float(v["close"])))
        if not rows:
            return None
        idx = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        return pd.Series(vals, index=idx, name=symbol)
    except Exception:
        return None

def yf_last_price_smart(ticker: str):
    """
    yfinance で『できるだけ今に近い』価格を取得（主に 1540.T 用）
      1m->5m->15m->fast_info->1d
    """
    def hist_last(period, interval):
        try:
            h = yf.Ticker(ticker).history(period=period, interval=interval)
            if h is not None and len(h)>0:
                close = h["Close"].dropna()
                if len(close)>0:
                    ts = close.index[-1]
                    # indexにtzがあればJSTに
                    try:
                        tsj = ts.tz_convert(JST)
                    except Exception:
                        tsj = ts
                    return float(close.iloc[-1]), str(tsj)
        except Exception:
            pass
        return None, None

    p, ts = hist_last("1d","1m")
    if p is not None: return p, ts
    p, ts = hist_last("5d","5m")
    if p is not None: return p, ts
    p, ts = hist_last("60d","15m")
    if p is not None: return p, ts
    try:
        fi = yf.Ticker(ticker).fast_info or {}
        lp = fi.get("last_price")
        if lp and math.isfinite(lp):
            return float(lp), "fast_info"
    except Exception:
        pass
    p, ts = hist_last("5d","1d")
    if p is not None: return p, ts
    return None, None

def yf_series_close(ticker: str, period: str, interval: str):
    try:
        s = yf.Ticker(ticker).history(period=period, interval=interval)["Close"].dropna()
        return s
    except Exception:
        return None

def align_join(x: pd.Series, j: pd.Series, e: pd.Series, tail=None):
    if x is None or j is None or e is None:
        return None
    df = pd.concat([x.rename("xau"), j.rename("jpy"), e.rename("etf")], axis=1).dropna()
    if tail: df = df.tail(tail)
    df = df[np.isfinite(df).all(1)]
    return df if len(df)>0 else None

def estimate_k(df: pd.DataFrame):
    if df is None or len(df) < 5: return None
    X = df["xau"] * df["jpy"]
    Y = df["etf"]
    num = float((X*Y).sum()); den = float((X*X).sum())
    if den <= 0: return None
    return num/den

def main():
    # --- live prices from Twelve Data (gold & fx) ---
    # Twelve Data のシンボルは "XAU/USD", "USD/JPY"
    xauusd, t_xau = td_price("XAU/USD")
    usdjpy, t_jpy = td_price("USD/JPY")

    # フォールバック（キー未設定/失敗時）
    if xauusd is None or usdjpy is None:
        # 最低限、直近バーを取る
        x5 = td_series("XAU/USD","5min",5)
        j5 = td_series("USD/JPY","5min",5)
        if x5 is not None: xauusd = float(x5.dropna().iloc[-1]); t_xau = str(x5.dropna().index[-1])
        if j5 is not None: usdjpy = float(j5.dropna().iloc[-1]); t_jpy = str(j5.dropna().index[-1])

    # --- 1540.T from Yahoo ---
    price1540, t_etf = yf_last_price_smart("1540.T")

    # --- day mode (3営業日・日足) ---
    # 日足は yfinance（1540に合わせて東証のバーと整合を取りやすい）
    x_day = yf_series_close("XAUUSD=X","1mo","1d")
    j_day = yf_series_close("JPY=X","1mo","1d")
    e_day = yf_series_close("1540.T","1mo","1d")
    df_day = align_join(x_day, j_day, e_day, tail=10)
    k_day = theo_day = dev_day = None
    if df_day is not None:
        k_day = estimate_k(df_day.tail(3))

    # --- 5分足（XAU/USDとUSD/JPYは Twelve Data、1540はyfinance） ---
    x_5 = td_series("XAU/USD","5min",200)     # だいたい過去16時間分
    j_5 = td_series("USD/JPY","5min",200)
    e_5 = yf_series_close("1540.T","5d","5m") # 東証場中のみ更新が基本
    df_5 = align_join(x_5, j_5, e_5)
    k_5m = theo_5m = dev_5m = None
    if df_5 is not None:
        k_5m = estimate_k(df_5.tail(36))      # 直近約3時間

    # --- 15分足 ---
    x_15 = td_series("XAU/USD","15min",200)
    j_15 = td_series("USD/JPY","15min",200)
    e_15 = yf_series_close("1540.T","60d","15m")
    df_15 = align_join(x_15, j_15, e_15)
    k_15m = theo_15m = dev_15m = None
    if df_15 is not None:
        k_15m = estimate_k(df_15.tail(32))    # 直近約8時間

    # --- 理論値＆乖離計算（各モード） ---
    def calc(the_k):
        if the_k and xauusd and usdjpy and price1540:
            theo = xauusd * usdjpy * the_k
            return theo, (price1540/theo - 1.0)
        return None, None

    theo_day,  dev_day  = calc(k_day)
    theo_5m,   dev_5m   = calc(k_5m)
    theo_15m,  dev_15m  = calc(k_15m)

    out = {
        "time_jst": now_jst_str(),
        "provider": {"gold_fx": "TwelveData", "etf": "YahooFinance"},
        "xauusd": xauusd, "usdjpy": usdjpy, "price1540": price1540,
        "time_src": {"xau": t_xau, "jpy": t_jpy, "etf": t_etf},
        "k_day": k_day,   "theo_day": theo_day,   "dev_day": dev_day,
        "k_5m": k_5m,     "theo_5m":  theo_5m,    "dev_5m":  dev_5m,
        "k_15m": k_15m,   "theo_15m": theo_15m,   "dev_15m": dev_15m
    }

    # 最低限のサニティチェック（明らかにおかしいときに痕跡を残す）
    warn = []
    if xauusd and xauusd < 1000: warn.append("XAU/USD too low?")
    if xauusd and xauusd > 8000: warn.append("XAU/USD too high?")
    if usdjpy and (usdjpy < 50 or usdjpy > 500): warn.append("USD/JPY out of range?")
    if warn: out["warnings"] = warn

    with open("data.json","w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
