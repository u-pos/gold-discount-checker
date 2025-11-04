#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

JST = timezone(timedelta(hours=9))

def now_jst_str():
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

def last_price_smart(ticker: str):
    """
    できるだけ『いまに近い』終値を取る:
      1) 1分足 (1d,1m) の最後
      2) 5分足 (5d,5m) の最後
      3) 15分足 (60d,15m) の最後
      4) fast_info.last_price
      5) 1日足 (5d,1d) の最後
    返却: (price, timestamp_str)
    """
    def hist_last(period, interval):
        try:
            h = yf.Ticker(ticker).history(period=period, interval=interval)
            if h is not None and len(h) > 0:
                close = h["Close"].dropna()
                if len(close) > 0:
                    ts = close.index[-1]
                    # tz-aware to JST-readable
                    if hasattr(ts, "tz_convert"):
                        try:
                            tsj = ts.tz_convert(JST)
                        except Exception:
                            tsj = ts
                    else:
                        tsj = ts
                    return float(close.iloc[-1]), str(tsj)
        except Exception:
            pass
        return None, None

    # 1) 1分
    p, ts = hist_last("1d", "1m")
    if p is not None: return p, ts
    # 2) 5分
    p, ts = hist_last("5d", "5m")
    if p is not None: return p, ts
    # 3) 15分
    p, ts = hist_last("60d", "15m")
    if p is not None: return p, ts
    # 4) fast_info
    try:
        fi = yf.Ticker(ticker).fast_info or {}
        lp = fi.get("last_price")
        if lp and math.isfinite(lp):
            return float(lp), "fast_info"
    except Exception:
        pass
    # 5) 1日
    p, ts = hist_last("5d", "1d")
    if p is not None: return p, ts
    return None, None

def series_close(ticker: str, period: str, interval: str):
    try:
        s = yf.Ticker(ticker).history(period=period, interval=interval)["Close"].dropna()
        return s
    except Exception:
        return None

def estimate_k_regression(df: pd.DataFrame):
    # df: columns xau, jpy, etf (aligned and dropna)
    if df is None or len(df) < 5:
        return None
    df = df.dropna()
    if len(df) < 5:
        return None
    X = df["xau"] * df["jpy"]
    Y = df["etf"]
    num = float((X*Y).sum())
    den = float((X*X).sum())
    if den <= 0:
        return None
    return num/den

def align_join(x: pd.Series, j: pd.Series, e: pd.Series, tail=None):
    if x is None or j is None or e is None:
        return None
    df = pd.concat([x.rename("xau"), j.rename("jpy"), e.rename("etf")], axis=1).dropna()
    if tail:
        df = df.tail(tail)
    # すべてが数値であることを確認
    df = df[np.isfinite(df).all(1)]
    return df if len(df) > 0 else None

def main():
    # --- live prices (as close as possible) ---
    xauusd, t_xau = last_price_smart("XAUUSD=X")
    usdjpy, t_jpy = last_price_smart("JPY=X")
    price1540, t_etf = last_price_smart("1540.T")

    # --- day mode: last 3 daily bars ---
    x_day = series_close("XAUUSD=X", "1mo", "1d")
    j_day = series_close("JPY=X",      "1mo", "1d")
    e_day = series_close("1540.T",     "1mo", "1d")
    df_day = align_join(x_day, j_day, e_day, tail=10)
    k_day = theo_day = dev_day = None
    if df_day is not None:
        k_day = estimate_k_regression(df_day.tail(3))
    if k_day and xauusd and usdjpy and price1540:
        theo_day = xauusd * usdjpy * k_day
        dev_day  = price1540 / theo_day - 1.0

    # --- scalp 5m: last ~36 bars ---
    x_5 = series_close("XAUUSD=X", "5d", "5m")
    j_5 = series_close("JPY=X",     "5d", "5m")
    e_5 = series_close("1540.T",    "5d", "5m")  # 場中のみ更新のことが多い
    df_5 = align_join(x_5, j_5, e_5)
    k_5m = theo_5m = dev_5m = None
    if df_5 is not None:
        k_5m = estimate_k_regression(df_5.tail(36))
    if k_5m and xauusd and usdjpy and price1540:
        theo_5m = xauusd * usdjpy * k_5m
        dev_5m  = price1540 / theo_5m - 1.0

    # --- scalp 15m: last ~32 bars (~8 hours) ---
    x_15 = series_close("XAUUSD=X", "60d", "15m")
    j_15 = series_close("JPY=X",     "60d", "15m")
    e_15 = series_close("1540.T",    "60d", "15m")
    df_15 = align_join(x_15, j_15, e_15)
    k_15m = theo_15m = dev_15m = None
    if df_15 is not None:
        k_15m = estimate_k_regression(df_15.tail(32))
    if k_15m and xauusd and usdjpy and price1540:
        theo_15m = xauusd * usdjpy * k_15m
        dev_15m  = price1540 / theo_15m - 1.0

    out = {
        "time_jst": now_jst_str(),
        "xauusd": xauusd,
        "usdjpy": usdjpy,
        "price1540": price1540,
        "time_src": {
            "xau": t_xau,
            "jpy": t_jpy,
            "etf": t_etf
        },
        "k_day": k_day,       "theo_day": theo_day,       "dev_day": dev_day,
        "k_5m": k_5m,         "theo_5m":  theo_5m,        "dev_5m":  dev_5m,
        "k_15m": k_15m,       "theo_15m": theo_15m,       "dev_15m": dev_15m
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
