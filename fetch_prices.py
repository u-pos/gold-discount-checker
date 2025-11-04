#!/usr/bin/env python3
import json, math
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

JST = timezone(timedelta(hours=9))

def last_price(t):
    fi = {}
    try:
        fi = t.fast_info or {}
    except Exception:
        pass
    p = fi.get("last_price")
    if p and math.isfinite(p):
        return float(p)
    # fallback: last close
    hist = t.history(period="5d", interval="1m")
    if hist is None or len(hist)==0:
        hist = t.history(period="5d", interval="1d")
    if hist is not None and len(hist)>0:
        return float(hist["Close"].dropna().iloc[-1])
    return None

def series(ticker, period, interval, tail=None):
    s = yf.Ticker(ticker).history(period=period, interval=interval)["Close"].dropna()
    if tail:
        s = s.tail(tail)
    return s

def estimate_k_regression(x, j, e):
    df = pd.concat([x.rename("xau"), j.rename("jpy"), e.rename("etf")], axis=1).dropna()
    if len(df) < 5:
        return None
    df["X"] = df["xau"] * df["jpy"]
    df["Y"] = df["etf"]
    num = float((df["X"]*df["Y"]).sum())
    den = float((df["X"]*df["X"]).sum())
    if den <= 0:
        return None
    return num/den

def main():
    # Tickers
    t_xau = yf.Ticker("XAUUSD=X")  # alt GC=F not used here
    t_jpy = yf.Ticker("JPY=X")
    t_etf = yf.Ticker("1540.T")

    # Live prices
    xauusd = last_price(t_xau)
    usdjpy = last_price(t_jpy)
    price1540 = last_price(t_etf)

    # Day mode (3営業日)
    x_day = series("XAUUSD=X", "1mo", "1d", tail=10)
    j_day = series("JPY=X", "1mo", "1d", tail=10)
    e_day = series("1540.T", "1mo", "1d", tail=10)
    # align
    df_day = pd.concat([x_day.rename("xau"), j_day.rename("jpy"), e_day.rename("etf")], axis=1).dropna().tail(3)
    k_day = estimate_k_regression(df_day["xau"], df_day["jpy"], df_day["etf"])

    theo_day = dev_day = None
    if k_day and xauusd and usdjpy and price1540:
        theo_day = xauusd * usdjpy * k_day
        dev_day = price1540 / theo_day - 1.0

    # Scalp mode (5分×直近36本). 1540の分足が取れない場合があるので回帰できないことも
    try:
        x_5m = series("XAUUSD=X", "5d", "5m")
        j_5m = series("JPY=X", "5d", "5m")
        e_5m = series("1540.T", "5d", "5m")  # 取れないと空
        df_5m = pd.concat([x_5m.rename("xau"), j_5m.rename("jpy"), e_5m.rename("etf")], axis=1).dropna().tail(36)
        k_scalp = estimate_k_regression(df_5m["xau"], df_5m["jpy"], df_5m["etf"])
    except Exception:
        k_scalp = None

    theo_scalp = dev_scalp = None
    if k_scalp and xauusd and usdjpy and price1540:
        theo_scalp = xauusd * usdjpy * k_scalp
        dev_scalp = price1540 / theo_scalp - 1.0

    out = {
        "time_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        "xauusd": xauusd,
        "usdjpy": usdjpy,
        "price1540": price1540,
        "k_day": k_day,
        "theo_day": theo_day,
        "dev_day": dev_day,
        "k_scalp": k_scalp,
        "theo_scalp": theo_scalp,
        "dev_scalp": dev_scalp,
    }
    with open("data.json","w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
