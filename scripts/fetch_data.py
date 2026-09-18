import os
import json
import requests
import yfinance as yf
from datetime import datetime, timezone

FRED_API_KEY = os.environ.get("FRED_API_KEY")

def get_fred_series(series_id, limit=30):
    if not FRED_API_KEY:
        print(f"Warning: FRED_API_KEY is not set. Skipping FRED series {series_id}.")
        return []
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        obs = res.json().get("observations", [])
        data = []
        for item in reversed(obs):
            if item["value"] != ".":
                data.append({"date": item["date"], "value": float(item["value"])})
        return data
    except Exception as e:
        print(f"Error fetching FRED {series_id}: {e}")
        return []

def get_crypto_sentiment():
    url = "https://api.alternative.me/fng/?limit=30"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json().get("data", [])
        return [
            {
                "date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d"),
                "value": int(d["value"]),
                "classification": d["value_classification"]
            }
            for d in reversed(data)
        ]
    except Exception as e:
        print(f"Error fetching crypto sentiment: {e}")
        return []

def get_market_assets():
    tickers = {
        "DXY": "DX-Y.NYB",
        "SPX": "^GSPC",
        "NASDAQ": "^IXIC",
        "GOLD": "GC=F",
        "CRUDE_OIL": "CL=F",
        "BTC": "BTC-USD"
    }
    results = {}
    for key, symbol in tickers.items():
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1mo")
            if hist.empty:
                results[key] = {"current": None, "change_pct": None, "history": []}
                continue
            
            closes = hist["Close"].tolist()
            dates = [d.strftime("%Y-%m-%d") for d in hist.index]
            current = float(closes[-1])
            prev = float(closes[-2]) if len(closes) > 1 else current
            change_pct = round(((current - prev) / prev) * 100, 2)
            
            results[key] = {
                "current": round(current, 2),
                "change_pct": change_pct,
                "history": [{"date": d, "value": round(val, 2)} for d, val in zip(dates, closes)]
            }
        except Exception as e:
            print(f"Error fetching ticker {symbol}: {e}")
            results[key] = {"current": None, "change_pct": None, "history": []}
    return results

def main():
    us10y = get_fred_series("DGS10", limit=30)
    us2y = get_fred_series("DGS2", limit=30)
    fed_assets = get_fred_series("WALCL", limit=12)
    cpi = get_fred_series("CPIAUCSL", limit=12)

    crypto_fng = get_crypto_sentiment()
    market = get_market_assets()

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "crypto_sentiment": {
            "current": crypto_fng[-1] if crypto_fng else None,
            "history": crypto_fng
        },
        "market": market,
        "macro": {
            "us10y": {"current": us10y[-1]["value"] if us10y else None, "history": us10y},
            "us2y": {"current": us2y[-1]["value"] if us2y else None, "history": us2y},
            "fed_assets_mil": {"current": fed_assets[-1]["value"] if fed_assets else None, "history": fed_assets},
            "cpi": {"current": cpi[-1]["value"] if cpi else None, "history": cpi}
        }
    }

    os.makedirs("public", exist_ok=True)
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("data.json generated successfully.")

if __name__ == "__main__":
    main()
