import os
import json
import csv
import io
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from scripts.risk_engine import (
        evaluate_funding_stress, evaluate_move_volatility, evaluate_rates_and_breakeven,
        evaluate_credit_stress, evaluate_cross_asset_stress, evaluate_crypto_structural_risk,
        build_liquidity_calendar, synthesize_macro_regime, get_cme_front_month_expiry
    )
except ImportError:
    from risk_engine import (
        evaluate_funding_stress, evaluate_move_volatility, evaluate_rates_and_breakeven,
        evaluate_credit_stress, evaluate_cross_asset_stress, evaluate_crypto_structural_risk,
        build_liquidity_calendar, synthesize_macro_regime, get_cme_front_month_expiry
    )

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

FRED_API_KEY = os.environ.get("FRED_API_KEY")
COINGLASS_API_KEY = os.environ.get("COINGLASS_API_KEY")
CRYPTOQUANT_API_KEY = os.environ.get("CRYPTOQUANT_API_KEY")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def get_fred_series_all(prev_data=None):
    """
    Fetches all required FRED series sequentially using a shared session.
    Includes target rate upper/lower limits (DFEDTARU, DFEDTARL), daily effective rate (DFF),
    and money market & inflation series: SOFR, IORB, EFFR, T10YIE, BAMLC0A0CM.
    """
    series_ids = [
        "DGS10", "DGS2", "DFII10", "WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL", "BAMLH0A0HYM2", 
        "DFEDTARU", "DFEDTARL", "DFF", "FEDFUNDS",
        "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",
        "SOFR", "IORB", "EFFR", "T10YIE", "BAMLC0A0CM"
    ]
    results = {}
    start_date = (datetime.now(timezone.utc) - timedelta(days=450)).strftime("%Y-%m-%d")
    
    if HAS_CURL_CFFI:
        session = cffi_requests.Session(impersonate="chrome")
    else:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)

    for sid in series_ids:
        data = []
        # Method 1: If FRED_API_KEY is available
        if FRED_API_KEY:
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": sid,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "observation_start": start_date,
                    "sort_order": "asc"
                }
                res = session.get(url, params=params, timeout=15)
                if res.status_code == 200:
                    obs = res.json().get("observations", [])
                    for item in obs:
                        if item["value"] not in (".", "", None):
                            try:
                                data.append({"date": item["date"], "value": float(item["value"])})
                            except ValueError:
                                pass
            except Exception as e:
                print(f"FRED API key request failed for {sid}: {e}")

        # Method 2: Official FRED public CSV with cosd parameter
        if not data:
            try:
                csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start_date}"
                res = session.get(csv_url, timeout=15)
                if res.status_code == 200:
                    reader = csv.reader(io.StringIO(res.text))
                    next(reader, None)  # skip header
                    for row in reader:
                        if len(row) >= 2 and row[1] not in (".", "", None):
                            try:
                                data.append({"date": row[0].strip(), "value": float(row[1].strip())})
                            except ValueError:
                                continue
            except Exception as e:
                print(f"FRED CSV request failed for {sid}: {e}")

        # Fallback to previous data if network failed
        if not data and prev_data:
            cached_hist = None
            if sid == "DGS10" and prev_data.get("rates", {}).get("us10y", {}).get("history"):
                cached_hist = prev_data["rates"]["us10y"]["history"]
            elif sid == "DGS2" and prev_data.get("rates", {}).get("us2y", {}).get("history"):
                cached_hist = prev_data["rates"]["us2y"]["history"]
            elif sid == "DFII10" and prev_data.get("rates", {}).get("us10y_real", {}).get("history"):
                cached_hist = prev_data["rates"]["us10y_real"]["history"]
            elif sid == "BAMLH0A0HYM2" and prev_data.get("traditional_risk", {}).get("hy_spread", {}).get("history"):
                cached_hist = prev_data["traditional_risk"]["hy_spread"]["history"]
            if cached_hist:
                data = cached_hist

        results[sid] = data
        time.sleep(0.12)  # pause to respect rate limits

    return results

def calc_percentile(values, current_val):
    if not values or current_val is None:
        return None
    count = sum(1 for v in values if v <= current_val)
    return round((count / len(values)) * 100, 1)

def calc_changes(series, lookbacks=(1, 7, 30)):
    if not series:
        return None, {}, {}
    curr = series[-1]["value"]
    pcts = {}
    diffs = {}
    n = len(series)
    for lb in lookbacks:
        if n > lb:
            prev = series[-1 - lb]["value"]
            diff = round(curr - prev, 4)
            pct = round(((curr - prev) / prev) * 100, 2) if prev != 0 else 0.0
            diffs[lb] = diff
            pcts[lb] = pct
        else:
            diffs[lb] = None
            pcts[lb] = None
    return curr, pcts, diffs

def calc_rolling_corr(series_a, series_b, window=30):
    """
    Computes Pearson correlation coefficient between two time-series aligned on 'date'.
    """
    if not series_a or not series_b:
        return None
    df_a = pd.DataFrame(series_a).rename(columns={"value": "a"})
    df_b = pd.DataFrame(series_b).rename(columns={"value": "b"})
    merged = pd.merge(df_a, df_b, on="date").sort_values("date").dropna()
    if len(merged) < min(10, window // 2):
        return None
    tail_df = merged.tail(window)
    if len(tail_df) < 5 or tail_df["a"].std() == 0 or tail_df["b"].std() == 0:
        return None
    corr = tail_df["a"].corr(tail_df["b"])
    return round(float(corr), 3) if pd.notnull(corr) else None

def get_crypto_sentiment(prev_data=None):
    url = "https://api.alternative.me/fng/?limit=90"
    for attempt in range(2):
        try:
            res = requests.get(url, headers=DEFAULT_HEADERS, timeout=8)
            res.raise_for_status()
            data = res.json().get("data", [])
            if not data:
                continue
            history = [
                {
                    "date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "value": int(d["value"]),
                    "classification": d["value_classification"]
                }
                for d in reversed(data)
            ]
            curr = history[-1]
            c1d = round(curr["value"] - history[-2]["value"], 1) if len(history) > 1 else 0
            c7d = round(curr["value"] - history[-8]["value"], 1) if len(history) > 7 else 0
            c30d = round(curr["value"] - history[-31]["value"], 1) if len(history) > 30 else 0
            return {
                "current": curr["value"],
                "classification": curr["classification"],
                "change_1d": c1d,
                "change_7d": c7d,
                "change_30d": c30d,
                "updated": curr["date"],
                "frequency": "Daily",
                "source": "Alternative.me",
                "status": "ok",
                "history": history
            }
        except Exception:
            time.sleep(0.5)

    # Fallback to previous data if network is down
    if prev_data and "overview" in prev_data and prev_data["overview"].get("fear_and_greed"):
        cached = prev_data["overview"]["fear_and_greed"]
        cached["status"] = "stale"
        cached["stale_label"] = f"Stale · Last recorded {cached.get('updated', 'earlier')}"
        return cached
    return {
        "current": None,
        "classification": "Unavailable",
        "change_1d": None,
        "change_7d": None,
        "change_30d": None,
        "updated": None,
        "frequency": "Daily",
        "source": "Alternative.me",
        "status": "unavailable"
    }

def fetch_single_ticker(key, primary_symbol, fallback_symbol, provider_name, venue_name, prev_data=None):
    for sym in [primary_symbol, fallback_symbol]:
        if not sym:
            continue
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="1y")
            if hist.empty:
                hist = tk.history(period="6mo")
            if hist.empty:
                continue
            
            closes = [round(float(v), 2) for v in hist["Close"].tolist()]
            dates = [d.strftime("%Y-%m-%d") for d in hist.index]
            history = [{"date": d, "value": v} for d, v in zip(dates, closes)]
            
            curr = closes[-1]
            c1d = round(((curr - closes[-2]) / closes[-2]) * 100, 2) if len(closes) > 1 else 0.0
            c5d = round(((curr - closes[-6]) / closes[-6]) * 100, 2) if len(closes) > 5 else 0.0
            c7d = round(((curr - closes[-8]) / closes[-8]) * 100, 2) if len(closes) > 7 else 0.0
            c20d = round(((curr - closes[-21]) / closes[-21]) * 100, 2) if len(closes) > 20 else 0.0
            c30d = round(((curr - closes[-31]) / closes[-31]) * 100, 2) if len(closes) > 30 else 0.0
            diff_1d = round(curr - closes[-2], 2) if len(closes) > 1 else 0.0
            diff_5d = round(curr - closes[-6], 2) if len(closes) > 5 else 0.0
            diff_20d = round(curr - closes[-21], 2) if len(closes) > 20 else 0.0
            p1y = calc_percentile(closes, curr)

            w_closes = closes[-252:]
            mean_c = sum(w_closes) / len(w_closes)
            var_c = sum((x - mean_c) ** 2 for x in w_closes) / (len(w_closes) - 1 if len(w_closes) > 1 else 1)
            std_c = (var_c ** 0.5)
            zscore_1y = round((curr - mean_c) / std_c, 2) if std_c > 1e-5 else 0.0
            
            return key, {
                "symbol": sym,
                "provider": provider_name,
                "venue": venue_name,
                "current": curr,
                "change_1d": c1d,
                "change_5d": c5d,
                "change_7d": c7d,
                "change_20d": c20d,
                "change_30d": c30d,
                "diff_1d": diff_1d,
                "diff_5d": diff_5d,
                "diff_20d": diff_20d,
                "percentile_1y": p1y,
                "zscore": zscore_1y,
                "updated": dates[-1],
                "frequency": "Daily",
                "source": f"{provider_name} ({sym})",
                "status": "ok",
                "history": history[-90:]
            }
        except Exception as e:
            print(f"Error fetching ticker {sym}: {e}")

    # Fallback to cache if available
    if prev_data:
        for section in ["overview", "traditional_risk", "commodities"]:
            if section in prev_data and key.lower() in prev_data[section]:
                cached = prev_data[section][key.lower()]
                cached["status"] = "stale"
                cached["frequency"] = "Daily"
                cached["source"] = f"{provider_name} ({primary_symbol})"
                cached["stale_label"] = f"Stale · {cached.get('updated', 'earlier')}"
                return key, cached

    return key, {
        "symbol": primary_symbol,
        "provider": provider_name,
        "venue": venue_name,
        "current": None,
        "change_1d": None,
        "change_5d": None,
        "change_7d": None,
        "change_20d": None,
        "change_30d": None,
        "diff_1d": None,
        "diff_5d": None,
        "diff_20d": None,
        "percentile_1y": None,
        "zscore": None,
        "updated": None,
        "frequency": "Daily",
        "source": f"{provider_name} ({primary_symbol})",
        "status": "unavailable"
    }

def get_market_assets(prev_data=None):
    # Precise venue, provider and symbol configuration (no generic labels)
    ticker_configs = {
        "BTC": ("BTC-USD", None, "Yahoo Finance", "Crypto Spot"),
        "DXY": ("DX-Y.NYB", None, "ICE / Yahoo", "US Dollar Index"),
        "SPX": ("^GSPC", "SPY", "S&P Dow Jones / Yahoo", "S&P 500 Index"),
        "NASDAQ": ("^NDX", "^IXIC", "Nasdaq / Yahoo", "Nasdaq 100 Index"),
        "VIX": ("^VIX", None, "CBOE / Yahoo", "CBOE Volatility Index"),
        "GOLD": ("GC=F", None, "COMEX / Yahoo", "COMEX Gold Futures"),
        "CRUDE_OIL": ("CL=F", None, "NYMEX / Yahoo", "NYMEX WTI Crude Futures"),
        "MOVE": ("^MOVE", None, "ICE / Yahoo", "Treasury Volatility Index"),
        "USDJPY": ("JPY=X", None, "Forex / Yahoo", "USD/JPY Currency Pair"),
        "BTC_FUTURES": ("BTC=F", None, "CME / Yahoo", "CME Bitcoin Futures Front Month")
    }
    results = {}
    for k, (pri, fb, prov, ven) in ticker_configs.items():
        _, res = fetch_single_ticker(k, pri, fb, prov, ven, prev_data)
        results[k] = res
    return results

def get_btc_etf_flows(prev_data=None):
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    try:
        if HAS_CURL_CFFI:
            res = cffi_requests.get(url, impersonate="chrome", timeout=20)
            html = res.text
        else:
            res = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
            html = res.text

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            raise ValueError("Farside ETF table not found.")

        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)

        if not rows:
            raise ValueError("Farside table empty.")

        header = rows[0]
        total_idx = -1
        ibit_idx = -1
        fbtc_idx = -1
        gbtc_idx = -1

        for i, h in enumerate(header):
            hl = h.lower()
            if "total" in hl and total_idx == -1:
                total_idx = i
            elif "ibit" in hl:
                ibit_idx = i
            elif "fbtc" in hl:
                fbtc_idx = i
            elif "gbtc" in hl:
                gbtc_idx = i

        if total_idx == -1:
            total_idx = len(header) - 1

        def parse_val(v_str):
            if not v_str:
                return 0.0
            clean = v_str.replace(",", "").replace("$", "").strip()
            if clean in ("-", "", "."):
                return 0.0
            if clean.startswith("(") and clean.endswith(")"):
                return -float(clean[1:-1])
            try:
                return float(clean)
            except ValueError:
                return 0.0

        daily_flows = []
        for r in rows[1:]:
            if len(r) <= total_idx:
                continue
            date_str = r[0].strip()
            if any(k in date_str.lower() for k in ["total", "average", "maximum", "minimum"]):
                continue
            val = parse_val(r[total_idx])
            
            ibit_val = parse_val(r[ibit_idx]) if ibit_idx != -1 and len(r) > ibit_idx else 0.0
            fbtc_val = parse_val(r[fbtc_idx]) if fbtc_idx != -1 and len(r) > fbtc_idx else 0.0
            gbtc_val = parse_val(r[gbtc_idx]) if gbtc_idx != -1 and len(r) > gbtc_idx else 0.0
            
            parsed_date = date_str
            for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
                try:
                    parsed_date = datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

            daily_flows.append({
                "date": parsed_date,
                "value": round(val, 1),
                "ibit": round(ibit_val, 1),
                "fbtc": round(fbtc_val, 1),
                "gbtc": round(gbtc_val, 1)
            })

        if not daily_flows:
            raise ValueError("No valid daily ETF flow records parsed.")

        # Correct logic: DO NOT drop 0.0 net flows! 0.0 is legitimate flow data.
        # Latest trading day is simply the last parsed row in chronological order.
        latest_flow = daily_flows[-1]
        
        # Valid trading days accumulation
        flow_7d = sum(f["value"] for f in daily_flows[-7:])
        flow_30d = sum(f["value"] for f in daily_flows[-30:])

        return {
            "today_net_mil": latest_flow["value"],
            "today_date": latest_flow["date"],
            "flow_7d_mil": round(flow_7d, 1),
            "flow_7d_bil": round(flow_7d / 1000, 2),
            "flow_30d_mil": round(flow_30d, 1),
            "flow_30d_bil": round(flow_30d / 1000, 2),
            "latest_breakdown": {
                "ibit": latest_flow["ibit"],
                "fbtc": latest_flow["fbtc"],
                "gbtc": latest_flow["gbtc"]
            },
            "history_30d": daily_flows[-30:],
            "updated": latest_flow["date"],
            "frequency": "Daily (T+1)",
            "source": "Farside Investors",
            "status": "ok"
        }
    except Exception as e:
        print(f"Error or timeout fetching BTC ETF flows: {e}")
        if prev_data and "overview" in prev_data and prev_data["overview"].get("btc_etf_flow"):
            cached = prev_data["overview"]["btc_etf_flow"]
            cached["status"] = "stale"
            cached["stale_label"] = f"Stale · {cached.get('updated', 'earlier')}"
            return cached
        return {
            "today_net_mil": None,
            "today_date": None,
            "flow_7d_mil": None,
            "flow_7d_bil": None,
            "flow_30d_mil": None,
            "flow_30d_bil": None,
            "latest_breakdown": {"ibit": None, "fbtc": None, "gbtc": None},
            "history_30d": [],
            "updated": None,
            "frequency": "Daily (T+1)",
            "source": "Farside Investors",
            "status": "unavailable"
        }

def get_stablecoins_data(prev_data=None):
    try:
        url_pegged = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
        res_p = requests.get(url_pegged, headers=DEFAULT_HEADERS, timeout=15)
        res_p.raise_for_status()
        pegged_assets = res_p.json().get("peggedAssets", [])

        usdt = next((p for p in pegged_assets if p.get("symbol") == "USDT"), None)
        usdc = next((p for p in pegged_assets if p.get("symbol") == "USDC"), None)

        usdt_mcap = round(usdt["circulating"]["peggedUSD"] / 1e9, 2) if usdt and "circulating" in usdt else None
        usdc_mcap = round(usdc["circulating"]["peggedUSD"] / 1e9, 2) if usdc and "circulating" in usdc else None

        url_hist = "https://stablecoins.llama.fi/stablecoincharts/all"
        res_h = requests.get(url_hist, headers=DEFAULT_HEADERS, timeout=15)
        res_h.raise_for_status()
        hist_raw = res_h.json()

        daily_totals = []
        for item in hist_raw:
            dt = datetime.fromtimestamp(int(item["date"]), tz=timezone.utc).strftime("%Y-%m-%d")
            circ_usd = item.get("totalCirculatingUSD", {}).get("peggedUSD")
            if not circ_usd:
                circ_usd = item.get("totalCirculating", {}).get("peggedUSD", 0)
            daily_totals.append({"date": dt, "value": round(circ_usd / 1e9, 2)})

        if not daily_totals:
            raise ValueError("No stablecoin chart data.")

        curr = daily_totals[-1]["value"]
        d7 = daily_totals[-8]["value"] if len(daily_totals) >= 8 else daily_totals[0]["value"]
        d30 = daily_totals[-31]["value"] if len(daily_totals) >= 31 else daily_totals[0]["value"]
        d90 = daily_totals[-91]["value"] if len(daily_totals) >= 91 else daily_totals[0]["value"]

        c7d = round(((curr - d7) / d7) * 100, 2) if d7 else 0.0
        c30d = round(((curr - d30) / d30) * 100, 2) if d30 else 0.0
        c90d = round(((curr - d90) / d90) * 100, 2) if d90 else 0.0

        # Issuance change in Billions
        net_issuance_7d = round(curr - d7, 2)
        net_issuance_30d = round(curr - d30, 2)

        return {
            "total_mcap_bil": curr,
            "usdt_mcap_bil": usdt_mcap,
            "usdc_mcap_bil": usdc_mcap,
            "change_7d": c7d,
            "change_30d": c30d,
            "change_90d": c90d,
            "net_issuance_7d_bil": net_issuance_7d,
            "net_issuance_30d_bil": net_issuance_30d,
            "history": daily_totals[-90:],
            "updated": daily_totals[-1]["date"],
            "frequency": "Daily",
            "source": "DefiLlama",
            "status": "ok"
        }
    except Exception as e:
        print(f"Error fetching stablecoins data: {e}")
        if prev_data and "crypto_capital_flow" in prev_data and prev_data["crypto_capital_flow"].get("stablecoins"):
            cached = prev_data["crypto_capital_flow"]["stablecoins"]
            cached["status"] = "stale"
            cached["stale_label"] = f"Stale · {cached.get('updated', 'earlier')}"
            return cached
        return {
            "total_mcap_bil": None,
            "usdt_mcap_bil": None,
            "usdc_mcap_bil": None,
            "change_7d": None,
            "change_30d": None,
            "change_90d": None,
            "history": [],
            "updated": None,
            "frequency": "Daily",
            "source": "DefiLlama",
            "status": "unavailable"
        }

def get_crypto_leverage(prev_data=None):
    leverage = {
        "funding_rate": None,
        "open_interest": None,
        "liquidations": {
            "value": None,
            "status": "unavailable",
            "reason": "Requires CoinGlass API Key",
            "source": "CoinGlass"
        },
        "exchange_netflow": {
            "value": None,
            "status": "unavailable",
            "reason": "Requires CryptoQuant / CoinGlass API Key",
            "source": "CryptoQuant"
        }
    }

    # 1. Binance Funding Rate
    try:
        res = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", headers=DEFAULT_HEADERS, timeout=8)
        if res.status_code == 200:
            d = res.json()
            rate_8h = float(d.get("lastFundingRate", 0.0))
            annualized = rate_8h * 3 * 365 * 100
            
            if annualized > 15.0:
                heat = "Elevated"
            elif annualized < 4.0:
                heat = "Low"
            else:
                heat = "Normal"

            leverage["funding_rate"] = {
                "venue": "Binance BTCUSDT Perpetual",
                "rate_8h": round(rate_8h * 100, 4),
                "annualized_pct": round(annualized, 2),
                "heat": heat,
                "updated": datetime.fromtimestamp(int(d.get("time", 0)) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "frequency": "8-Hour Funding",
                "source": "Binance Futures (BTCUSDT)",
                "status": "ok"
            }
    except Exception as e:
        print(f"Error fetching Binance funding rate: {e}")
        if prev_data and prev_data.get("crypto_leverage", {}).get("funding_rate"):
            cached = prev_data["crypto_leverage"]["funding_rate"]
            cached["status"] = "stale"
            cached["source"] = "Binance Futures (BTCUSDT)"
            cached["venue"] = "Binance BTCUSDT Perpetual"
            leverage["funding_rate"] = cached

    # 2. Binance Open Interest
    try:
        res_hist = requests.get("https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1d&limit=31", headers=DEFAULT_HEADERS, timeout=8)
        if res_hist.status_code == 200:
            items = res_hist.json()
            if items:
                history = []
                for it in items:
                    dt = datetime.fromtimestamp(int(it["timestamp"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                    val_usd_bil = round(float(it["sumOpenInterestValue"]) / 1e9, 3)
                    history.append({"date": dt, "value": val_usd_bil, "btc": round(float(it["sumOpenInterest"]), 1)})
                
                curr = history[-1]["value"]
                c24h = round(((curr - history[-2]["value"]) / history[-2]["value"]) * 100, 2) if len(history) > 1 else 0.0
                c7d = round(((curr - history[-8]["value"]) / history[-8]["value"]) * 100, 2) if len(history) > 7 else 0.0

                leverage["open_interest"] = {
                    "venue": "Binance BTC Futures OI",
                    "total_usd_bil": curr,
                    "total_btc": history[-1]["btc"],
                    "change_24h": c24h,
                    "change_7d": c7d,
                    "history": history,
                    "updated": history[-1]["date"],
                    "frequency": "Daily",
                    "source": "Binance Futures (BTC)",
                    "status": "ok"
                }
    except Exception as e:
        print(f"Error fetching Binance Open Interest: {e}")
        if prev_data and prev_data.get("crypto_leverage", {}).get("open_interest"):
            cached = prev_data["crypto_leverage"]["open_interest"]
            cached["status"] = "stale"
            cached["source"] = "Binance Futures (BTC)"
            cached["venue"] = "Binance BTC Futures OI"
            leverage["open_interest"] = cached

    # 3. CoinGlass Liquidations
    if COINGLASS_API_KEY:
        try:
            cg_headers = {"CG-API-KEY": COINGLASS_API_KEY}
            cg_res = requests.get("https://open-api-v4.coinglass.com/api/futures/liquidation/exchange-list?symbol=BTC&range=24h", headers=cg_headers, timeout=10)
            if cg_res.status_code == 200 and cg_res.json().get("code") == "0":
                liq_data = cg_res.json().get("data", {})
                total_liq = round(float(liq_data.get("liquidation_usd", 0)) / 1e6, 2)
                long_liq = round(float(liq_data.get("long_liquidation_usd", 0)) / 1e6, 2)
                short_liq = round(float(liq_data.get("short_liquidation_usd", 0)) / 1e6, 2)
                leverage["liquidations"] = {
                    "total_mil": total_liq,
                    "long_mil": long_liq,
                    "short_mil": short_liq,
                    "status": "ok",
                    "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "frequency": "24H Rolling",
                    "source": "CoinGlass"
                }
        except Exception as e:
            print(f"Error fetching CoinGlass liquidations: {e}")

    # 4. Binance BTC Spot 24H Volume (Quote USD & BTC)
    try:
        res_v = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", headers=DEFAULT_HEADERS, timeout=8)
        if res_v.status_code == 200:
            vd = res_v.json()
            q_vol = round(float(vd.get("quoteVolume", 0.0)) / 1e9, 3)
            b_vol = round(float(vd.get("volume", 0.0)), 1)
            leverage["spot_volume_24h"] = {
                "volume_usd_bil": q_vol,
                "volume_btc": b_vol,
                "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "source": "Binance Spot (BTCUSDT)",
                "frequency": "24H Rolling",
                "status": "ok"
            }
        else:
            leverage["spot_volume_24h"] = {"volume_usd_bil": None, "volume_btc": None, "source": "Binance Spot", "status": "unavailable"}
    except Exception as e:
        print(f"Error fetching Binance spot volume: {e}")
        leverage["spot_volume_24h"] = {"volume_usd_bil": None, "volume_btc": None, "source": "Binance Spot", "status": "unavailable"}

    return leverage

def build_macro_regime(dxy_data, us10y_real, us_liquidity_proxy, spx_data, vix_data, 
                       hy_spread, etf_flows, crypto_leverage):
    """
    Multi-dimensional Macro Regime Evaluation:
    Combines Level (Percentile), Trend (30D/7D Momentum), and Stress/State.
    CRITICAL FIX: None is NEVER evaluated as 'Normal'!
    """
    # 1. USD State (Level + Trend)
    dxy_curr = dxy_data.get("current") if dxy_data else None
    dxy_p1y = dxy_data.get("percentile_1y") if dxy_data else None
    dxy_c30 = dxy_data.get("change_30d") if dxy_data else None

    if dxy_curr is None:
        usd_dim = {"state": "Unavailable", "level": "Unknown", "trend": "Unknown", "percentile_1y": None}
    else:
        lvl = "High" if (dxy_p1y and dxy_p1y >= 65) else ("Low" if (dxy_p1y and dxy_p1y <= 35) else "Normal")
        tr = "Rising" if (dxy_c30 and dxy_c30 > 0.8) else ("Falling" if (dxy_c30 and dxy_c30 < -0.8) else "Stable")
        if lvl == "High" or (tr == "Rising" and lvl != "Low"):
            state = "Strong"
        elif lvl == "Low" or (tr == "Falling" and lvl != "High"):
            state = "Weak"
        else:
            state = "Neutral"
        usd_dim = {"state": state, "level": lvl, "trend": tr, "percentile_1y": dxy_p1y, "change_30d": dxy_c30}

    # 2. 10Y Real Yield (Level + Trend + Stress)
    ry_curr = us10y_real.get("current") if us10y_real else None
    ry_p1y = us10y_real.get("percentile_1y") if us10y_real else None
    ry_c30_bp = us10y_real.get("change_30d_bp") if us10y_real else None

    if ry_curr is None:
        ry_dim = {"state": "Unavailable", "level": "Unknown", "trend": "Unknown", "stress": "Unknown"}
    else:
        lvl = "High" if (ry_p1y and ry_p1y >= 65) else ("Low" if (ry_p1y and ry_p1y <= 35) else "Normal")
        tr = "Rising" if (ry_c30_bp and ry_c30_bp > 8) else ("Falling" if (ry_c30_bp and ry_c30_bp < -8) else "Stable")
        stress = "High" if (lvl == "High" and tr == "Rising") else ("Low" if tr == "Falling" else "Medium")
        state = "Rising" if tr == "Rising" else ("Falling" if tr == "Falling" else "Stable")
        ry_dim = {"state": state, "level": lvl, "trend": tr, "stress": stress, "percentile_1y": ry_p1y}

    # 3. US Net Liquidity Proxy (Trend + Magnitude)
    liq_c30 = us_liquidity_proxy.get("change_30d") if us_liquidity_proxy else None
    if liq_c30 is None:
        liq_dim = {"state": "Unavailable", "trend": "Unknown", "stress": "Unknown"}
    else:
        state = "Expanding" if liq_c30 > 0.4 else ("Contracting" if liq_c30 < -0.4 else "Neutral")
        liq_dim = {"state": state, "trend": state, "change_30d": liq_c30}

    # 4. TradFi Equities & Volatility (SPX Trend + VIX Level & Spike)
    spx_c30 = spx_data.get("change_30d") if spx_data else None
    vix_curr = vix_data.get("current") if vix_data else None
    vix_c30 = vix_data.get("change_30d") if vix_data else None

    if spx_c30 is None or vix_curr is None:
        eq_dim = {"state": "Unavailable", "risk": "Unknown", "vix_level": "Unknown"}
    else:
        # Avoid static VIX < 20 assumption: check if VIX spiked recently or elevated
        vix_spiking = vix_c30 is not None and vix_c30 > 25.0
        if vix_curr < 18.5 and not vix_spiking and (spx_c30 is not None and spx_c30 >= -0.5):
            state = "Risk-On"
        elif vix_curr > 23.0 or vix_spiking or (spx_c30 is not None and spx_c30 < -2.5):
            state = "Risk-Off"
        else:
            state = "Neutral"
        eq_dim = {"state": state, "vix_current": vix_curr, "vix_spike": vix_spiking, "spx_change_30d": spx_c30}

    # 5. Credit Market (HY Spread OAS Level + Trend)
    hy_curr = hy_spread.get("current") if hy_spread else None
    hy_c1m_bp = hy_spread.get("change_1m_bp") if hy_spread else None
    hy_p1y = hy_spread.get("percentile_1y") if hy_spread else None

    if hy_curr is None:
        credit_dim = {"state": "Unavailable", "level": "Unknown", "stress": "Unknown"}
    else:
        # FRED ICE BofA HY OAS: historical normal ~3.2% - 4.5%. Currently ~2.7% is tight.
        widening = hy_c1m_bp is not None and hy_c1m_bp > 25.0
        if hy_curr < 3.4 and not widening:
            state = "Healthy"
        elif hy_curr > 4.2 or widening:
            state = "Stress"
        else:
            state = "Neutral"
        credit_dim = {"state": state, "current_spread": hy_curr, "widening": widening, "percentile_1y": hy_p1y}

    # 6. BTC ETF Capital Flow
    etf_7d = etf_flows.get("flow_7d_mil") if etf_flows else None
    etf_30d = etf_flows.get("flow_30d_mil") if etf_flows else None
    if etf_7d is None:
        etf_dim = {"state": "Unavailable", "flow_7d_mil": None}
    else:
        state = "Inflow" if etf_7d > 50.0 else ("Outflow" if etf_7d < -50.0 else "Neutral")
        etf_dim = {"state": state, "flow_7d_mil": etf_7d, "flow_30d_mil": etf_30d}

    # 7. Crypto Leverage (CRITICAL P0 FIX: None -> Unavailable, NEVER Normal!)
    funding = crypto_leverage.get("funding_rate") if crypto_leverage else None
    oi = crypto_leverage.get("open_interest") if crypto_leverage else None

    funding_ann = funding.get("annualized_pct") if (funding and funding.get("status") == "ok") else None
    oi_c7 = oi.get("change_7d") if (oi and oi.get("status") == "ok") else None

    if funding_ann is None and oi_c7 is None:
        # Data is genuinely missing/unavailable -> NEVER label as Normal!
        lev_state = "Unavailable"
        lev_heat = "Unknown"
    elif funding_ann is not None and (funding_ann > 15.0 or (oi_c7 is not None and oi_c7 > 15.0)):
        lev_state = "Elevated"
        lev_heat = "Elevated"
    elif funding_ann is not None and funding_ann < 3.0:
        lev_state = "Low"
        lev_heat = "Low"
    else:
        lev_state = "Normal"
        lev_heat = "Normal"

    lev_dim = {
        "state": lev_state,
        "heat": lev_heat,
        "funding_annualized": funding_ann,
        "oi_change_7d": oi_c7,
        "status": "ok" if (funding_ann is not None or oi_c7 is not None) else "unavailable"
    }

    # Synthesize concise institutional macro summary line
    summary_parts = []
    if liq_dim["state"] != "Unavailable":
        summary_parts.append(f"{liq_dim['state']} Liquidity")
    if usd_dim["state"] != "Unavailable":
        summary_parts.append(f"{usd_dim['state']} USD")
    if ry_dim["state"] != "Unavailable":
        summary_parts.append(f"{ry_dim['state']} Real Yield")
    if credit_dim["state"] != "Unavailable":
        summary_parts.append(f"{credit_dim['state']} Credit")
    if eq_dim["state"] != "Unavailable":
        summary_parts.append(f"{eq_dim['state']} Equities")
    summary_parts.append(f"ETF {etf_dim['state']}")
    summary_parts.append(f"Leverage {lev_dim['state']}")
    regime_summary = " · ".join(summary_parts)

    return {
        # Backward-compatible top-level keys
        "usd": usd_dim["state"],
        "real_yield": ry_dim["state"],
        "liquidity": liq_dim["state"],
        "equities": eq_dim["state"],
        "credit": credit_dim["state"],
        "btc_etf_flow": etf_dim["state"],
        "crypto_leverage": lev_dim["state"],
        "summary": regime_summary,
        # Multi-dimensional indicators
        "dimensions": {
            "usd": usd_dim,
            "real_yield": ry_dim,
            "liquidity": liq_dim,
            "equities": eq_dim,
            "credit": credit_dim,
            "btc_etf_flow": etf_dim,
            "crypto_leverage": lev_dim
        }
    }

def detect_market_divergence(regime, btc_data, dxy_data, nasdaq_data, us10y_real, 
                             gold_data, etf_flows, crypto_leverage, hy_spread):
    """
    Rigorously detects Directional Divergence, Correlation Shifts, and Flow Divergences.
    Includes rolling 30D and 90D Pearson correlation metrics.
    """
    cases = []
    
    # Extract histories for rolling correlation
    btc_hist = btc_data.get("history", []) if btc_data else []
    dxy_hist = dxy_data.get("history", []) if dxy_data else []
    ndx_hist = nasdaq_data.get("history", []) if nasdaq_data else []
    ry_hist = us10y_real.get("history", []) if us10y_real else []
    gold_hist = gold_data.get("history", []) if gold_data else []

    # Calculate actual rolling correlations
    corr_btc_ndx_30d = calc_rolling_corr(btc_hist, ndx_hist, 30)
    corr_btc_ndx_90d = calc_rolling_corr(btc_hist, ndx_hist, 90)
    corr_btc_dxy_30d = calc_rolling_corr(btc_hist, dxy_hist, 30)
    corr_btc_dxy_90d = calc_rolling_corr(btc_hist, dxy_hist, 90)
    corr_btc_ry_30d = calc_rolling_corr(btc_hist, ry_hist, 30)
    corr_btc_ry_90d = calc_rolling_corr(btc_hist, ry_hist, 90)
    corr_btc_gold_30d = calc_rolling_corr(btc_hist, gold_hist, 30)
    corr_btc_gold_90d = calc_rolling_corr(btc_hist, gold_hist, 90)

    correlations = {
        "btc_nasdaq": {"corr_30d": corr_btc_ndx_30d, "corr_90d": corr_btc_ndx_90d},
        "btc_dxy": {"corr_30d": corr_btc_dxy_30d, "corr_90d": corr_btc_dxy_90d},
        "btc_real_yield": {"corr_30d": corr_btc_ry_30d, "corr_90d": corr_btc_ry_90d},
        "btc_gold": {"corr_30d": corr_btc_gold_30d, "corr_90d": corr_btc_gold_90d}
    }

    btc_c7d = btc_data.get("change_7d") if btc_data else None
    btc_c30d = btc_data.get("change_30d") if btc_data else None
    dxy_c30d = dxy_data.get("change_30d") if dxy_data else None
    ndx_c30d = nasdaq_data.get("change_30d") if nasdaq_data else None
    ry_c30d_bp = us10y_real.get("change_30d_bp") if us10y_real else None
    etf_7d = etf_flows.get("flow_7d_mil") if etf_flows else None
    hy_curr = hy_spread.get("current") if hy_spread else None

    funding = crypto_leverage.get("funding_rate") if crypto_leverage else None
    oi = crypto_leverage.get("open_interest") if crypto_leverage else None
    funding_ann = funding.get("annualized_pct") if (funding and funding.get("status") == "ok") else None
    oi_c7 = oi.get("change_7d") if (oi and oi.get("status") == "ok") else None

    # Check 1: Correlation Regime Shift (Statistical decorrelation)
    if corr_btc_ndx_30d is not None and corr_btc_ndx_90d is not None:
        if corr_btc_ndx_90d > 0.4 and corr_btc_ndx_30d < 0.05:
            cases.append({
                "case_id": "Shift-Corr",
                "type": "Correlation Regime Shift",
                "name": "BTC / Tech Equity Decorrelation",
                "message": f"Rolling 30D correlation between BTC and Nasdaq dropped from +{corr_btc_ndx_90d} (90D) to {corr_btc_ndx_30d}, demonstrating statistical decorrelation from traditional tech equities."
            })

    # Check 2: Directional Divergence (Asset Class divergence, not yet statistical decorrelation)
    if (dxy_c30d is not None and dxy_c30d > 0.5) and \
       (ry_c30d_bp is not None and ry_c30d_bp > 5) and \
       (ndx_c30d is not None and ndx_c30d < -1.0) and \
       (btc_c30d is not None and btc_c30d > 2.0):
        cases.append({
            "case_id": "Case A",
            "type": "Directional Divergence",
            "name": "Macro Headwind Absorption",
            "message": "BTC is exhibiting positive directional divergence, absorbing capital despite USD strengthening and real yields rising."
        })

    # Check 3: Macro & Spot Flow Alignment
    if (dxy_c30d is not None and dxy_c30d < 0) and \
       (ry_c30d_bp is not None and ry_c30d_bp < 0) and \
       (etf_7d is not None and etf_7d > 100) and \
       (btc_c7d is not None and btc_c7d > 0):
        cases.append({
            "case_id": "Case B",
            "type": "Regime Alignment",
            "name": "Macro & ETF Spot Alignment",
            "message": "Macro liquidity easing (USD/Real Yield drop) and institutional spot ETF net inflows are mutually aligned in supporting crypto assets."
        })

    # Check 4: Leverage vs Spot Flow Divergence
    if (btc_c7d is not None and btc_c7d > 0) and \
       (etf_7d is not None and etf_7d < -50) and \
       (oi_c7 is not None and oi_c7 > 5) and \
       (funding_ann is not None and funding_ann > 10):
        cases.append({
            "case_id": "Case C",
            "type": "Flow Divergence",
            "name": "Leverage vs Spot Flow Divergence",
            "message": "BTC price upside is accompanied by escalating derivatives leverage while spot ETF net flow is negative (fragile leverage-driven structure)."
        })

    # Check 5: Credit vs Equity Divergence
    if (hy_curr is not None and hy_curr > 3.8) and \
       (ndx_c30d is not None and ndx_c30d > 2.0):
        cases.append({
            "case_id": "Case D",
            "type": "TradFi Credit Divergence",
            "name": "Credit vs Equity Divergence",
            "message": "High yield credit spreads indicate growing corporate financial stress while equity indices remain buoyant."
        })

    if not cases:
        cases.append({
            "case_id": "Neutral",
            "type": "Regime Congruence",
            "name": "Regime Congruence",
            "message": "Cross-asset relationships and liquidity indicators are operating within standard historical correlation bands without acute divergence."
        })

    return {
        "cases": cases,
        "correlations": correlations
    }

def build_data_quality_report(tracked_indicators):
    """
    Rigorously computes live data availability, stale count, and missing indicators.
    Replaces static/fake '100% Live Sync' with actual operational metrics.
    """
    total = len(tracked_indicators)
    available_cnt = 0
    stale_cnt = 0
    unavailable_cnt = 0

    metric_details = []
    for ind in tracked_indicators:
        status = ind.get("status", "unavailable")
        if status == "ok":
            available_cnt += 1
        elif status == "stale":
            stale_cnt += 1
            available_cnt += 1  # count as degraded available
        else:
            unavailable_cnt += 1

        metric_details.append({
            "name": ind.get("name"),
            "category": ind.get("category"),
            "status": status,
            "frequency": ind.get("frequency", "Daily"),
            "source": ind.get("source", "Official"),
            "updated": ind.get("updated")
        })

    pct = round((available_cnt / total) * 100, 1) if total > 0 else 0.0

    return {
        "total_metrics": total,
        "available_count": available_cnt,
        "stale_count": stale_cnt,
        "unavailable_count": unavailable_cnt,
        "availability_pct": pct,
        "summary": f"{available_cnt}/{total} Available ({pct}%)",
        "health_state": "Optimal" if pct >= 90 else ("Operational" if pct >= 70 else "Degraded"),
        "metrics": metric_details
    }

def main():
    print("Starting Macro Daily Dashboard pipeline (v2.1 Precision Upgrade)...")
    now_utc = datetime.now(timezone.utc)

    prev_data = {}
    if os.path.exists("public/data.json"):
        try:
            with open("public/data.json", "r", encoding="utf-8") as f:
                prev_data = json.load(f)
        except Exception:
            pass
    
    # Run async thread pool for independent services
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_market = executor.submit(get_market_assets, prev_data)
        future_fng = executor.submit(get_crypto_sentiment, prev_data)
        future_etf = executor.submit(get_btc_etf_flows, prev_data)
        future_stable = executor.submit(get_stablecoins_data, prev_data)
        future_lev = executor.submit(get_crypto_leverage, prev_data)
        future_fred = executor.submit(get_fred_series_all, prev_data)

        market = future_market.result()
        crypto_fng = future_fng.result()
        etf_flows = future_etf.result()
        stablecoins = future_stable.result()
        crypto_leverage = future_lev.result()
        fred_data = future_fred.result()

    btc_data = market.get("BTC")
    dxy_data = market.get("DXY")
    spx_data = market.get("SPX")
    nasdaq_data = market.get("NASDAQ")
    vix_data = market.get("VIX")
    gold_data = market.get("GOLD")
    oil_data = market.get("CRUDE_OIL")

    us10y_obs = fred_data.get("DGS10", [])
    us2y_obs = fred_data.get("DGS2", [])
    dfii10_obs = fred_data.get("DFII10", [])
    walcl_obs = fred_data.get("WALCL", [])
    wdtgal_obs = fred_data.get("WDTGAL", [])
    rrp_obs = fred_data.get("RRPONTSYD", [])
    reserves_obs = fred_data.get("WRBWFRBL", [])
    hy_obs = fred_data.get("BAMLH0A0HYM2", [])
    ig_obs = fred_data.get("BAMLC0A0CM", [])
    sofr_obs = fred_data.get("SOFR", [])
    iorb_obs = fred_data.get("IORB", [])
    effr_obs = fred_data.get("EFFR", [])
    t10yie_obs = fred_data.get("T10YIE", [])
    
    # Official Fed Funds target range & effective rates
    dfedtaru_obs = fred_data.get("DFEDTARU", [])
    dfedtarl_obs = fred_data.get("DFEDTARL", [])
    dff_obs = fred_data.get("DFF", [])
    fedfunds_obs = fred_data.get("FEDFUNDS", [])

    # 1. Process US 10Y
    u10y_curr, u10y_pct, u10y_diff = calc_changes(us10y_obs)
    us10y = {
        "current": u10y_curr,
        "change_1d_bp": round(u10y_diff.get(1, 0) * 100, 1) if u10y_diff.get(1) is not None else None,
        "change_7d_bp": round(u10y_diff.get(7, 0) * 100, 1) if u10y_diff.get(7) is not None else None,
        "change_30d_bp": round(u10y_diff.get(30, 0) * 100, 1) if u10y_diff.get(30) is not None else None,
        "percentile_1y": calc_percentile([o["value"] for o in us10y_obs[-260:]], u10y_curr) if us10y_obs else None,
        "updated": us10y_obs[-1]["date"] if us10y_obs else None,
        "frequency": "Daily",
        "source": "FRED (U.S. Treasury DGS10)",
        "status": "ok" if us10y_obs else "unavailable",
        "history": us10y_obs[-90:] if us10y_obs else []
    }

    # 2. Process US 2Y
    u2y_curr, u2y_pct, u2y_diff = calc_changes(us2y_obs)
    us2y = {
        "current": u2y_curr,
        "change_1d_bp": round(u2y_diff.get(1, 0) * 100, 1) if u2y_diff.get(1) is not None else None,
        "change_7d_bp": round(u2y_diff.get(7, 0) * 100, 1) if u2y_diff.get(7) is not None else None,
        "change_30d_bp": round(u2y_diff.get(30, 0) * 100, 1) if u2y_diff.get(30) is not None else None,
        "percentile_1y": calc_percentile([o["value"] for o in us2y_obs[-260:]], u2y_curr) if us2y_obs else None,
        "updated": us2y_obs[-1]["date"] if us2y_obs else None,
        "frequency": "Daily",
        "source": "FRED (U.S. Treasury DGS2)",
        "status": "ok" if us2y_obs else "unavailable",
        "history": us2y_obs[-90:] if us2y_obs else []
    }

    # 3. Process 10Y-2Y Spread
    spread_obs = []
    if us10y_obs and us2y_obs:
        d2_map = {o["date"]: o["value"] for o in us2y_obs}
        for o in us10y_obs:
            if o["date"] in d2_map:
                diff_bp = round((o["value"] - d2_map[o["date"]]) * 100, 1)
                spread_obs.append({"date": o["date"], "value": diff_bp})
    
    sp_curr, sp_pct, sp_diff = calc_changes(spread_obs)
    spread_10y_2y = {
        "current_bp": sp_curr,
        "change_1d_bp": sp_diff.get(1),
        "change_7d_bp": sp_diff.get(7),
        "change_30d_bp": sp_diff.get(30),
        "updated": spread_obs[-1]["date"] if spread_obs else None,
        "frequency": "Daily",
        "source": "FRED / Calculated (10Y-2Y)",
        "status": "ok" if spread_obs else "unavailable",
        "history": spread_obs[-90:] if spread_obs else []
    }

    # 4. Process 10Y Real Yield (DFII10)
    dfii_curr, dfii_pct, dfii_diff = calc_changes(dfii10_obs)
    us10y_real = {
        "current": dfii_curr,
        "change_1d_bp": round(dfii_diff.get(1, 0) * 100, 1) if dfii_diff.get(1) is not None else None,
        "change_7d_bp": round(dfii_diff.get(7, 0) * 100, 1) if dfii_diff.get(7) is not None else None,
        "change_30d_bp": round(dfii_diff.get(30, 0) * 100, 1) if dfii_diff.get(30) is not None else None,
        "percentile_1y": calc_percentile([o["value"] for o in dfii10_obs[-260:]], dfii_curr) if dfii10_obs else None,
        "updated": dfii10_obs[-1]["date"] if dfii10_obs else None,
        "frequency": "Daily",
        "source": "FRED (10Y TIPS DFII10)",
        "status": "ok" if dfii10_obs else "unavailable",
        "history": dfii10_obs[-90:] if dfii10_obs else []
    }

    # 5. Process Fed Balance Sheet (WALCL)
    fed_bs = None
    if walcl_obs:
        curr_walcl = walcl_obs[-1]["value"]
        curr_t = round(curr_walcl / 1e6, 2)
        n_w = len(walcl_obs)
        c1m = round(((curr_walcl - walcl_obs[-5]["value"]) / walcl_obs[-5]["value"]) * 100, 2) if n_w >= 5 else None
        c3m = round(((curr_walcl - walcl_obs[-14]["value"]) / walcl_obs[-14]["value"]) * 100, 2) if n_w >= 14 else None
        c6m = round(((curr_walcl - walcl_obs[-27]["value"]) / walcl_obs[-27]["value"]) * 100, 2) if n_w >= 27 else None
        cyoy = round(((curr_walcl - walcl_obs[-53]["value"]) / walcl_obs[-53]["value"]) * 100, 2) if n_w >= 53 else None

        fed_bs = {
            "current_tril": curr_t,
            "change_1m": c1m,
            "change_3m": c3m,
            "change_6m": c6m,
            "change_yoy": cyoy,
            "updated": walcl_obs[-1]["date"],
            "frequency": "Weekly (Wed / Rel Thu)",
            "source": "Federal Reserve (H.4.1 WALCL)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"] / 1e6, 2)} for o in walcl_obs[-52:]]
        }

    # 6. Process TGA (WDTGAL)
    tga = None
    if wdtgal_obs:
        curr_tga = wdtgal_obs[-1]["value"]
        curr_b = round(curr_tga / 1000, 1)
        n_tga = len(wdtgal_obs)
        c7d = round(((curr_tga - wdtgal_obs[-2]["value"]) / wdtgal_obs[-2]["value"]) * 100, 1) if n_tga >= 2 else None
        c30d = round(((curr_tga - wdtgal_obs[-5]["value"]) / wdtgal_obs[-5]["value"]) * 100, 1) if n_tga >= 5 else None
        tga = {
            "current_bil": curr_b,
            "change_7d": c7d,
            "change_30d": c30d,
            "updated": wdtgal_obs[-1]["date"],
            "frequency": "Weekly",
            "source": "U.S. Treasury / FRED (WDTGAL)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"] / 1000, 1)} for o in wdtgal_obs[-52:]]
        }

    # 7. Process RRP (RRPONTSYD)
    rrp = None
    if rrp_obs:
        curr_rrp = rrp_obs[-1]["value"]
        curr_b = round(curr_rrp, 2)
        n_rrp = len(rrp_obs)
        c7d = round(((curr_rrp - rrp_obs[-8]["value"]) / rrp_obs[-8]["value"]) * 100, 1) if n_rrp >= 8 and rrp_obs[-8]["value"] > 0 else None
        c30d = round(((curr_rrp - rrp_obs[-31]["value"]) / rrp_obs[-31]["value"]) * 100, 1) if n_rrp >= 31 and rrp_obs[-31]["value"] > 0 else None
        rrp = {
            "current_bil": curr_b,
            "change_7d": c7d,
            "change_30d": c30d,
            "updated": rrp_obs[-1]["date"],
            "frequency": "Daily",
            "source": "NY Fed / FRED (RRPONTSYD)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"], 2)} for o in rrp_obs[-90:]]
        }

    # 8. Process Bank Reserves (WRBWFRBL)
    bank_reserves = None
    if reserves_obs:
        curr_res = reserves_obs[-1]["value"]
        curr_b = round(curr_res / 1000, 1)
        n_res = len(reserves_obs)
        c30d = round(((curr_res - reserves_obs[-5]["value"]) / reserves_obs[-5]["value"]) * 100, 1) if n_res >= 5 else None
        bank_reserves = {
            "current_bil": curr_b,
            "change_30d": c30d,
            "updated": reserves_obs[-1]["date"],
            "frequency": "Weekly (H.4.1)",
            "source": "Federal Reserve (WRBWFRBL)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"] / 1000, 1)} for o in reserves_obs[-52:]]
        }

    # 9. Compute US Net Liquidity Proxy: Fed Balance Sheet - TGA - RRP
    us_liquidity_proxy = None
    if walcl_obs and wdtgal_obs:
        tga_map = {o["date"]: o["value"] for o in wdtgal_obs}
        rrp_map = {o["date"]: o["value"] * 1000 for o in (rrp_obs or [])}
        
        liq_history = []
        for w in walcl_obs:
            dt = w["date"]
            if dt in tga_map:
                fed_val = w["value"]
                tga_val = tga_map[dt]
                rrp_val = rrp_map.get(dt, 0.0)
                proxy_val = round((fed_val - tga_val - rrp_val) / 1e6, 2)
                liq_history.append({"date": dt, "value": proxy_val})
        
        if liq_history:
            curr_liq = liq_history[-1]["value"]
            n_l = len(liq_history)
            c7d = round(((curr_liq - liq_history[-2]["value"]) / liq_history[-2]["value"]) * 100, 2) if n_l >= 2 else 0.0
            c30d = round(((curr_liq - liq_history[-5]["value"]) / liq_history[-5]["value"]) * 100, 2) if n_l >= 5 else 0.0
            us_liquidity_proxy = {
                "name": "US Net Liquidity Proxy",
                "current_tril": curr_liq,
                "change_7d": c7d,
                "change_30d": c30d,
                "formula": "Fed Balance Sheet (Weekly) - TGA (Weekly) - RRP (Daily)",
                "note": "Market proxy for domestic dollar liquidity available to financial markets",
                "is_derived": True,
                "updated": liq_history[-1]["date"],
                "frequency": "Weekly (Wed snapshot)",
                "source": "Derived (Fed + Treasury H.4.1)",
                "status": "ok",
                "history": liq_history[-52:]
            }

    # 10. Process High Yield Credit Spread (BAMLH0A0HYM2) & IG Spread (BAMLC0A0CM)
    credit_result = evaluate_credit_stress(hy_obs, ig_obs)
    hy_spread = None
    if hy_obs:
        hy_curr, hy_pct, hy_diff = calc_changes(hy_obs)
        c1d_bp = round(hy_diff.get(1, 0) * 100, 1) if hy_diff.get(1) is not None else None
        c5d_bp = round(hy_diff.get(5, 0) * 100, 1) if hy_diff.get(5) is not None else None
        c20d_bp = round(hy_diff.get(20, 0) * 100, 1) if hy_diff.get(20) is not None else None
        hy_spread = {
            "current": hy_curr,
            "change_1d_bp": c1d_bp,
            "change_5d_bp": c5d_bp,
            "change_20d_bp": c20d_bp,
            "change_1w_bp": c5d_bp,
            "change_1m_bp": c20d_bp,
            "percentile_1y": credit_result.get("percentile"),
            "zscore": credit_result.get("zscore"),
            "status": credit_result.get("status", "ok"),
            "updated": hy_obs[-1]["date"],
            "frequency": "Daily",
            "source": "ICE BofA / FRED (BAMLH0A0HYM2)",
            "history": hy_obs[-90:]
        }

    ig_spread = None
    if ig_obs:
        ig_curr, ig_pct, ig_diff = calc_changes(ig_obs)
        ig_spread = {
            "current": ig_curr,
            "change_1d_bp": round(ig_diff.get(1, 0) * 100, 1) if ig_diff.get(1) is not None else None,
            "change_5d_bp": round(ig_diff.get(5, 0) * 100, 1) if ig_diff.get(5) is not None else None,
            "change_20d_bp": round(ig_diff.get(20, 0) * 100, 1) if ig_diff.get(20) is not None else None,
            "percentile_1y": calc_percentile([o["value"] for o in ig_obs[-260:]], ig_curr),
            "updated": ig_obs[-1]["date"],
            "frequency": "Daily",
            "source": "ICE BofA / FRED (BAMLC0A0CM)",
            "status": "ok",
            "history": ig_obs[-90:]
        }

    # 11. P0 FIX: Rigorous Federal Funds Rate (Target Range vs Effective Rate)
    # Distinguishes FOMC Target Range (DFEDTARU, DFEDTARL) and Daily Effective Fed Funds Rate (DFF)
    fed_funds = None
    if dfedtaru_obs and dfedtarl_obs:
        target_upper = dfedtaru_obs[-1]["value"]
        target_lower = dfedtarl_obs[-1]["value"]
        target_range = f"{target_lower:.2f}% - {target_upper:.2f}%"
        target_date = dfedtaru_obs[-1]["date"]

        # Check target change
        target_change_bp = 0.0
        if len(dfedtaru_obs) > 1:
            prev_u = dfedtaru_obs[-2]["value"]
            target_change_bp = round((target_upper - prev_u) * 100, 1)

        dff_val = dff_obs[-1]["value"] if dff_obs else None
        dff_date = dff_obs[-1]["date"] if dff_obs else None
        monthly_val = fedfunds_obs[-1]["value"] if fedfunds_obs else None

        fed_funds = {
            "target_upper": target_upper,
            "target_lower": target_lower,
            "target_range": target_range,
            "target_updated": target_date,
            "target_change_bp": target_change_bp,
            "effective_rate": dff_val,
            "effective_updated": dff_date,
            "monthly_average": monthly_val,
            "current": dff_val if dff_val is not None else target_upper,
            "updated": target_date,
            "frequency": "Daily / FOMC Decision",
            "source": "Federal Reserve (FRED DFEDTARU / DFEDTARL / DFF)",
            "status": "ok",
            "history": [{"date": o["date"], "value": o["value"]} for o in dfedtaru_obs[-60:]]
        }
    elif fedfunds_obs:
        fed_funds = {
            "target_upper": None,
            "target_lower": None,
            "target_range": "--",
            "target_updated": None,
            "target_change_bp": None,
            "effective_rate": fedfunds_obs[-1]["value"],
            "effective_updated": fedfunds_obs[-1]["date"],
            "monthly_average": fedfunds_obs[-1]["value"],
            "current": fedfunds_obs[-1]["value"],
            "updated": fedfunds_obs[-1]["date"],
            "frequency": "Monthly",
            "source": "Federal Reserve (FRED FEDFUNDS)",
            "status": "ok",
            "history": fedfunds_obs[-24:]
        }

    # 12. P0 FIX: Rigorous Treasury Short-End Bill Curve (1M, 3M, 6M, 1Y)
    dgs1mo_obs = fred_data.get("DGS1MO", [])
    dgs3mo_obs = fred_data.get("DGS3MO", [])
    dgs6mo_obs = fred_data.get("DGS6MO", [])
    dgs1_obs = fred_data.get("DGS1", [])

    rate_path = {
        "title": "US Treasury Short-End Bill Curve",
        "description": "Yields across 1M, 3M, 6M, 12M Treasury bills. Driven by bill supply, money-market demand, and Fed rate expectations.",
        "m1": dgs1mo_obs[-1]["value"] if dgs1mo_obs else None,
        "m3": dgs3mo_obs[-1]["value"] if dgs3mo_obs else None,
        "m6": dgs6mo_obs[-1]["value"] if dgs6mo_obs else None,
        "m12": dgs1_obs[-1]["value"] if dgs1_obs else None,
        "updated": dgs1mo_obs[-1]["date"] if dgs1mo_obs else None,
        "frequency": "Daily",
        "source": "U.S. Treasury / FRED",
        "status": "ok" if dgs1mo_obs else "unavailable"
    }

    # 13. P0: 10Y Breakeven Inflation & Rates Regime
    rates_result = evaluate_rates_and_breakeven(us10y_obs, dfii10_obs, t10yie_obs)
    us10y_breakeven = None
    if t10yie_obs:
        be_curr, be_pct, be_diff = calc_changes(t10yie_obs)
        us10y_breakeven = {
            "current": be_curr,
            "change_1d_bp": round(be_diff.get(1, 0) * 100, 1) if be_diff.get(1) is not None else None,
            "change_5d_bp": round(be_diff.get(5, 0) * 100, 1) if be_diff.get(5) is not None else None,
            "change_20d_bp": round(be_diff.get(20, 0) * 100, 1) if be_diff.get(20) is not None else None,
            "percentile_1y": calc_percentile([o["value"] for o in t10yie_obs[-260:]], be_curr),
            "updated": t10yie_obs[-1]["date"],
            "frequency": "Daily",
            "source": "FRED (10Y Breakeven T10YIE)",
            "status": "ok",
            "history": t10yie_obs[-90:]
        }
    elif rates_result.get("breakeven_10y") is not None:
        us10y_breakeven = {
            "current": rates_result.get("breakeven_10y"),
            "change_1d_bp": rates_result.get("breakeven_change_1d_bp"),
            "change_5d_bp": rates_result.get("breakeven_change_5d_bp"),
            "change_20d_bp": rates_result.get("breakeven_change_20d_bp"),
            "percentile_1y": None,
            "updated": us10y.get("updated"),
            "frequency": "Daily",
            "source": "Derived (Nominal 10Y - TIPS Real 10Y)",
            "status": "ok",
            "history": []
        }

    # 14. P0: Funding Stress (SOFR, IORB, EFFR, Spreads, Percentiles, Z-Scores)
    funding_result = evaluate_funding_stress(sofr_obs, iorb_obs, effr_obs, dfedtaru_obs)
    sofr_data = {
        "sofr": funding_result.get("sofr"),
        "iorb": funding_result.get("iorb"),
        "effr": funding_result.get("effr"),
        "sofr_iorb_spread_bp": funding_result.get("sofr_iorb_spread_bp"),
        "sofr_effr_spread_bp": funding_result.get("sofr_effr_spread_bp"),
        "change_1d_bp": funding_result.get("sofr_change_1d_bp"),
        "spread_change_5d_bp": funding_result.get("spread_change_5d_bp"),
        "spread_change_20d_bp": funding_result.get("spread_change_20d_bp"),
        "percentile_1y": funding_result.get("percentile"),
        "zscore": funding_result.get("zscore"),
        "status": funding_result.get("status"),
        "updated": sofr_obs[-1]["date"] if sofr_obs else None,
        "frequency": "Daily",
        "source": "NY Fed / Federal Reserve / FRED",
        "history": funding_result.get("history", [])
    }

    # 15. P0: MOVE Bond Volatility Evaluation
    move_result = evaluate_move_volatility(market.get("MOVE"))

    # 16. P1: Cross-Asset Stress Engine & Liquidity Liquidation Detection
    cross_asset_result = evaluate_cross_asset_stress(market, move_result, credit_result, rates_result)

    # 17. P1: CME BTC Futures Basis
    btc_futures_data = market.get("BTC_FUTURES")
    cme_basis_data = {
        "futures_price": None,
        "spot_price": None,
        "basis_usd": None,
        "annualized_pct": None,
        "days_to_expiry": None,
        "expiry_date": None,
        "is_compressed": False,
        "source": "CME Futures (BTC=F) vs Spot (BTC-USD)",
        "frequency": "Daily",
        "status": "unavailable"
    }
    btc_spot_curr = btc_data.get("current") if btc_data else None
    btc_fut_curr = btc_futures_data.get("current") if btc_futures_data else None
    if btc_spot_curr and btc_fut_curr:
        exp_date, days_exp = get_cme_front_month_expiry()
        basis_usd = round(btc_fut_curr - btc_spot_curr, 2)
        ann_pct = round((basis_usd / btc_spot_curr) * (365.0 / days_exp) * 100.0, 2)
        cme_basis_data = {
            "futures_price": btc_fut_curr,
            "spot_price": btc_spot_curr,
            "basis_usd": basis_usd,
            "annualized_pct": ann_pct,
            "days_to_expiry": days_exp,
            "expiry_date": exp_date,
            "is_compressed": ann_pct < 4.0,
            "source": "CME Futures (BTC=F) vs Spot (BTC-USD)",
            "frequency": "Daily",
            "status": "ok",
            "updated": btc_futures_data.get("updated")
        }

    # 18. P1: Crypto Structural Risk (ETF Flows, CME Basis, OI/Spot Volume, Leverage Fragility)
    crypto_structural_result = evaluate_crypto_structural_risk(
        etf_flows=etf_flows,
        cme_basis_data=cme_basis_data,
        crypto_leverage=crypto_leverage,
        stablecoins=stablecoins,
        btc_spot_vol=crypto_leverage.get("spot_volume_24h")
    )

    # 19. P2: Overall Macro Risk Regime & Explanation Engine
    macro_risk_regime = synthesize_macro_regime(
        funding_res=funding_result,
        rates_res=rates_result,
        credit_res=credit_result,
        cross_asset_res=cross_asset_result,
        crypto_res=crypto_structural_result,
        liquidity_res=us_liquidity_proxy
    )

    # 20. P1: Next Liquidity Risk Window (Calendar)
    liquidity_calendar = build_liquidity_calendar()

    # Legacy Backward-Compatible Macro Regime Engine
    macro_regime = build_macro_regime(
        dxy_data=dxy_data,
        us10y_real=us10y_real,
        us_liquidity_proxy=us_liquidity_proxy,
        spx_data=spx_data,
        vix_data=vix_data,
        hy_spread=hy_spread,
        etf_flows=etf_flows,
        crypto_leverage=crypto_leverage
    )

    # Rigorous Divergence Analysis with Real Rolling Pearson Correlations
    divergence_data = detect_market_divergence(
        regime=macro_regime,
        btc_data=btc_data,
        dxy_data=dxy_data,
        nasdaq_data=nasdaq_data,
        us10y_real=us10y_real,
        gold_data=gold_data,
        etf_flows=etf_flows,
        crypto_leverage=crypto_leverage,
        hy_spread=hy_spread
    )

    # Tracked Data Quality Metrics Audit
    tracked_metrics = [
        {"name": "Bitcoin (BTC)", "category": "Crypto Spot", "status": btc_data.get("status"), "source": btc_data.get("source"), "updated": btc_data.get("updated"), "frequency": "Daily"},
        {"name": "US Dollar Index (DXY)", "category": "Currency", "status": dxy_data.get("status"), "source": dxy_data.get("source"), "updated": dxy_data.get("updated"), "frequency": "Daily"},
        {"name": "USD/JPY Currency Pair", "category": "Currency", "status": market.get("USDJPY", {}).get("status"), "source": "Forex / Yahoo", "updated": market.get("USDJPY", {}).get("updated"), "frequency": "Daily"},
        {"name": "S&P 500 (SPX)", "category": "Equities", "status": spx_data.get("status"), "source": spx_data.get("source"), "updated": spx_data.get("updated"), "frequency": "Daily"},
        {"name": "Nasdaq 100 (NDX)", "category": "Equities", "status": nasdaq_data.get("status"), "source": nasdaq_data.get("source"), "updated": nasdaq_data.get("updated"), "frequency": "Daily"},
        {"name": "VIX Volatility", "category": "Volatility", "status": vix_data.get("status"), "source": vix_data.get("source"), "updated": vix_data.get("updated"), "frequency": "Daily"},
        {"name": "MOVE Bond Volatility", "category": "Volatility", "status": market.get("MOVE", {}).get("status"), "source": "ICE / Yahoo", "updated": market.get("MOVE", {}).get("updated"), "frequency": "Daily"},
        {"name": "COMEX Gold", "category": "Commodities", "status": gold_data.get("status"), "source": gold_data.get("source"), "updated": gold_data.get("updated"), "frequency": "Daily"},
        {"name": "NYMEX Crude Oil", "category": "Commodities", "status": oil_data.get("status"), "source": oil_data.get("source"), "updated": oil_data.get("updated"), "frequency": "Daily"},
        {"name": "US 10Y Yield", "category": "Rates", "status": us10y.get("status"), "source": us10y.get("source"), "updated": us10y.get("updated"), "frequency": "Daily"},
        {"name": "US 2Y Yield", "category": "Rates", "status": us2y.get("status"), "source": us2y.get("source"), "updated": us2y.get("updated"), "frequency": "Daily"},
        {"name": "10Y Real Yield (TIPS)", "category": "Rates", "status": us10y_real.get("status"), "source": us10y_real.get("source"), "updated": us10y_real.get("updated"), "frequency": "Daily"},
        {"name": "10Y Breakeven Inflation", "category": "Rates", "status": us10y_breakeven.get("status") if us10y_breakeven else "unavailable", "source": "FRED", "updated": us10y_breakeven.get("updated") if us10y_breakeven else None, "frequency": "Daily"},
        {"name": "10Y-2Y Yield Spread", "category": "Rates", "status": spread_10y_2y.get("status"), "source": spread_10y_2y.get("source"), "updated": spread_10y_2y.get("updated"), "frequency": "Daily"},
        {"name": "SOFR Overnight Rate", "category": "Funding Stress", "status": sofr_data.get("status"), "source": "NY Fed", "updated": sofr_data.get("updated"), "frequency": "Daily"},
        {"name": "Fed Target Range (DFEDTARU/L)", "category": "Policy", "status": fed_funds.get("status") if fed_funds else "unavailable", "source": "FRED", "updated": fed_funds.get("target_updated") if fed_funds else None, "frequency": "Daily"},
        {"name": "Effective Fed Funds Rate (DFF)", "category": "Policy", "status": "ok" if fed_funds and fed_funds.get("effective_rate") else "unavailable", "source": "FRED", "updated": fed_funds.get("effective_updated") if fed_funds else None, "frequency": "Daily"},
        {"name": "Treasury Bill Curve", "category": "Rates", "status": rate_path.get("status"), "source": "FRED", "updated": rate_path.get("updated"), "frequency": "Daily"},
        {"name": "HY Credit Spread (OAS)", "category": "Credit", "status": hy_spread.get("status") if hy_spread else "unavailable", "source": "FRED", "updated": hy_spread.get("updated") if hy_spread else None, "frequency": "Daily"},
        {"name": "IG Credit Spread (OAS)", "category": "Credit", "status": ig_spread.get("status") if ig_spread else "unavailable", "source": "FRED", "updated": ig_spread.get("updated") if ig_spread else None, "frequency": "Daily"},
        {"name": "Fed Balance Sheet (WALCL)", "category": "Liquidity", "status": fed_bs.get("status") if fed_bs else "unavailable", "source": "Federal Reserve", "updated": fed_bs.get("updated") if fed_bs else None, "frequency": "Weekly"},
        {"name": "Treasury General Account (TGA)", "category": "Liquidity", "status": tga.get("status") if tga else "unavailable", "source": "U.S. Treasury", "updated": tga.get("updated") if tga else None, "frequency": "Weekly"},
        {"name": "Reverse Repo (RRP)", "category": "Liquidity", "status": rrp.get("status") if rrp else "unavailable", "source": "NY Fed", "updated": rrp.get("updated") if rrp else None, "frequency": "Daily"},
        {"name": "Bank Reserves", "category": "Liquidity", "status": bank_reserves.get("status") if bank_reserves else "unavailable", "source": "Federal Reserve", "updated": bank_reserves.get("updated") if bank_reserves else None, "frequency": "Weekly"},
        {"name": "US Net Liquidity Proxy", "category": "Liquidity", "status": us_liquidity_proxy.get("status") if us_liquidity_proxy else "unavailable", "source": "Derived", "updated": us_liquidity_proxy.get("updated") if us_liquidity_proxy else None, "frequency": "Weekly"},
        {"name": "BTC Spot ETF Net Flow", "category": "Crypto Capital Flow", "status": etf_flows.get("status") if etf_flows else "unavailable", "source": "Farside Investors", "updated": etf_flows.get("updated") if etf_flows else None, "frequency": "Daily"},
        {"name": "CME BTC Basis", "category": "Crypto Leverage", "status": cme_basis_data.get("status"), "source": "CME / Yahoo", "updated": cme_basis_data.get("updated"), "frequency": "Daily"},
        {"name": "Binance BTC Spot 24H Vol", "category": "Crypto Capital Flow", "status": crypto_leverage.get("spot_volume_24h", {}).get("status", "unavailable"), "source": "Binance Spot", "updated": crypto_leverage.get("spot_volume_24h", {}).get("updated"), "frequency": "24H"},
        {"name": "Stablecoin Market Cap", "category": "Crypto Capital Flow", "status": stablecoins.get("status") if stablecoins else "unavailable", "source": "DefiLlama", "updated": stablecoins.get("updated") if stablecoins else None, "frequency": "Daily"},
        {"name": "Binance BTC Funding Rate", "category": "Crypto Leverage", "status": crypto_leverage.get("funding_rate", {}).get("status", "unavailable") if crypto_leverage.get("funding_rate") else "unavailable", "source": "Binance Futures", "updated": crypto_leverage.get("funding_rate", {}).get("updated") if crypto_leverage.get("funding_rate") else None, "frequency": "8-Hour"},
        {"name": "Binance BTC Open Interest", "category": "Crypto Leverage", "status": crypto_leverage.get("open_interest", {}).get("status", "unavailable") if crypto_leverage.get("open_interest") else "unavailable", "source": "Binance Futures", "updated": crypto_leverage.get("open_interest", {}).get("updated") if crypto_leverage.get("open_interest") else None, "frequency": "Daily"},
        {"name": "Futures Liquidations", "category": "Crypto Leverage", "status": crypto_leverage.get("liquidations", {}).get("status", "unavailable"), "source": "CoinGlass", "updated": None, "frequency": "24H"},
        {"name": "Exchange Netflow", "category": "Crypto Capital Flow", "status": crypto_leverage.get("exchange_netflow", {}).get("status", "unavailable"), "source": "CryptoQuant", "updated": None, "frequency": "Daily"},
        {"name": "Crypto Fear & Greed", "category": "Sentiment", "status": crypto_fng.get("status") if crypto_fng else "unavailable", "source": "Alternative.me", "updated": crypto_fng.get("updated") if crypto_fng else None, "frequency": "Daily"}
    ]
    data_quality = build_data_quality_report(tracked_metrics)

    output = {
        "updated_at": now_utc.isoformat(),
        "data_quality": data_quality,
        "risk_engine": macro_risk_regime,
        "liquidity_calendar": liquidity_calendar,
        "macro_regime": macro_regime,
        "market_divergence": divergence_data.get("cases", []),
        "rolling_correlations": divergence_data.get("correlations", {}),
        "overview": {
            "btc": btc_data,
            "btc_etf_flow": etf_flows,
            "dxy": dxy_data,
            "fear_and_greed": crypto_fng
        },
        "traditional_risk": {
            "spx": spx_data,
            "nasdaq": nasdaq_data,
            "vix": vix_data,
            "move": market.get("MOVE"),
            "usdjpy": market.get("USDJPY"),
            "hy_spread": hy_spread,
            "ig_spread": ig_spread
        },
        "rates": {
            "us10y": us10y,
            "us2y": us2y,
            "us10y_real": us10y_real,
            "us10y_breakeven": us10y_breakeven,
            "spread_10y_2y": spread_10y_2y,
            "fed_funds": fed_funds,
            "rate_path": rate_path,
            "sofr": sofr_data,
            "funding_stress": funding_result,
            "rates_regime": rates_result
        },
        "us_liquidity": {
            "fed_balance_sheet": fed_bs,
            "tga": tga,
            "rrp": rrp,
            "bank_reserves": bank_reserves,
            "us_liquidity_proxy": us_liquidity_proxy
        },
        "crypto_capital_flow": {
            "stablecoins": stablecoins,
            "btc_etf_flow": etf_flows,
            "btc_exchange_netflow": crypto_leverage.get("exchange_netflow"),
            "cme_basis": cme_basis_data
        },
        "crypto_leverage": {
            "funding_rate": crypto_leverage.get("funding_rate"),
            "open_interest": crypto_leverage.get("open_interest"),
            "liquidations": crypto_leverage.get("liquidations"),
            "exchange_netflow": crypto_leverage.get("exchange_netflow"),
            "spot_volume_24h": crypto_leverage.get("spot_volume_24h"),
            "oi_spot_ratio": crypto_structural_result.get("oi_spot_ratio"),
            "structural_risk": crypto_structural_result
        },
        "commodities": {
            "gold": gold_data,
            "crude_oil": oil_data
        }
    }

    os.makedirs("public", exist_ok=True)
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Success: public/data.json generated. Quality: {data_quality['summary']}.")
    print(f"Macro Risk Regime: {macro_risk_regime['overall_regime']} ({macro_risk_regime['risk_level']})")

if __name__ == "__main__":
    main()

