import os
import json
import csv
import io
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import requests
import yfinance as yf
from bs4 import BeautifulSoup

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

def get_fred_series_all():
    """
    Fetches all required FRED series sequentially using a shared session.
    St. Louis Fed drops connections if hit with many concurrent requests from the same IP.
    """
    series_ids = [
        "DGS10", "DGS2", "DFII10", "WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL", "BAMLH0A0HYM2", "FEDFUNDS",
        "DGS1MO", "DGS3MO", "DGS6MO", "DGS1"
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

        results[sid] = data
        time.sleep(0.15)  # brief pause to respect rate limits

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

def get_crypto_sentiment(prev_data=None):
    url = "https://api.alternative.me/fng/?limit=90"
    for attempt in range(2):
        try:
            res = requests.get(url, headers=DEFAULT_HEADERS, timeout=5)
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
        except Exception as e:
            time.sleep(0.5)

    # Fallback to previous data if network is down
    if prev_data and "overview" in prev_data and prev_data["overview"].get("fear_and_greed"):
        cached = prev_data["overview"]["fear_and_greed"]
        cached["status"] = "stale"
        cached["stale_label"] = f"Stale · Last recorded {cached.get('updated', 'earlier')}"
        return cached
    elif prev_data and "crypto_sentiment" in prev_data and prev_data["crypto_sentiment"].get("current"):
        old_c = prev_data["crypto_sentiment"]["current"]
        return {
            "current": old_c.get("value"),
            "classification": old_c.get("classification"),
            "change_1d": 0,
            "change_7d": 0,
            "change_30d": 0,
            "updated": old_c.get("date"),
            "frequency": "Daily",
            "source": "Alternative.me",
            "status": "stale",
            "stale_label": f"Stale · {old_c.get('date')}",
            "history": prev_data["crypto_sentiment"].get("history", [])
        }
    return None

def fetch_single_ticker(key, primary_symbol, fallback_symbol=None):
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
            c7d = round(((curr - closes[-8]) / closes[-8]) * 100, 2) if len(closes) > 7 else 0.0
            c30d = round(((curr - closes[-31]) / closes[-31]) * 100, 2) if len(closes) > 30 else 0.0
            p1y = calc_percentile(closes, curr)
            
            return key, {
                "symbol": sym,
                "current": curr,
                "change_1d": c1d,
                "change_7d": c7d,
                "change_30d": c30d,
                "percentile_1y": p1y,
                "updated": dates[-1],
                "frequency": "Intraday" if key in ("BTC", "DXY") else "Daily",
                "source": "Exchange (CBOE/CME/NYBOT)",
                "status": "ok",
                "history": history[-90:]
            }
        except Exception as e:
            print(f"Error fetching ticker {sym}: {e}")
    return key, None

def get_market_assets():
    tickers = {
        "BTC": ("BTC-USD", None),
        "DXY": ("DX-Y.NYB", None),
        "SPX": ("^GSPC", "SPY"),
        "NASDAQ": ("^NDX", "^IXIC"),
        "VIX": ("^VIX", None),
        "GOLD": ("GC=F", None),
        "CRUDE_OIL": ("CL=F", None)
    }
    results = {}
    for k, (pri, fb) in tickers.items():
        _, res = fetch_single_ticker(k, pri, fb)
        results[k] = res
    return results

def get_btc_etf_flows():
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    try:
        if HAS_CURL_CFFI:
            res = cffi_requests.get(url, impersonate="chrome", timeout=25)
            html = res.text
        else:
            res = requests.get(url, headers=DEFAULT_HEADERS, timeout=25)
            html = res.text

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            print("Warning: Farside ETF table not found.")
            return None

        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)

        if not rows:
            return None

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
            return None

        active_flows = [f for f in daily_flows if f["value"] != 0.0]
        latest_flow = active_flows[-1] if active_flows else daily_flows[-1]
        
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
        print(f"Error fetching BTC ETF flows: {e}")
        return None

def get_stablecoins_data():
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
            return None

        curr = daily_totals[-1]["value"]
        d7 = daily_totals[-8]["value"] if len(daily_totals) >= 8 else daily_totals[0]["value"]
        d30 = daily_totals[-31]["value"] if len(daily_totals) >= 31 else daily_totals[0]["value"]
        d90 = daily_totals[-91]["value"] if len(daily_totals) >= 91 else daily_totals[0]["value"]

        c7d = round(((curr - d7) / d7) * 100, 2) if d7 else 0.0
        c30d = round(((curr - d30) / d30) * 100, 2) if d30 else 0.0
        c90d = round(((curr - d90) / d90) * 100, 2) if d90 else 0.0

        return {
            "total_mcap_bil": curr,
            "usdt_mcap_bil": usdt_mcap,
            "usdc_mcap_bil": usdc_mcap,
            "change_7d": c7d,
            "change_30d": c30d,
            "change_90d": c90d,
            "history": daily_totals[-90:],
            "updated": daily_totals[-1]["date"],
            "frequency": "Daily",
            "source": "DefiLlama",
            "status": "ok"
        }
    except Exception as e:
        print(f"Error fetching stablecoins data: {e}")
        return None

def get_crypto_leverage():
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

    try:
        res = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", headers=DEFAULT_HEADERS, timeout=10)
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
                "rate_8h": round(rate_8h * 100, 4),
                "annualized_pct": round(annualized, 2),
                "heat": heat,
                "updated": datetime.fromtimestamp(int(d.get("time", 0)) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "frequency": "8 Hours",
                "source": "Binance Futures",
                "status": "ok"
            }
    except Exception as e:
        print(f"Error fetching Binance funding rate: {e}")

    try:
        res_hist = requests.get("https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1d&limit=31", headers=DEFAULT_HEADERS, timeout=10)
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
                    "total_usd_bil": curr,
                    "total_btc": history[-1]["btc"],
                    "change_24h": c24h,
                    "change_7d": c7d,
                    "history": history,
                    "updated": history[-1]["date"],
                    "frequency": "Daily",
                    "source": "Binance Futures",
                    "status": "ok"
                }
    except Exception as e:
        print(f"Error fetching Binance Open Interest: {e}")

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

    return leverage

def build_macro_regime(usd_pct_30d, dxy_p1y, real_yield_change_30d, liq_proxy_change_30d, 
                       spx_change_30d, vix_curr, hy_spread_curr, hy_change_1m, 
                       etf_flow_7d, funding_annualized, oi_change_7d):
    usd_state = "Strong" if (usd_pct_30d is not None and usd_pct_30d > 0) or (dxy_p1y is not None and dxy_p1y > 60) else "Weak"
    real_yield_state = "Rising" if (real_yield_change_30d is not None and real_yield_change_30d > 0) else "Falling"
    liquidity_state = "Expanding" if (liq_proxy_change_30d is not None and liq_proxy_change_30d > 0) else "Contracting"
    equities_state = "Risk-On" if (spx_change_30d is not None and spx_change_30d >= 0 and vix_curr is not None and vix_curr < 20) else "Risk-Off"
    credit_state = "Healthy" if (hy_spread_curr is not None and hy_spread_curr < 3.5 and (hy_change_1m is None or hy_change_1m <= 0)) else "Stress"
    etf_state = "Inflow" if (etf_flow_7d is not None and etf_flow_7d > 0) else "Outflow"
    
    if funding_annualized is not None and funding_annualized > 15.0 or (oi_change_7d is not None and oi_change_7d > 15.0):
        leverage_state = "Elevated"
    elif funding_annualized is not None and funding_annualized < 4.0:
        leverage_state = "Low"
    else:
        leverage_state = "Normal"

    return {
        "usd": usd_state,
        "real_yield": real_yield_state,
        "liquidity": liquidity_state,
        "equities": equities_state,
        "credit": credit_state,
        "btc_etf_flow": etf_state,
        "crypto_leverage": leverage_state
    }

def detect_market_divergence(regime, btc_change_7d, btc_change_30d, dxy_change_30d, 
                             nasdaq_change_30d, real_yield_change_30d, etf_flow_7d, 
                             oi_change_7d, funding_annualized, hy_spread_curr):
    cases = []
    
    # Case A: DXY ↑, Real Yield ↑, Nasdaq ↓, BTC ↑
    if (dxy_change_30d is not None and dxy_change_30d > 0) and \
       (real_yield_change_30d is not None and real_yield_change_30d > 0) and \
       (nasdaq_change_30d is not None and nasdaq_change_30d < 0) and \
       (btc_change_30d is not None and btc_change_30d > 0):
        cases.append({
            "case_id": "Case A",
            "name": "Asset Class Divergence",
            "message": "BTC is diverging from traditional risk assets (showing decorrelation and idiosyncratic capital absorption despite dollar and real yield headwinds)."
        })

    # Case B: DXY ↓, Real Yield ↓, ETF Flow ↑, BTC ↑
    if (dxy_change_30d is not None and dxy_change_30d < 0) and \
       (real_yield_change_30d is not None and real_yield_change_30d < 0) and \
       (etf_flow_7d is not None and etf_flow_7d > 0) and \
       (btc_change_7d is not None and btc_change_7d > 0):
        cases.append({
            "case_id": "Case B",
            "name": "Macro & Spot Flow Alignment",
            "message": "Macro liquidity tailwinds and institutional spot ETF capital flows are strongly aligned in supporting crypto assets."
        })

    # Case C: BTC ↑, ETF Flow ↓, OI ↑, Funding ↑
    if (btc_change_7d is not None and btc_change_7d > 0) and \
       (etf_flow_7d is not None and etf_flow_7d < 0) and \
       (oi_change_7d is not None and oi_change_7d > 5) and \
       (funding_annualized is not None and funding_annualized > 10):
        cases.append({
            "case_id": "Case C",
            "name": "Leverage vs Spot Flow Divergence",
            "message": "BTC upside is accompanied by increasing derivatives leverage while spot ETF net flow is weakening (fragile leverage-driven structure)."
        })

    # Case D: Credit spread widening while risk assets expand
    if (hy_spread_curr is not None and hy_spread_curr > 3.8) and \
       (nasdaq_change_30d is not None and nasdaq_change_30d > 2):
        cases.append({
            "case_id": "Case D",
            "name": "Credit vs Equity Divergence",
            "message": "High yield credit spreads indicate underlying financial stress while equity indices remain buoyant."
        })

    if not cases:
        cases.append({
            "case_id": "Neutral",
            "name": "Regime Congruence",
            "message": "Cross-asset relationships and liquidity indicators are operating within standard historical correlation bands without acute divergence."
        })

    return cases

def main():
    print("Starting Macro Daily Dashboard pipeline...")
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
        future_market = executor.submit(get_market_assets)
        future_fng = executor.submit(get_crypto_sentiment, prev_data)
        future_etf = executor.submit(get_btc_etf_flows)
        future_stable = executor.submit(get_stablecoins_data)
        future_lev = executor.submit(get_crypto_leverage)
        future_fred = executor.submit(get_fred_series_all)

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
    fedfunds_obs = fred_data.get("FEDFUNDS", [])

    # Process US 10Y
    u10y_curr, u10y_pct, u10y_diff = calc_changes(us10y_obs)
    us10y = {
        "current": u10y_curr,
        "change_1d_bp": round(u10y_diff.get(1, 0) * 100, 1) if u10y_diff.get(1) is not None else None,
        "change_7d_bp": round(u10y_diff.get(7, 0) * 100, 1) if u10y_diff.get(7) is not None else None,
        "change_30d_bp": round(u10y_diff.get(30, 0) * 100, 1) if u10y_diff.get(30) is not None else None,
        "percentile_1y": calc_percentile([o["value"] for o in us10y_obs[-260:]], u10y_curr) if us10y_obs else None,
        "updated": us10y_obs[-1]["date"] if us10y_obs else None,
        "frequency": "Daily",
        "source": "FRED (Federal Reserve)",
        "status": "ok" if us10y_obs else "unavailable",
        "history": us10y_obs[-90:] if us10y_obs else []
    }

    # Process US 2Y
    u2y_curr, u2y_pct, u2y_diff = calc_changes(us2y_obs)
    us2y = {
        "current": u2y_curr,
        "change_1d_bp": round(u2y_diff.get(1, 0) * 100, 1) if u2y_diff.get(1) is not None else None,
        "change_7d_bp": round(u2y_diff.get(7, 0) * 100, 1) if u2y_diff.get(7) is not None else None,
        "change_30d_bp": round(u2y_diff.get(30, 0) * 100, 1) if u2y_diff.get(30) is not None else None,
        "percentile_1y": calc_percentile([o["value"] for o in us2y_obs[-260:]], u2y_curr) if us2y_obs else None,
        "updated": us2y_obs[-1]["date"] if us2y_obs else None,
        "frequency": "Daily",
        "source": "FRED (Federal Reserve)",
        "status": "ok" if us2y_obs else "unavailable",
        "history": us2y_obs[-90:] if us2y_obs else []
    }

    # Process 10Y-2Y Spread
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
        "source": "FRED / Calculated",
        "status": "ok" if spread_obs else "unavailable",
        "history": spread_obs[-90:] if spread_obs else []
    }

    # Process 10Y Real Yield (DFII10)
    dfii_curr, dfii_pct, dfii_diff = calc_changes(dfii10_obs)
    us10y_real = {
        "current": dfii_curr,
        "change_1d_bp": round(dfii_diff.get(1, 0) * 100, 1) if dfii_diff.get(1) is not None else None,
        "change_7d_bp": round(dfii_diff.get(7, 0) * 100, 1) if dfii_diff.get(7) is not None else None,
        "change_30d_bp": round(dfii_diff.get(30, 0) * 100, 1) if dfii_diff.get(30) is not None else None,
        "percentile_1y": calc_percentile([o["value"] for o in dfii10_obs[-260:]], dfii_curr) if dfii10_obs else None,
        "updated": dfii10_obs[-1]["date"] if dfii10_obs else None,
        "frequency": "Daily",
        "source": "FRED (Federal Reserve)",
        "status": "ok" if dfii10_obs else "unavailable",
        "history": dfii10_obs[-90:] if dfii10_obs else []
    }

    # Process Fed Balance Sheet (WALCL)
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
            "frequency": "Weekly",
            "source": "Federal Reserve (H.4.1)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"] / 1e6, 2)} for o in walcl_obs[-52:]]
        }

    # Process TGA (WDTGAL)
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
            "source": "U.S. Treasury / FRED",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"] / 1000, 1)} for o in wdtgal_obs[-52:]]
        }

    # Process RRP (RRPONTSYD)
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
            "source": "Federal Reserve (NY Fed / FRED)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"], 2)} for o in rrp_obs[-90:]]
        }

    # Process Bank Reserves (WRBWFRBL)
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
            "frequency": "Weekly",
            "source": "Federal Reserve (FRED)",
            "status": "ok",
            "history": [{"date": o["date"], "value": round(o["value"] / 1000, 1)} for o in reserves_obs[-52:]]
        }

    # Compute US Liquidity Proxy: Fed Balance Sheet - TGA - RRP
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
                "current_tril": curr_liq,
                "change_7d": c7d,
                "change_30d": c30d,
                "formula": "Fed Balance Sheet - TGA - RRP",
                "is_derived": True,
                "updated": liq_history[-1]["date"],
                "frequency": "Weekly",
                "source": "Derived (Fed + Treasury Data)",
                "status": "ok",
                "history": liq_history[-52:]
            }

    # Process High Yield Credit Spread
    hy_spread = None
    if hy_obs:
        hy_curr, hy_pct, hy_diff = calc_changes(hy_obs)
        c1w_bp = round(hy_diff.get(5, 0) * 100, 1) if hy_diff.get(5) is not None else None
        c1m_bp = round(hy_diff.get(22, 0) * 100, 1) if hy_diff.get(22) is not None else None
        hy_spread = {
            "current": hy_curr,
            "change_1w_bp": c1w_bp,
            "change_1m_bp": c1m_bp,
            "percentile_1y": calc_percentile([o["value"] for o in hy_obs[-260:]], hy_curr),
            "updated": hy_obs[-1]["date"],
            "frequency": "Daily",
            "source": "ICE BofA / FRED",
            "status": "ok",
            "history": hy_obs[-90:]
        }

    # Process Fed Funds
    fed_funds = None
    if fedfunds_obs:
        fed_funds = {
            "current": fedfunds_obs[-1]["value"],
            "updated": fedfunds_obs[-1]["date"],
            "frequency": "Monthly",
            "source": "Federal Reserve",
            "status": "ok",
            "history": fedfunds_obs[-24:]
        }

    # Process Treasury Bill Curve (Expected Rate Path: 1M, 3M, 6M, 12M)
    dgs1mo_obs = fred_data.get("DGS1MO", [])
    dgs3mo_obs = fred_data.get("DGS3MO", [])
    dgs6mo_obs = fred_data.get("DGS6MO", [])
    dgs1_obs = fred_data.get("DGS1", [])

    rate_path = {
        "m1": dgs1mo_obs[-1]["value"] if dgs1mo_obs else None,
        "m3": dgs3mo_obs[-1]["value"] if dgs3mo_obs else None,
        "m6": dgs6mo_obs[-1]["value"] if dgs6mo_obs else None,
        "m12": dgs1_obs[-1]["value"] if dgs1_obs else None,
        "updated": dgs1mo_obs[-1]["date"] if dgs1mo_obs else None,
        "frequency": "Daily",
        "source": "FRED (U.S. Treasury Curve)",
        "status": "ok"
    }

    # Regime synthesis
    macro_regime = build_macro_regime(
        usd_pct_30d=dxy_data["change_30d"] if dxy_data else None,
        dxy_p1y=dxy_data["percentile_1y"] if dxy_data else None,
        real_yield_change_30d=us10y_real["change_30d_bp"] if us10y_real else None,
        liq_proxy_change_30d=us_liquidity_proxy["change_30d"] if us_liquidity_proxy else None,
        spx_change_30d=spx_data["change_30d"] if spx_data else None,
        vix_curr=vix_data["current"] if vix_data else None,
        hy_spread_curr=hy_spread["current"] if hy_spread else None,
        hy_change_1m=hy_spread["change_1m_bp"] if hy_spread else None,
        etf_flow_7d=etf_flows["flow_7d_mil"] if etf_flows else None,
        funding_annualized=crypto_leverage.get("funding_rate", {}).get("annualized_pct") if crypto_leverage.get("funding_rate") else None,
        oi_change_7d=crypto_leverage.get("open_interest", {}).get("change_7d") if crypto_leverage.get("open_interest") else None
    )

    # Divergence synthesis
    market_divergence = detect_market_divergence(
        regime=macro_regime,
        btc_change_7d=btc_data["change_7d"] if btc_data else None,
        btc_change_30d=btc_data["change_30d"] if btc_data else None,
        dxy_change_30d=dxy_data["change_30d"] if dxy_data else None,
        nasdaq_change_30d=nasdaq_data["change_30d"] if nasdaq_data else None,
        real_yield_change_30d=us10y_real["change_30d_bp"] if us10y_real else None,
        etf_flow_7d=etf_flows["flow_7d_mil"] if etf_flows else None,
        oi_change_7d=crypto_leverage.get("open_interest", {}).get("change_7d") if crypto_leverage.get("open_interest") else None,
        funding_annualized=crypto_leverage.get("funding_rate", {}).get("annualized_pct") if crypto_leverage.get("funding_rate") else None,
        hy_spread_curr=hy_spread["current"] if hy_spread else None
    )

    output = {
        "updated_at": now_utc.isoformat(),
        "macro_regime": macro_regime,
        "market_divergence": market_divergence,
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
            "hy_spread": hy_spread
        },
        "rates": {
            "us10y": us10y,
            "us2y": us2y,
            "us10y_real": us10y_real,
            "spread_10y_2y": spread_10y_2y,
            "fed_funds": fed_funds,
            "rate_path": rate_path
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
            "btc_exchange_netflow": crypto_leverage.get("exchange_netflow")
        },
        "crypto_leverage": crypto_leverage,
        "commodities": {
            "gold": gold_data,
            "crude_oil": oil_data
        }
    }

    os.makedirs("public", exist_ok=True)
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Success: public/data.json generated with complete macro & crypto dataset.")

if __name__ == "__main__":
    main()
