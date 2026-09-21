"""
Risk Engine & Quant Analytical Module for Macro-Dashboard.
Computes multi-dimensional risk regimes, rolling percentiles, z-scores,
transmission chains, rule-based drivers, and liquidity risk calendar.
"""

import math
from datetime import datetime, timezone, timedelta
import calendar

def calc_stats(values, lookback=252):
    """
    Computes rolling percentile and z-score over a lookback window (default 252 trading days).
    Returns (current, percentile, zscore, mean, std) or (None, ...) if insufficient data.
    """
    if not values:
        return None, None, None, None, None
    
    # Filter valid floats
    valid_vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid_vals:
        return None, None, None, None, None
    
    curr = valid_vals[-1]
    window_vals = valid_vals[-lookback:]
    n = len(window_vals)
    
    if n < 20:
        return curr, None, None, None, None
    
    # Percentile: % of historical values <= current
    count_le = sum(1 for v in window_vals if v <= curr)
    percentile = round((count_le / n) * 100.0, 1)
    
    mean = sum(window_vals) / n
    variance = sum((v - mean) ** 2 for v in window_vals) / (n - 1 if n > 1 else 1)
    std = math.sqrt(variance)
    
    zscore = round((curr - mean) / std, 2) if std > 1e-6 else 0.0
    return curr, percentile, zscore, round(mean, 4), round(std, 4)

def calc_changes(series, lookbacks=(1, 5, 20)):
    """
    Calculates differences (diff) and percent changes (pct) for lookbacks in a chronological series.
    series is a list of {"date": ..., "value": float} or floats.
    """
    diffs = {}
    pcts = {}
    if not series:
        return None, diffs, pcts
    
    vals = [s["value"] if isinstance(s, dict) else s for s in series if s is not None]
    if not vals:
        return None, diffs, pcts
    
    curr = vals[-1]
    n = len(vals)
    for lb in lookbacks:
        if n > lb:
            prev = vals[-1 - lb]
            if prev is not None:
                d = round(curr - prev, 4)
                p = round(((curr - prev) / abs(prev)) * 100.0, 2) if prev != 0 else 0.0
                diffs[lb] = d
                pcts[lb] = p
            else:
                diffs[lb] = None
                pcts[lb] = None
        else:
            diffs[lb] = None
            pcts[lb] = None
    return curr, diffs, pcts

def get_cme_front_month_expiry(ref_date=None):
    """
    CME Bitcoin futures expire on the last Friday of the contract month at 4:00 PM London time.
    Calculates the front-month expiration date relative to ref_date.
    """
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).date()
    elif isinstance(ref_date, datetime):
        ref_date = ref_date.date()
    
    year, month = ref_date.year, ref_date.month
    
    def last_friday(y, m):
        last_day = calendar.monthrange(y, m)[1]
        for d in range(last_day, 0, -1):
            dt = datetime(y, m, d).date()
            if dt.weekday() == 4:  # Friday
                return dt
        return datetime(y, m, last_day).date()
    
    exp = last_friday(year, month)
    if ref_date >= exp:
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        exp = last_friday(year, month)
    
    days_to_expiry = max(1, (exp - ref_date).days)
    return exp.strftime("%Y-%m-%d"), days_to_expiry

def evaluate_funding_stress(sofr_obs, iorb_obs, effr_obs, dfedtaru_obs):
    """
    P0 Funding Stress Module:
    SOFR, IORB, EFFR, SOFR-IORB Spread, SOFR-EFFR Spread, SOFR Daily Change.
    Rolling percentile (252D), rolling z-score.
    State Machine: NORMAL, WATCH, STRESS, HIGH_STRESS.
    """
    drivers = []
    sofr_curr = sofr_obs[-1]["value"] if sofr_obs else None
    iorb_curr = iorb_obs[-1]["value"] if iorb_obs else None
    effr_curr = effr_obs[-1]["value"] if effr_obs else None
    target_upper = dfedtaru_obs[-1]["value"] if dfedtaru_obs else None
    
    if sofr_curr is None or iorb_curr is None:
        return {
            "sofr": sofr_curr,
            "iorb": iorb_curr,
            "effr": effr_curr,
            "sofr_iorb_spread_bp": None,
            "sofr_effr_spread_bp": None,
            "sofr_change_1d_bp": None,
            "spread_change_5d_bp": None,
            "spread_change_20d_bp": None,
            "percentile": None,
            "zscore": None,
            "status": "INSUFFICIENT_HISTORY",
            "severity": 0,
            "confidence": 0.0,
            "drivers": ["资金市场关键利率数据缺失"]
        }
    
    # Align SOFR and IORB by date
    iorb_map = {o["date"]: o["value"] for o in iorb_obs}
    effr_map = {o["date"]: o["value"] for o in effr_obs} if effr_obs else {}
    
    spread_series = []
    spread_effr_series = []
    for o in sofr_obs:
        dt = o["date"]
        s_val = o["value"]
        if dt in iorb_map:
            # basis points (SOFR - IORB) * 100
            spread_bp = round((s_val - iorb_map[dt]) * 100.0, 2)
            spread_series.append({"date": dt, "value": spread_bp})
        if dt in effr_map:
            effr_bp = round((s_val - effr_map[dt]) * 100.0, 2)
            spread_effr_series.append({"date": dt, "value": effr_bp})
    
    spread_vals = [s["value"] for s in spread_series]
    curr_spread_bp, spread_pct, spread_z, _, _ = calc_stats(spread_vals, 252)
    _, spread_diffs, _ = calc_changes(spread_series, lookbacks=(1, 5, 20))
    
    sofr_vals = [s["value"] for s in sofr_obs]
    _, sofr_diffs, _ = calc_changes(sofr_obs, lookbacks=(1, 5, 20))
    sofr_1d_bp = round(sofr_diffs.get(1, 0.0) * 100.0, 1) if sofr_diffs.get(1) is not None else None
    
    # SOFR daily change distribution percentile
    sofr_daily_changes = [
        round((sofr_vals[i] - sofr_vals[i-1]) * 100.0, 2)
        for i in range(1, len(sofr_vals))
    ]
    _, sofr_chg_pct, _, _, _ = calc_stats(sofr_daily_changes, 252)
    
    effr_spread_bp = round((sofr_curr - effr_curr) * 100.0, 2) if effr_curr is not None else None
    
    # State Machine
    status = "NORMAL"
    severity = 0
    
    # Check extreme stress first
    if spread_pct is not None and spread_pct >= 99.0 and (sofr_1d_bp is not None and sofr_1d_bp > 3.0):
        status = "HIGH_STRESS"
        severity = 3
        drivers.append(f"SOFR-IORB利差处于历史99%极端高位 ({curr_spread_bp:+.1f}bp)")
        drivers.append("SOFR隔夜融资利率正在急剧跳升")
        if target_upper is not None and effr_curr is not None and (target_upper - effr_curr) <= 0.03:
            drivers.append("EFFR向美联储目标区间上沿显著逼近")
    elif (spread_pct is not None and spread_pct >= 95.0) or (sofr_chg_pct is not None and sofr_chg_pct >= 95.0):
        status = "STRESS"
        severity = 2
        if spread_pct is not None and spread_pct >= 95.0:
            drivers.append(f"SOFR-IORB利差处于历史95%分位承压区间 ({curr_spread_bp:+.1f}bp)")
        if sofr_chg_pct is not None and sofr_chg_pct >= 95.0:
            drivers.append(f"SOFR单日跳升幅度 ({sofr_1d_bp:+.1f}bp) 处于历史95%高位")
    elif (spread_pct is not None and spread_pct >= 90.0) or (spread_diffs.get(5) is not None and spread_diffs.get(5) > 5.0):
        status = "WATCH"
        severity = 1
        drivers.append(f"SOFR-IORB利差走阔至历史90%分位预警区 ({curr_spread_bp:+.1f}bp)")
    else:
        status = "NORMAL"
        severity = 0
        drivers.append(f"SOFR-IORB利差运行于历史常态区间 ({curr_spread_bp:+.1f}bp，历史分位 {spread_pct or 0}%)")
    
    return {
        "sofr": sofr_curr,
        "iorb": iorb_curr,
        "effr": effr_curr,
        "sofr_iorb_spread_bp": curr_spread_bp,
        "sofr_effr_spread_bp": effr_spread_bp,
        "sofr_change_1d_bp": sofr_1d_bp,
        "spread_change_5d_bp": spread_diffs.get(5),
        "spread_change_20d_bp": spread_diffs.get(20),
        "percentile": spread_pct,
        "zscore": spread_z,
        "status": status,
        "severity": severity,
        "confidence": 1.0,
        "drivers": drivers,
        "history": spread_series[-90:] if spread_series else []
    }

def evaluate_move_volatility(move_data):
    """
    P0 MOVE Bond Volatility Engine:
    Level, 5D change, 20D change, rolling 252D percentile, rolling z-score.
    Status: NORMAL, WATCH (>=90%), STRESS (>=95%), HIGH_STRESS (>=99%).
    """
    drivers = []
    if not move_data or move_data.get("current") is None:
        return {
            "current": None,
            "change_1d": None,
            "change_5d": None,
            "change_20d": None,
            "percentile": None,
            "zscore": None,
            "status": "INSUFFICIENT_HISTORY",
            "severity": 0,
            "confidence": 0.0,
            "drivers": ["MOVE债券波动率数据未获取"]
        }
    
    history = move_data.get("history", [])
    vals = [h["value"] for h in history] if history else [move_data["current"]]
    curr, p1y, zscore, _, _ = calc_stats(vals, 252)
    _, diffs, pcts = calc_changes(history, lookbacks=(1, 5, 20))
    
    c5d = pcts.get(5)
    c20d = pcts.get(20)
    
    status = "NORMAL"
    severity = 0
    if p1y is not None and p1y >= 99.0:
        status = "HIGH_STRESS"
        severity = 3
        drivers.append(f"MOVE国债波动率 ({curr:.1f}) 突破历史99%分位，美债市场流动性极端承压")
    elif p1y is not None and p1y >= 95.0:
        status = "STRESS"
        severity = 2
        drivers.append(f"MOVE国债波动率 ({curr:.1f}) 处于历史95%分位高压区间")
    elif (p1y is not None and p1y >= 90.0) or (c5d is not None and c5d >= 15.0):
        status = "WATCH"
        severity = 1
        drivers.append(f"MOVE国债波动率 ({curr:.1f}) 突破90%分位或短期剧烈攀升 (+{c5d or 0}%)")
    else:
        status = "NORMAL"
        severity = 0
        drivers.append(f"MOVE国债波动率 ({curr:.1f}) 处于常态波动区间 (历史分位 {p1y or 0}%)")
        
    return {
        "current": curr,
        "change_1d": diffs.get(1),
        "change_5d": diffs.get(5),
        "change_20d": diffs.get(20),
        "change_5d_pct": c5d,
        "change_20d_pct": c20d,
        "percentile": p1y,
        "zscore": zscore,
        "status": status,
        "severity": severity,
        "confidence": 1.0,
        "drivers": drivers,
        "history": history[-90:] if history else []
    }

def evaluate_rates_and_breakeven(us10y_obs, dfii10_obs, t10yie_obs):
    """
    P0 10Y Breakeven Inflation & Rates Regime:
    Nominal 10Y, 10Y TIPS Real, 10Y Breakeven.
    Changes: 1D, 5D, 20D.
    Regime:
    A. Growth / Real Yield: Real Yield ↑, Breakeven →
    B. Inflation: Real Yield →, Breakeven ↑
    C. Stagflation Risk: Real Yield ↑, Breakeven ↑
    D. Financial Tightening: Real Yield ↑↑, Breakeven ↓
    E. Balanced / Easing
    """
    drivers = []
    u10y_curr, u10y_diffs, _ = calc_changes(us10y_obs, lookbacks=(1, 5, 20))
    real_curr, real_diffs, _ = calc_changes(dfii10_obs, lookbacks=(1, 5, 20))
    be_curr, be_diffs, _ = calc_changes(t10yie_obs, lookbacks=(1, 5, 20))
    
    # If breakeven not directly fetched, derive from Nominal - Real
    if be_curr is None and u10y_curr is not None and real_curr is not None:
        be_curr = round(u10y_curr - real_curr, 2)
    
    nom_c1d_bp = round(u10y_diffs.get(1, 0.0) * 100.0, 1) if u10y_diffs.get(1) is not None else None
    nom_c5d_bp = round(u10y_diffs.get(5, 0.0) * 100.0, 1) if u10y_diffs.get(5) is not None else None
    nom_c20d_bp = round(u10y_diffs.get(20, 0.0) * 100.0, 1) if u10y_diffs.get(20) is not None else None
    
    real_c1d_bp = round(real_diffs.get(1, 0.0) * 100.0, 1) if real_diffs.get(1) is not None else None
    real_c5d_bp = round(real_diffs.get(5, 0.0) * 100.0, 1) if real_diffs.get(5) is not None else None
    real_c20d_bp = round(real_diffs.get(20, 0.0) * 100.0, 1) if real_diffs.get(20) is not None else None
    
    be_c1d_bp = round(be_diffs.get(1, 0.0) * 100.0, 1) if be_diffs.get(1) is not None else None
    be_c5d_bp = round(be_diffs.get(5, 0.0) * 100.0, 1) if be_diffs.get(5) is not None else None
    be_c20d_bp = round(be_diffs.get(20, 0.0) * 100.0, 1) if be_diffs.get(20) is not None else None
    
    # Identify Rates Regime based on 20D/5D trend
    regime = "BALANCED"
    regime_cn = "收益率常态均衡"
    status = "NORMAL"
    severity = 0
    
    r_bp = real_c20d_bp if real_c20d_bp is not None else 0.0
    b_bp = be_c20d_bp if be_c20d_bp is not None else 0.0
    
    if r_bp >= 12.0 and b_bp <= -6.0:
        regime = "FINANCIAL_TIGHTENING"
        regime_cn = "金融条件急剧收紧"
        status = "STRESS"
        severity = 2
        drivers.append(f"实际利率快速飙升 ({r_bp:+.1f}bp) 伴随通胀预期下修，金融条件显著收紧")
    elif r_bp >= 8.0 and b_bp >= 8.0:
        regime = "STAGFLATION_RISK"
        regime_cn = "潜在滞胀风险"
        status = "STRESS"
        severity = 2
        drivers.append(f"实际利率 ({r_bp:+.1f}bp) 与通胀预期 ({b_bp:+.1f}bp) 共振走高，呈现滞胀风险特征")
    elif r_bp >= 8.0 and abs(b_bp) < 8.0:
        regime = "GROWTH_REAL_YIELD"
        regime_cn = "实际利率上行驱动"
        status = "WATCH"
        severity = 1
        drivers.append(f"实际利率上行 (+{r_bp:.1f}bp) 为名义收益率主要驱动，压制高估值风险资产")
    elif abs(r_bp) < 8.0 and b_bp >= 8.0:
        regime = "INFLATION_DRIVEN"
        regime_cn = "通胀预期主导驱动"
        status = "WATCH"
        severity = 1
        drivers.append(f"通胀预期升温 (+{b_bp:.1f}bp) 主导名义收益率上行")
    elif r_bp <= -8.0:
        regime = "REAL_EASING"
        regime_cn = "实际利率趋松改善"
        status = "NORMAL"
        severity = 0
        drivers.append(f"实际利率回落 ({r_bp:.1f}bp)，宏观估值折现压力减轻")
    else:
        regime = "BALANCED"
        regime_cn = "名义/实际利率相对平稳"
        status = "NORMAL"
        severity = 0
        drivers.append("10Y名义与实际利率波动处于历史常态波动区间")
        
    return {
        "nominal_10y": u10y_curr,
        "real_10y": real_curr,
        "breakeven_10y": be_curr,
        "nominal_change_1d_bp": nom_c1d_bp,
        "nominal_change_5d_bp": nom_c5d_bp,
        "nominal_change_20d_bp": nom_c20d_bp,
        "real_change_1d_bp": real_c1d_bp,
        "real_change_5d_bp": real_c5d_bp,
        "real_change_20d_bp": real_c20d_bp,
        "breakeven_change_1d_bp": be_c1d_bp,
        "breakeven_change_5d_bp": be_c5d_bp,
        "breakeven_change_20d_bp": be_c20d_bp,
        "regime": regime,
        "regime_cn": regime_cn,
        "status": status,
        "severity": severity,
        "confidence": 1.0,
        "drivers": drivers
    }

def evaluate_credit_stress(hy_obs, ig_obs=None):
    """
    P0 Credit Stress Enhancement:
    HY OAS level, 1D/5D/20D change, percentile (252D), z-score.
    IG OAS if available.
    """
    drivers = []
    if not hy_obs:
        return {
            "hy_oas": None,
            "ig_oas": None,
            "change_1d_bp": None,
            "change_5d_bp": None,
            "change_20d_bp": None,
            "percentile": None,
            "zscore": None,
            "status": "INSUFFICIENT_HISTORY",
            "severity": 0,
            "confidence": 0.0,
            "drivers": ["信用利差数据未获取"]
        }
    
    hy_vals = [o["value"] for o in hy_obs]
    curr_hy, p1y, zscore, _, _ = calc_stats(hy_vals, 252)
    _, diffs, _ = calc_changes(hy_obs, lookbacks=(1, 5, 20))
    
    c1d_bp = round(diffs.get(1, 0.0) * 100.0, 1) if diffs.get(1) is not None else None
    c5d_bp = round(diffs.get(5, 0.0) * 100.0, 1) if diffs.get(5) is not None else None
    c20d_bp = round(diffs.get(20, 0.0) * 100.0, 1) if diffs.get(20) is not None else None
    
    curr_ig = ig_obs[-1]["value"] if ig_obs else None
    
    status = "NORMAL"
    severity = 0
    if (p1y is not None and p1y >= 95.0) or (c5d_bp is not None and c5d_bp >= 35.0 and (p1y or 0) >= 80.0):
        status = "HIGH_STRESS"
        severity = 3
        drivers.append(f"高收益信用利差 ({curr_hy:.2f}%) 处于历史95%高位或短期急剧走阔 (+{c5d_bp}bp)")
    elif (p1y is not None and p1y >= 90.0) or (c5d_bp is not None and c5d_bp >= 25.0):
        status = "STRESS"
        severity = 2
        drivers.append(f"高收益信用利差 5日走阔 {c5d_bp:+.1f}bp，债务违约风险溢价攀升")
    elif (p1y is not None and p1y >= 80.0) or (c5d_bp is not None and c5d_bp >= 15.0):
        status = "WATCH"
        severity = 1
        drivers.append(f"信用利差出现走阔迹象 (5日变化: {c5d_bp:+.1f}bp)")
    else:
        status = "NORMAL"
        severity = 0
        drivers.append(f"高收益信用利差 ({curr_hy:.2f}%) 保持收窄健康，企业融资溢价偏低")
        
    return {
        "hy_oas": curr_hy,
        "ig_oas": curr_ig,
        "change_1d_bp": c1d_bp,
        "change_5d_bp": c5d_bp,
        "change_20d_bp": c20d_bp,
        "percentile": p1y,
        "zscore": zscore,
        "status": status,
        "severity": severity,
        "confidence": 1.0,
        "drivers": drivers,
        "history": hy_obs[-90:] if hy_obs else []
    }

def evaluate_cross_asset_stress(market_assets, move_result, credit_result, rates_result):
    """
    P1 Cross-Asset Stress Engine & Liquidity Liquidation Detection:
    Monitors VIX, MOVE, DXY, USDJPY, S&P 500, Nasdaq 100, Gold, BTC, 10Y Treasury, HY OAS.
    Identifies 'LIQUIDITY_LIQUIDATION' / Dash-for-cash candidate.
    """
    drivers = []
    btc = market_assets.get("BTC", {})
    spx = market_assets.get("SPX", {})
    ndx = market_assets.get("NASDAQ", {})
    vix = market_assets.get("VIX", {})
    gold = market_assets.get("GOLD", {})
    dxy = market_assets.get("DXY", {})
    usdjpy = market_assets.get("USDJPY", {})
    
    btc_c5 = btc.get("change_7d")  # 7D / 5D proxy
    ndx_c5 = ndx.get("change_7d")
    spx_c5 = spx.get("change_7d")
    gold_c5 = gold.get("change_7d")
    dxy_c5 = dxy.get("change_7d")
    vix_curr = vix.get("current")
    vix_c5 = vix.get("change_7d")
    move_pct = move_result.get("percentile")
    move_c5 = move_result.get("change_5d_pct")
    hy_c5_bp = credit_result.get("change_5d_bp")
    u10y_c5_bp = rates_result.get("nominal_change_5d_bp")
    usdjpy_c5 = usdjpy.get("change_7d")
    
    # Dash-for-cash / Liquidity Liquidation signals
    signals = 0
    signal_details = []
    
    if btc_c5 is not None and btc_c5 < -4.0:
        signals += 1
        signal_details.append(f"BTC短期重挫 ({btc_c5}%)")
    if ndx_c5 is not None and ndx_c5 < -2.5:
        signals += 1
        signal_details.append(f"纳斯达克100快速走弱 ({ndx_c5}%)")
    if gold_c5 is not None and gold_c5 < -1.5:
        signals += 1
        signal_details.append(f"黄金避险资产遭抛售变现 ({gold_c5}%)")
    if dxy_c5 is not None and dxy_c5 > 0.6:
        signals += 1
        signal_details.append(f"美元指数急升吸水 (+{dxy_c5}%)")
    if u10y_c5_bp is not None and u10y_c5_bp > 8.0:
        signals += 1
        signal_details.append(f"10Y美债收益率攀升/国债抛售 (+{u10y_c5_bp}bp)")
    if hy_c5_bp is not None and hy_c5_bp > 15.0:
        signals += 1
        signal_details.append(f"高收益债利差走阔 (+{hy_c5_bp}bp)")
    if (move_c5 is not None and move_c5 > 10.0) or (move_pct is not None and move_pct >= 90.0):
        signals += 1
        signal_details.append("MOVE债券波动率突破高位")
    if vix_curr is not None and vix_curr > 22.0:
        signals += 1
        signal_details.append(f"VIX恐慌指数高位 ({vix_curr})")
        
    is_liquidation = (signals >= 5)
    
    # Yen Carry Trade Unwind check
    if usdjpy_c5 is not None and usdjpy_c5 < -2.0 and (vix_c5 is not None and vix_c5 > 15.0):
        drivers.append(f"日元急剧升值 ({usdjpy_c5}%) 伴随波动率上升，警惕日元套息交易平仓去杠杆")
    
    status = "NORMAL"
    severity = 0
    if is_liquidation:
        status = "LIQUIDITY_LIQUIDATION"
        severity = 4
        drivers.append(f"触发跨资产流动性踩踏特征 (满足 {signals}/8 项同步抛售条件)")
        drivers.extend(signal_details[:4])
    elif signals >= 3 or (vix_curr and vix_curr > 24.0) or (move_pct and move_pct >= 95.0):
        status = "STRESS"
        severity = 2
        drivers.append("多类金融资产同步释放波动压力")
        drivers.extend(signal_details[:3])
    elif signals >= 2 or (vix_curr and vix_curr > 19.5):
        status = "WATCH"
        severity = 1
        drivers.append("个别跨市场指标显露局部压力")
        drivers.extend(signal_details[:2])
    else:
        status = "NORMAL"
        severity = 0
        drivers.append("跨资产风险溢价与资产联动处于平稳常态")
        
    return {
        "status": status,
        "severity": severity,
        "confidence": 1.0,
        "liquidation_signals": signals,
        "is_liquidity_liquidation": is_liquidation,
        "drivers": drivers,
        "vix": vix,
        "usdjpy": usdjpy,
        "dxy": dxy
    }

def evaluate_crypto_structural_risk(etf_flows, cme_basis_data, crypto_leverage, stablecoins, btc_spot_vol):
    """
    P1 Crypto Structural Risk Module:
    ETF net flow, CME Basis, OI/Spot Volume ratio, Funding rate, Liquidations, Stablecoin supply.
    """
    drivers = []
    etf_7d = etf_flows.get("flow_7d_mil") if etf_flows else None
    etf_today = etf_flows.get("today_net_mil") if etf_flows else None
    
    funding_info = crypto_leverage.get("funding_rate", {}) if crypto_leverage else {}
    funding_ann = funding_info.get("annualized_pct") if funding_info else None
    
    oi_info = crypto_leverage.get("open_interest", {}) if crypto_leverage else {}
    oi_curr = oi_info.get("total_usd_bil") if oi_info else None
    oi_c7 = oi_info.get("change_7d") if oi_info else None
    
    # OI / Spot Volume ratio
    oi_spot_ratio = None
    spot_vol_bil = btc_spot_vol.get("volume_usd_bil") if btc_spot_vol else None
    if oi_curr is not None and spot_vol_bil and spot_vol_bil > 0:
        oi_spot_ratio = round(oi_curr / spot_vol_bil, 2)
    
    # CME Basis
    cme_basis_usd = cme_basis_data.get("basis_usd") if cme_basis_data else None
    cme_basis_ann = cme_basis_data.get("annualized_pct") if cme_basis_data else None
    basis_compressed = cme_basis_ann is not None and cme_basis_ann < 4.0
    
    # Stablecoin Supply & Share
    stable_mcap = stablecoins.get("total_mcap_bil") if stablecoins else None
    stable_c7d = stablecoins.get("change_7d") if stablecoins else None
    stable_c30d = stablecoins.get("change_30d") if stablecoins else None
    usdt_mcap = stablecoins.get("usdt_mcap_bil") if stablecoins else None
    usdc_mcap = stablecoins.get("usdc_mcap_bil") if stablecoins else None
    
    usdt_share = round((usdt_mcap / stable_mcap) * 100.0, 1) if (usdt_mcap and stable_mcap) else None
    usdc_share = round((usdc_mcap / stable_mcap) * 100.0, 1) if (usdc_mcap and stable_mcap) else None
    stable_contracting = (stable_c7d is not None and stable_c7d < -0.3) or (stable_c30d is not None and stable_c30d < -1.0)
    
    # Structural Risk Assessment
    status = "NORMAL"
    severity = 0
    
    fragility_points = 0
    if basis_compressed:
        fragility_points += 1
        drivers.append(f"CME期现基差收窄至低位 ({cme_basis_ann}%)，机构套利需求降温")
    if etf_7d is not None and etf_7d < -100.0:
        fragility_points += 1
        drivers.append(f"BTC现货ETF过去7日呈现持续净流出 (${etf_7d}M)")
    if oi_c7 is not None and oi_c7 > 12.0 and (funding_ann is not None and funding_ann > 15.0):
        fragility_points += 1
        drivers.append(f"未平仓合约快速攀升 (+{oi_c7}%) 伴随资金费率过热 ({funding_ann}%)")
    if oi_spot_ratio is not None and oi_spot_ratio > 10.0:
        fragility_points += 1
        drivers.append(f"合约持仓/现货成交量比率偏高 ({oi_spot_ratio}x)，显示杠杆交易活跃度主导")
    if stable_contracting:
        fragility_points += 1
        drivers.append(f"全网稳定币总市值收缩 (30日净变动: {stable_c30d}%)")
        
    if fragility_points >= 3 or (basis_compressed and etf_7d is not None and etf_7d < -150.0 and (oi_c7 or 0) > 10.0):
        status = "HIGH_STRESS"
        severity = 3
        drivers.insert(0, "Crypto呈现典型结构性高杠杆脆弱特征")
    elif fragility_points >= 2:
        status = "FRAGILE"
        severity = 2
        drivers.insert(0, "加密市场局部存在杠杆与现货资金背离")
    elif fragility_points == 1:
        status = "WATCH"
        severity = 1
    else:
        status = "NORMAL"
        severity = 0
        drivers.append("加密衍生品杠杆结构与现货资金流入保持健康平衡")
        
    return {
        "status": status,
        "severity": severity,
        "confidence": 1.0,
        "etf_flow_7d_mil": etf_7d,
        "cme_basis": cme_basis_data,
        "oi_usd_bil": oi_curr,
        "oi_spot_ratio": oi_spot_ratio,
        "spot_vol_24h_bil": spot_vol_bil,
        "funding_annualized": funding_ann,
        "stablecoin_supply_bil": stable_mcap,
        "stablecoin_change_7d": stable_c7d,
        "stablecoin_change_30d": stable_c30d,
        "usdt_share_pct": usdt_share,
        "usdc_share_pct": usdc_share,
        "drivers": drivers
    }

def build_liquidity_calendar(ref_date=None):
    """
    P1 Liquidity Event Calendar:
    Monitors upcoming 30 days for FOMC, CPI, NFP, Treasury Auctions/Settlements, Quarter-end, Tax dates.
    Outputs upcoming events with days remaining (T-x days) and potential market impacts.
    """
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).date()
    elif isinstance(ref_date, datetime):
        ref_date = ref_date.date()
    
    # Canonical macro events schedule for late 2024 - 2026
    candidate_events = [
        # 2026 FOMC dates
        {"date": "2026-09-16", "name": "FOMC 利率决议与经济预测(SEP)", "impact": "利率 / 美元 / 全球宏观资产"},
        {"date": "2026-09-30", "name": "Q3 季度末流动性与银行监管窗口", "impact": "资金市场 / 回购利率 / 美元流动性"},
        {"date": "2026-10-02", "name": "美国 9月 非农就业报告 (NFP)", "impact": "美债收益率 / 美元 / 风险偏好"},
        {"date": "2026-10-14", "name": "美国 9月 CPI 通胀数据公布", "impact": "通胀预期 / 10Y美债 / 降息路径"},
        {"date": "2026-10-15", "name": "美国财政部月中发债缴款结算", "impact": "TGA账户 / 银行准备金流动性"},
        {"date": "2026-10-30", "name": "财政部 Q4 季度再融资发债计划(QRA)", "impact": "美债供给 / 期限溢价 / 净流动性"},
        {"date": "2026-11-05", "name": "FOMC 11月 利率决议", "impact": "基准利率 / 缩表节奏 / 全球流动性"},
        {"date": "2026-11-13", "name": "美国 10月 CPI 通胀数据", "impact": "通胀溢价 / 实际利率"},
        {"date": "2026-12-15", "name": "美国企业第四季度预缴税征收日", "impact": "TGA余额激增 / 银行流动性抽水"},
        {"date": "2026-12-16", "name": "FOMC 12月 利率决议与点阵图", "impact": "2027年宏观路径预估"},
        {"date": "2026-12-31", "name": "年终跨年资产负债表与监管结算", "impact": "隔夜资金市场 / 回购利差极端敏感期"}
    ]
    
    upcoming = []
    for ev in candidate_events:
        ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        diff = (ev_date - ref_date).days
        if 0 <= diff <= 35:
            upcoming.append({
                "event": ev["name"],
                "date": ev["date"],
                "days_remaining": diff,
                "days_label": "今日" if diff == 0 else f"T-{diff}天",
                "potential_impact": ev["impact"]
            })
    
    upcoming.sort(key=lambda x: x["days_remaining"])
    
    # Fallback if beyond predefined schedule: compute nearest Treasury & month-end
    if not upcoming:
        # compute next month end
        last_day = calendar.monthrange(ref_date.year, ref_date.month)[1]
        m_end = datetime(ref_date.year, ref_date.month, last_day).date()
        diff = (m_end - ref_date).days
        upcoming.append({
            "event": "月末金融机构资产负债表例行结算",
            "date": m_end.strftime("%Y-%m-%d"),
            "days_remaining": diff,
            "days_label": f"T-{diff}天",
            "potential_impact": "隔夜回购资金 / 银行流动性"
        })
    
    next_ev = upcoming[0] if upcoming else None
    return {
        "next_event": next_ev,
        "upcoming_events": upcoming
    }

def synthesize_macro_regime(funding_res, rates_res, credit_res, cross_asset_res, crypto_res, liquidity_res):
    """
    Synthesizes overall Macro Risk Regime, Risk Level, Drivers, and Transmission Chain.
    Overall Regime:
    - NORMAL
    - LIQUIDITY_TIGHTENING
    - MARKET_FRAGILITY
    - CROSS_ASSET_STRESS
    - FORCED_DELEVERAGING
    - CRISIS_CANDIDATE
    
    Risk Level:
    - NORMAL
    - WATCH
    - STRESS
    - HIGH STRESS
    - CRISIS CANDIDATE
    """
    f_sev = funding_res.get("severity", 0)
    r_sev = rates_res.get("severity", 0)
    c_sev = credit_res.get("severity", 0)
    x_sev = cross_asset_res.get("severity", 0)
    cr_sev = crypto_res.get("severity", 0)
    
    if isinstance(liquidity_res, dict):
        c30 = liquidity_res.get("change_30d")
        if c30 is not None:
            liq_status = "Expanding" if c30 > 0.4 else ("Contracting" if c30 < -0.4 else "Neutral")
        else:
            liq_status = liquidity_res.get("state", "Neutral")
    else:
        liq_status = "Neutral"
    is_liq_contracting = (liq_status == "Contracting")
    
    max_sev = max(f_sev, r_sev, c_sev, x_sev, cr_sev)
    
    # Risk Level
    if max_sev >= 4 or cross_asset_res.get("is_liquidity_liquidation"):
        risk_level = "CRISIS CANDIDATE"
    elif max_sev == 3:
        risk_level = "HIGH STRESS"
    elif max_sev == 2:
        risk_level = "STRESS"
    elif max_sev == 1:
        risk_level = "WATCH"
    else:
        risk_level = "NORMAL"
        
    # Overall Macro Regime
    if cross_asset_res.get("is_liquidity_liquidation"):
        overall_regime = "FORCED_DELEVERAGING"
        regime_title = "强制去杠杆 / 跨资产踩踏"
    elif x_sev >= 2 or (c_sev >= 2 and r_sev >= 2):
        overall_regime = "CROSS_ASSET_STRESS"
        regime_title = "跨资产风险溢价传导承压"
    elif f_sev >= 2:
        overall_regime = "MARKET_FRAGILITY"
        regime_title = "资金市场流动性摩擦脆弱"
    elif is_liq_contracting and (r_sev >= 1 or rates_res.get("regime") == "FINANCIAL_TIGHTENING"):
        overall_regime = "LIQUIDITY_TIGHTENING"
        regime_title = "央行流动性抽水与金融条件收紧"
    elif cr_sev >= 2:
        overall_regime = "MARKET_FRAGILITY"
        regime_title = "加密衍生结构局部脆弱"
    elif max_sev == 1:
        overall_regime = "MARKET_FRAGILITY"
        regime_title = "局部指标进入观察预警"
    else:
        overall_regime = "NORMAL"
        regime_title = "宏观流动性与风险溢价平稳"
        
    # Aggregate prioritized drivers (max 5)
    all_drivers = []
    for mod in [cross_asset_res, funding_res, rates_res, credit_res, crypto_res]:
        for d in mod.get("drivers", []):
            if "常态" not in d and "平稳" not in d and "未获取" not in d:
                all_drivers.append(d)
    
    if not all_drivers:
        all_drivers = [
            "美元流动性与资金市场回购利率运行平稳",
            "美债收益率曲线与信用利差处于历史合理区间",
            "加密现货资金与衍生品杠杆未见急性结构性脆弱"
        ]
    
    # Transmission Chain evaluation
    # Rates -> Funding -> Credit -> Risk Assets -> Crypto Leverage
    transmission_chain = [
        {
            "stage": "Rates (美债利率/波动)",
            "status": "STRESS" if r_sev >= 2 else ("WATCH" if r_sev == 1 else "NORMAL"),
            "detail": rates_res.get("regime_cn", "常态")
        },
        {
            "stage": "Funding (资金市场回购)",
            "status": funding_res.get("status", "NORMAL"),
            "detail": f"SOFR-IORB {funding_res.get('sofr_iorb_spread_bp', 0):+.1f}bp"
        },
        {
            "stage": "Credit (企业信用利差)",
            "status": credit_res.get("status", "NORMAL"),
            "detail": f"HY利差 {credit_res.get('hy_oas', 0)}%"
        },
        {
            "stage": "Risk Assets (权益与大宗)",
            "status": cross_asset_res.get("status", "NORMAL"),
            "detail": f"跨市场信号 {cross_asset_res.get('liquidation_signals', 0)}项"
        },
        {
            "stage": "Crypto Leverage (加密杠杆结构)",
            "status": crypto_res.get("status", "NORMAL"),
            "detail": f"基差/ETF共振评估"
        }
    ]
    
    return {
        "overall_regime": overall_regime,
        "regime_title": regime_title,
        "risk_level": risk_level,
        "liquidity": {
            "status": "STRESS" if is_liq_contracting else "NORMAL",
            "severity": 1 if is_liq_contracting else 0,
            "confidence": 1.0,
            "drivers": [f"美元净流动性处于{'收缩' if is_liq_contracting else ('扩张' if liq_status == 'Expanding' else '常态平稳')}阶段"]
        },
        "funding": funding_res,
        "rates": rates_res,
        "credit": credit_res,
        "cross_asset": cross_asset_res,
        "crypto": crypto_res,
        "drivers": all_drivers[:5],
        "transmission_chain": transmission_chain
    }
