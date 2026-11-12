"""Nifty Breakout Trader - simple, one-file Streamlit app.

Built on the NR7 + relative-volume + 20-bar-pivot logic (daily, weekly, monthly) and extended into a complete,
easy trade plan: buy trigger, stop, targets, room to the 52-week high, position size, skip rule.
Free end-of-day data from Yahoo Finance. Research tool, not investment advice.
"""
import io
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

st.set_page_config(page_title="Nifty Breakout Trader", page_icon="📈", layout="wide")

LIST_URLS = ["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
             "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"]
UA = {"User-Agent": "Mozilla/5.0"}
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Style presets: one click instead of five sliders. "Strict" = the original V2 defaults.
PRESETS = {
    "Balanced (recommended)": dict(near_pct=5.0, min_rvol=1.5, breakout_rvol=2.5, need_trend=True, need_momentum=True),
    "Strict (original V2 rules)": dict(near_pct=5.0, min_rvol=2.0, breakout_rvol=5.0, need_trend=True, need_momentum=False),
    "Wide (more ideas)": dict(near_pct=7.0, min_rvol=1.2, breakout_rvol=1.8, need_trend=True, need_momentum=False),
}


# =====================================================================================================
# DATA
# =====================================================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def nifty500_list():
    """Official Nifty 500 constituents (two official URLs). Returns DataFrame[Symbol] or None."""
    for url in LIST_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
            if col:
                out = pd.DataFrame({"Symbol": df[col].astype(str).str.strip()}).dropna()
                return out.drop_duplicates("Symbol").sort_values("Symbol").reset_index(drop=True)
        except Exception:
            continue
    return None


def clean_ohlcv(df):
    """Standardise one stock's Yahoo frame: Open/High/Low/Close/Volume, numeric, tz-naive date index, no duplicates."""
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    if not set(OHLCV).issubset(df.columns):
        return None
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce").dropna()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


@st.cache_data(ttl=1800, show_spinner=False)
def download_chunk(tickers, period="5y"):
    """ONE Yahoo call for up to 50 tickers -> {ticker: clean frame}. Sequential batches (not many threads): yfinance keeps
    shared global state, so parallel download() calls can drop or mix results."""
    out = {}
    if yf is None or not tickers:
        return out
    try:
        data = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True, group_by="ticker",
                           threads=True, progress=False, timeout=30)
    except Exception:
        return out
    if data is None or len(data) == 0:
        return out
    multi = isinstance(data.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if multi:
                lv0, lv1 = data.columns.get_level_values(0), data.columns.get_level_values(1)
                sub = data[t] if t in lv0 else (data.xs(t, axis=1, level=1) if t in lv1 else None)
            else:
                sub = data
            frame = clean_ohlcv(sub)
            if frame is not None:
                out[t] = frame
        except Exception:
            continue
    return out


def download_all(symbols, progress=None):
    """Download every symbol in cached chunks of 50, one retry for an empty chunk. Returns (frames, empty_chunks)."""
    frames, empty = {}, 0
    chunks = [symbols[i:i + 50] for i in range(0, len(symbols), 50)]
    for k, ch in enumerate(chunks):
        tick = tuple(s + ".NS" for s in ch)
        got = download_chunk(tick)
        if not got:
            time.sleep(2)
            try:
                download_chunk.clear()
            except Exception:
                pass
            got = download_chunk(tick)
        if not got:
            empty += 1
        for t, f in got.items():
            frames[t[:-3]] = f
        if len(chunks) > 8:
            time.sleep(0.3)                      # polite pacing on big scans (fewer rate-limit errors)
        if progress is not None:
            progress.progress((k + 1) / len(chunks), text=f"Downloading prices {min((k + 1) * 50, len(symbols))}/{len(symbols)}")
    return frames, empty


def drop_unfinished_today(frame):
    """Before 4 PM IST on a weekday, today's daily candle is still forming: drop it so signals use closed candles only."""
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    if now.weekday() < 5 and now.hour < 16 and frame.index[-1].date() == now.date():
        return frame.iloc[:-1]
    return frame


# =====================================================================================================
# INDICATORS + SIGNALS (daily / weekly / monthly)
# =====================================================================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi14(close):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + gain / loss)          # no losses at all -> RSI 100


def resample_complete(daily, rule):
    """Weekly (W-FRI) / monthly (ME) bars, keeping only periods that were COMPLETE on the last daily date."""
    bars = pd.DataFrame({"Open": daily["Open"].resample(rule).first(), "High": daily["High"].resample(rule).max(),
                         "Low": daily["Low"].resample(rule).min(), "Close": daily["Close"].resample(rule).last(),
                         "Volume": daily["Volume"].resample(rule).sum()}).dropna()
    return bars[bars.index <= daily.index[-1]]


def with_indicators(f):
    x = f.copy()
    x["Range"] = x["High"] - x["Low"]
    x["NR7"] = x["Range"] <= x["Range"].rolling(7).min()
    x["RVOL"] = x["Volume"] / x["Volume"].shift(1).rolling(20).mean().replace(0, np.nan)   # today not in its own average
    x["Pivot20"] = x["High"].shift(1).rolling(20).max()
    x["Low20"] = x["Low"].shift(1).rolling(20).min()
    x["EMA20"], x["EMA50"] = ema(x["Close"], 20), ema(x["Close"], 50)
    x["RSI"] = rsi14(x["Close"])
    macd = ema(x["Close"], 12) - ema(x["Close"], 26)
    x["MACD"], x["MACDSig"] = macd, ema(macd, 9)
    pc = x["Close"].shift(1)
    x["ATR"] = pd.concat([x["High"] - x["Low"], (x["High"] - pc).abs(), (x["Low"] - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
    x["High252"] = x["High"].shift(1).rolling(252, min_periods=120).max()
    return x


def read_frame(frame, min_bars, p):
    """Signal flags for the LAST bar of one timeframe, or None if there is not enough history."""
    if len(frame) < min_bars:
        return None
    x = with_indicators(frame)
    r = x.iloc[-1]
    if r[["Pivot20", "RVOL", "RSI", "EMA50", "MACDSig"]].isna().any():
        return None
    c, pivot = float(r["Close"]), float(r["Pivot20"])
    near = pivot * (1 - p["near_pct"] / 100) <= c <= pivot * 1.03
    return dict(close=c, open=float(r["Open"]), high=float(r["High"]), low=float(r["Low"]), pivot=pivot,
                support=float(r["Low20"]), rvol=float(r["RVOL"]), nr7=bool(r["NR7"]), rsi=float(r["RSI"]),
                atr=float(r["ATR"]) if pd.notna(r["ATR"]) else np.nan,
                high252=float(r["High252"]) if pd.notna(r["High252"]) else np.nan,
                ready=bool(r["NR7"]) and near and float(r["RVOL"]) >= p["min_rvol"],
                confirmed=c > pivot and c > float(r["Open"]) and float(r["RVOL"]) >= p["breakout_rvol"],
                trend=c > float(r["EMA20"]) > float(r["EMA50"]),
                momentum=55 <= float(r["RSI"]) <= 80 and float(r["MACD"]) > float(r["MACDSig"]),
                date=x.index[-1])


def analyse(sym, daily, p):
    """Returns (row_or_None, reason). reason names the first gate that rejected the stock (for the 'why no trade' funnel)."""
    if len(daily) < 260:
        return None, "short history"
    last = daily.iloc[-1]
    if float(last["Close"]) < p["min_price"]:
        return None, "price too low"
    if float((daily["Close"] * daily["Volume"]).tail(20).median()) < p["min_turnover"] * 1e7:
        return None, "illiquid"
    if float(last["High"]) <= float(last["Low"]):
        return None, "frozen candle"
    d = read_frame(daily, 60, p)
    w = read_frame(resample_complete(daily, "W-FRI"), 40, p)
    m = read_frame(resample_complete(daily, "ME"), 30, p)          # new listings simply have no monthly frame
    if d is None or w is None:
        return None, "short history"
    frames = [f for f in (d, w, m) if f]
    need = min(2, len(frames))
    n_trend, n_mom = sum(f["trend"] for f in frames), sum(f["momentum"] for f in frames)
    ready_now = d["ready"] and d["close"] <= d["pivot"]            # still under the pivot = a clean buy-above level exists
    if not (d["confirmed"] or ready_now):
        return None, "no daily setup"
    # momentum (RSI/MACD up) confirms a breakout that is happening; a coil has not expanded yet, so it needs trend only
    if (p["need_trend"] and n_trend < need) or (p["need_momentum"] and d["confirmed"] and n_mom < need):
        return None, "trend/momentum weak"

    atr = d["atr"]
    if not (atr > 0):
        return None, "short history"
    if d["confirmed"]:
        kind, entry = "BREAKOUT CONFIRMED", d["high"] * 1.001                      # buy-stop just above today's high
        stop = min(d["low"], d["pivot"] - 0.25 * atr) - 0.1 * atr                  # below the bar AND back inside the old range
    else:
        kind, entry = "READY - buy above pivot", d["pivot"] * 1.001
        stop = min(d["low"], float(daily["Low"].tail(5).min())) - 0.25 * atr
    stop = min(stop, entry - 0.5 * atr)                                            # never inside normal daily noise
    risk = entry - stop
    risk_pct = risk / entry * 100
    if risk <= 0 or risk_pct > p["max_stop_pct"]:
        return None, "stop too wide"
    # entry already at/within 2% of the 52-week high = a new-high breakout (clear air above); otherwise measure the room to it
    room_r = 4.0 if (math.isnan(d["high252"]) or entry >= d["high252"] * 0.98) else (d["high252"] - entry) / risk
    if room_r < p["min_room"]:
        return None, "52-week high too close"
    risk_rs_per_share = risk
    qty = int(min(math.floor(p["capital"] * p["risk_eff"] / 100 / risk_rs_per_share),
                  math.floor(p["capital"] * p["max_pos"] / 100 / entry)))
    if qty < 1:
        return None, "position size < 1 share"

    score = (25 if d["confirmed"] else 15) + 15 * int(w["confirmed"]) + 10 * int(bool(m and m["confirmed"]))
    score += 8 * int(d["nr7"]) + 6 * int(w["nr7"]) + 6 * int(bool(m and m["nr7"])) + 5 * n_trend + 5 * n_mom
    score += min(10, max(0, d["rvol"] - p["min_rvol"]) * 2) + (5 if room_r >= 4.0 else 0)
    if d["confirmed"]:
        why = (f"Closed above its 20-day high Rs {d['pivot']:,.2f} on a bullish candle with {d['rvol']:.1f}x volume.")
    else:
        why = (f"Narrow-range (NR7) candle {(d['pivot'] / d['close'] - 1) * 100:.1f}% under its 20-day high "
               f"Rs {d['pivot']:,.2f}, volume {d['rvol']:.1f}x. Buy ONLY if price trades above the trigger.")
    why += f" Trend up on {n_trend}/{len(frames)} and momentum OK on {n_mom}/{len(frames)} timeframes."
    if room_r >= 4.0:
        why += " Clear air above (at/near a 52-week high)."
    return dict(
        Symbol=sym, Signal=kind, Score=round(min(score, 100), 1), Close=round(d["close"], 2), Pivot=round(d["pivot"], 2),
        Entry=round(entry, 2), Stop=round(stop, 2), RiskPct=round(risk_pct, 1), T1=round(entry + 2 * risk, 2),
        T2=round(entry + 3 * risk, 2), RangeTarget=round(d["pivot"] + (d["pivot"] - d["support"]), 2),
        RoomR=round(min(room_r, 9.9), 1), SkipIfOpensAbove=round(entry * 1.02, 2), Qty=qty,
        Invest=round(qty * entry), RiskRs=round(qty * risk), RVOL_d=round(d["rvol"], 1), RVOL_w=round(w["rvol"], 1),
        RVOL_m=round(m["rvol"], 1) if m else np.nan, Trend=f"{n_trend}/{len(frames)}", Momentum=f"{n_mom}/{len(frames)}",
        RSI=round(d["rsi"], 0), NR7_d=d["nr7"], Candle=str(d["date"].date()), Why=why), "ok"


def market_status():
    """Is the broad market healthy? Nifty 500 (fallback Nifty 50) vs its 50/200-day averages."""
    for t in ("^CRSLDX", "^NSEI"):
        f = download_chunk((t,), "2y").get(t)
        if f is not None and len(f) > 210:
            c = f["Close"]
            e50, e200 = float(ema(c, 50).iloc[-1]), float(ema(c, 200).iloc[-1])
            last = float(c.iloc[-1])
            label = "UP" if last > e50 > e200 else "DOWN" if last < e50 and last < e200 else "MIXED"
            return dict(label=label, name="Nifty 500" if t == "^CRSLDX" else "Nifty 50", close=last, ema50=e50, ema200=e200,
                        asof=f.index[-1].strftime("%d-%b-%Y"))
    return None


# ===== UI =====
def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


st.title("📈 Nifty Breakout Trader")
st.caption("Finds stocks coiling or breaking out of their 20-day high with volume, checks daily + weekly + monthly trend, "
           "and gives a ready trade plan. Free end-of-day Yahoo data - run it after the market closes.")
if yf is None:
    st.error("yfinance is not installed. Add `yfinance` to requirements.txt and reboot the app.")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("1. Your money")
capital = sb.number_input("Capital (Rs)", 10000.0, value=200000.0, step=10000.0)
risk_pct = sb.number_input("Risk per trade (% of capital)", 0.25, 5.0, 1.0, 0.25,
                           help="If the stop is hit you lose about this much. 1% is a common ceiling.")
sb.header("2. Style")
style = sb.radio("How strict?", list(PRESETS), help="Strict = the original V2 rules (very few results). Balanced = recommended. "
                                                    "Wide = more ideas, lower quality.")
sb.header("3. Stocks")
universe_mode = sb.radio("Scan", ["Nifty 500 (official list)", "My own symbols"])
my_text = sb.text_area("Symbols (comma separated)", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, ITC, LT, SBIN",
                       disabled=universe_mode != "My own symbols")
with sb.expander("Advanced (optional)"):
    pr = PRESETS[style]
    near_pct = st.slider("Ready: max distance below the pivot (%)", 1.0, 12.0, pr["near_pct"], 0.5)
    min_rvol = st.slider("Ready: volume at least x average", 1.0, 5.0, pr["min_rvol"], 0.25)
    breakout_rvol = st.slider("Confirmed breakout: volume at least x average", 1.5, 10.0, pr["breakout_rvol"], 0.5)
    need_trend = st.checkbox("Trend (close > EMA20 > EMA50) on at least 2 timeframes", pr["need_trend"])
    need_momentum = st.checkbox("Momentum (RSI 55-80 + MACD up) on at least 2 timeframes - confirmed breakouts", pr["need_momentum"])
    min_turnover = st.slider("Min daily turnover (Rs crore, 20-day median)", 1, 50, 5)
    min_price = st.slider("Min share price (Rs)", 5, 200, 20)
    max_stop_pct = st.slider("Max stop distance (%)", 3.0, 15.0, 8.0, 0.5)
    min_room = st.slider("Min room to the 52-week high (x risk)", 0.5, 4.0, 1.5, 0.5)
    max_pos = st.slider("Max % of capital in one stock", 5, 50, 25)
    top_n = st.slider("Trade ideas to show", 1, 10, 5)
    scan_n = st.slider("Scan first N stocks", 25, 500, 500, 25)
    halve_weak = st.checkbox("Halve the risk when the market trend is DOWN", True)
    skip_unfinished = st.checkbox("Ignore today's unfinished candle (before 4 PM IST)", True)
go = sb.button("🚀 Find today's trades", type="primary")
sb.header("Check your data")
if sb.button("🧪 Quick data test (10 s)"):
    with st.spinner("Downloading RELIANCE and TCS from Yahoo..."):
        try:
            tst = yf.download(["RELIANCE.NS", "TCS.NS"], period="10d", interval="1d", auto_adjust=True,
                              group_by="ticker", progress=False, timeout=20)
            rows = []
            for t in ("RELIANCE.NS", "TCS.NS"):
                f = clean_ohlcv(tst[t]) if t in tst.columns.get_level_values(0) else None
                if f is None:
                    rows.append({"Stock": t, "Result": "no data"})
                else:
                    rows.append({"Stock": t, "Last candle": f.index[-1].strftime("%d-%b-%Y"), "Close": round(float(f["Close"].iloc[-1]), 2),
                                 "Previous close": round(float(f["Close"].iloc[-2]), 2), "Volume": int(f["Volume"].iloc[-1])})
            st.session_state["TEST"] = rows
        except Exception as e:
            st.session_state["TEST"] = [{"Stock": "-", "Result": f"FAILED: {str(e)[:120]}"}]
if st.session_state.get("TEST"):
    st.info("Data test - compare these closes with your broker chart (they should match):")
    show_df(pd.DataFrame(st.session_state["TEST"]))

def settings_signature():
    return repr([style, universe_mode, my_text if universe_mode == "My own symbols" else "", capital, risk_pct, near_pct, min_rvol,
                 breakout_rvol, need_trend, need_momentum, min_turnover, min_price, max_stop_pct, min_room, max_pos, scan_n,
                 halve_weak, skip_unfinished])


p = dict(near_pct=near_pct, min_rvol=min_rvol, breakout_rvol=breakout_rvol, need_trend=need_trend,
         need_momentum=need_momentum, min_turnover=min_turnover, min_price=min_price, max_stop_pct=max_stop_pct,
         min_room=min_room, max_pos=max_pos, capital=capital)

# ----------------------------------------------------------------------------- scan
if go:
    if universe_mode.startswith("Nifty"):
        lst = nifty500_list()
        if lst is None:
            st.error("Could not download the official Nifty 500 list. Choose 'My own symbols' in the sidebar and try again.")
            st.stop()
        symbols = lst["Symbol"].tolist()[:scan_n]
    else:
        symbols = sorted({s.strip().upper().replace(".NS", "") for s in my_text.replace("\n", ",").split(",") if s.strip()})
    if not symbols:
        st.error("No symbols to scan.")
        st.stop()
    bar = st.progress(0.0, text=f"Downloading prices 0/{len(symbols)}")
    frames, empty_chunks = download_all(symbols, bar)
    bar.empty()
    if not frames:
        st.error("No price data came back from Yahoo (blocked, rate-limited or offline). Wait a few minutes and try again, or "
                 "run the quick data test in the sidebar.")
        st.stop()
    mkt = market_status()
    risk_eff = risk_pct * (0.5 if (halve_weak and mkt and mkt["label"] == "DOWN") else 1.0)
    p["risk_eff"] = risk_eff
    rows, why_not = [], {}
    work = st.progress(0.0, text="Analysing...")
    items = list(frames.items())
    for i, (sym, f) in enumerate(items):
        if skip_unfinished:
            f = drop_unfinished_today(f)
        try:
            row, reason = analyse(sym, f, p)
        except Exception:
            row, reason = None, "calculation error"
        if row:
            rows.append(row)
        else:
            why_not[reason] = why_not.get(reason, 0) + 1
        if i % 25 == 0:
            work.progress((i + 1) / len(items), text=f"Analysing {i + 1}/{len(items)}")
    work.empty()
    df = pd.DataFrame(rows).sort_values(["Score", "RVOL_d"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    asof = max(f.index[-1] for f in frames.values()).strftime("%d-%b-%Y")
    st.session_state["R"] = dict(df=df, why_not=why_not, attempted=len(symbols), got=len(frames), empty=empty_chunks,
                                 asof=asof, at=datetime.now().strftime("%d-%b-%Y %H:%M"), mkt=mkt, risk_eff=risk_eff,
                                 style=style, sig=settings_signature())

R = st.session_state.get("R")
if not R:
    st.info("Set your capital and risk in the sidebar, then press **🚀 Find today's trades**. "
            "The first scan of 500 stocks takes a few minutes; repeat scans are much faster.")
    st.stop()

# ----------------------------------------------------------------------------- results
df, mkt = R["df"], R["mkt"]
st.caption(f"📅 Price data as of **{R['asof']}** (last completed daily candle) · scan run {R['at']} · style: {R['style']} · "
           f"{R['got']} of {R['attempted']} stocks had data. End-of-day data, not live.")
if R["empty"]:
    st.warning(f"{R['empty']} download batch(es) returned nothing (Yahoo rate limit). Those stocks are missing; press the "
               "button again in a few minutes - finished batches are cached.")
if mkt:
    msg = (f"Market ({mkt['name']}, {mkt['asof']}): **{mkt['label']}** - close {mkt['close']:,.0f}, "
           f"EMA50 {mkt['ema50']:,.0f}, EMA200 {mkt['ema200']:,.0f}.")
    {"UP": st.success, "MIXED": st.warning, "DOWN": st.error}[mkt["label"]](
        msg + (f" Risk per trade is cut to {R['risk_eff']:.2f}%." if R["risk_eff"] < risk_pct else ""))
else:
    st.info("Market trend unavailable (index download failed) - judge the market yourself.")

if R.get("sig") != settings_signature():
    st.info("⚙️ You changed a setting after this scan. Press **🚀 Find today's trades** again to apply it "
            "(repeat scans reuse the downloaded prices, so it is quick).")
top = df.head(top_n) if not df.empty else df
if df.empty:
    st.warning("⛔ No trade today. This is a valid result - do not loosen the rules just to force a trade.")
else:
    st.success(f"✅ {len(top)} trade idea(s) shown (of {len(df)} that passed every rule). "
               "Entries are BUY-STOP triggers: buy only if price trades above the trigger.")

t1, t2, t3, t4 = st.tabs(["🎯 Trade ideas", "📋 All candidates", "❓ Why not more?", "📖 Rules & checklist"])

with t1:
    for i, r in top.iterrows():
        with st.container(border=True):
            st.markdown(f"### {i + 1}. {r['Symbol']} · {r['Signal']} · score {r['Score']:.0f}")
            c = st.columns(5)
            c[0].metric("BUY above", f"Rs {r['Entry']:,.2f}")
            c[1].metric("Stop-loss", f"Rs {r['Stop']:,.2f}", f"-{r['RiskPct']}%", delta_color="off")
            c[2].metric("Target 1 (2R)", f"Rs {r['T1']:,.2f}")
            c[3].metric("Target 2 (3R)", f"Rs {r['T2']:,.2f}")
            c[4].metric("Quantity", f"{int(r['Qty']):,}", f"Rs {int(r['Invest']):,}", delta_color="off")
            st.write(r["Why"])
            st.caption(f"Risk if stopped: Rs {int(r['RiskRs']):,} · skip it if it opens above Rs {r['SkipIfOpensAbove']:,.2f} (gap) · "
                       f"room to the 52-week high: {r['RoomR']}x risk · range target Rs {r['RangeTarget']:,.2f} · "
                       f"volume d/w/m: {r['RVOL_d']}x / {r['RVOL_w']}x / {r['RVOL_m']}x · RSI {r['RSI']:.0f} · candle {r['Candle']}")
            st.info("Plan: place a BUY-STOP at the trigger (valid 1-3 sessions). The moment it fills, place the stop-loss. At Target 1 sell "
                    "half and move the stop to your entry; trail the rest under the 20-day average. Never move a stop down.")
    if not top.empty:
        st.download_button("⬇️ Download these trade plans (CSV)", top.drop(columns=["Why"]).to_csv(index=False).encode(),
                           "trade_plans.csv", "text/csv")

with t2:
    if df.empty:
        st.info("Nothing passed.")
    else:
        cols = ["Symbol", "Signal", "Score", "Close", "Entry", "Stop", "RiskPct", "T1", "RoomR", "Qty", "RVOL_d", "RVOL_w",
                "RVOL_m", "Trend", "Momentum", "RSI", "Candle"]
        show_df(df[cols])
        st.download_button("⬇️ Download all candidates (CSV)", df.drop(columns=["Why"]).to_csv(index=False).encode(),
                           "all_candidates.csv", "text/csv")

with t3:
    st.caption("Stocks rejected, by the FIRST rule that stopped them. 'no daily setup' is normally the biggest bucket.")
    wn = pd.DataFrame(sorted(R["why_not"].items(), key=lambda kv: -kv[1]), columns=["Reason", "Stocks"])
    show_df(wn) if len(wn) else st.write("-")

with t4:
    st.markdown("""
**Rules (from the V2 breakout logic, made tradable)**
1. **Pivot** = highest high of the previous 20 completed candles. **NR7** = today's range is the smallest of the last 7.
2. **RVOL** = volume / average volume of the previous 20 candles (today is not in its own average).
3. **READY** = NR7 candle, still within the chosen % *below* the pivot, volume above the ready threshold. Buy only above the pivot.
4. **CONFIRMED** = closed above the pivot on a bullish candle with volume above the breakout threshold. Buy above today's high.
5. **Trend** = close > EMA20 > EMA50. **Momentum** = RSI 55-80 and MACD above its signal. Checked on daily, weekly and monthly
   bars (completed weeks/months only); two timeframes must agree on trend for every idea, and on momentum for confirmed
   breakouts (a coil has not expanded yet, so it is judged on trend only).
6. **Trade plan:** stop under the signal candle and under the pivot (a failed breakout = back inside the old range), never inside
   normal noise, capped by *Max stop distance*; targets 2R and 3R; at least *Min room* to the 52-week high; quantity from your
   risk %, capped by *Max % of capital*. Stocks that are illiquid, cheap or frozen at a circuit are skipped.
7. **Score** ranks candidates; it is NOT a probability of profit.

**Before you buy (2 minutes):** price and volume match your broker · no results/board meeting in the next few days · no bad news ·
the stock is not in a circuit/ASM/GSM list · if it opens above the "skip" price, do not chase.
""")
st.warning("Free Yahoo end-of-day data can be delayed, incomplete or wrong. Verify every candidate in your broker before ordering. "
           "This app does not place trades and is not investment advice.")
