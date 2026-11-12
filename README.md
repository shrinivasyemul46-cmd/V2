# 📈 Nifty Breakout Trader

One-file Streamlit app. Finds Nifty 500 stocks that are **coiling (NR7) or breaking out of their 20-day high with volume**, checks
daily + weekly + monthly trend, and gives a **ready trade plan**: buy trigger, stop-loss, two targets, quantity, skip rule.
Built on the NR7 + relative-volume + 20-bar-pivot logic of the attached "Nifty 500 Breakout Screener V2".

## Files (3, all in the GitHub repo root, exact names)
`app.py` · `requirements.txt` · `README.md`

## Use it (3 steps)
1. Sidebar: set **Capital** and **Risk per trade %**; pick a **Style** (Balanced is recommended; Strict = the original V2 rules).
2. Press **🚀 Find today's trades** after the market closes (about 4 PM IST).
3. Read the **🎯 Trade ideas** cards (at most 5 by default). Each shows BUY-above trigger, stop, Target 1/2, quantity, risk in rupees,
   a "skip if it opens above" price, and a plain-language reason. Place a BUY-STOP at the trigger; once filled, place the stop.

The sidebar **🧪 Quick data test** downloads RELIANCE and TCS so you can compare the closes with your broker before trusting a scan.

## What changed vs the attached V2 app
* **Trade plan added** (V2 only listed candidates): entry, stop (under the signal candle and under the pivot, never inside normal
  daily noise, capped by Max stop distance), targets 2R/3R, range target, room to the 52-week high, position size from your risk %.
* **READY now means still below the pivot** (V2 also accepted closes up to 3% above it, where no clean buy-above level exists).
* **Momentum is demanded of confirmed breakouts only**; a coil is judged on trend (it has not expanded yet).
* **Monthly history no longer needs 60 bars** (V2's 60-bar rule silently dropped every stock listed under ~5 years); new listings
  are judged on the frames they have.
* **Ranking fixed:** V2 sorted the three labels alphabetically (multi-timeframe READY ended last). Candidates now rank by score.
* **Faster and safer downloads:** sequential cached batches of 50 with progress, one retry, polite pacing (V2 called yfinance once
  per stock from several threads; yfinance keeps shared global state, so parallel calls can drop or mix results).
* **Market banner** (Nifty 500 vs EMA50/200) and optional automatic half-risk when the market trend is DOWN.
* **Liquidity, price and circuit-candle filters**; unfinished today-candle dropped before 4 PM IST; only COMPLETE weeks/months used.
* **"Why not more?"** tab shows which rule stopped how many stocks (so "no trade" is explained).
* Settings-changed notice, CSV downloads, mobile-friendly cards, no deprecated Streamlit arguments.

## Honest limits
* Free end-of-day Yahoo data: can be delayed, incomplete or wrong. Verify price, volume, results dates and news in your broker.
* Tested only on synthetic data (logic, arithmetic, no look-ahead, completed periods, failure paths) - not yet on real NSE history.
* No fundamentals, no backtest. The score ranks setups; it is NOT a probability of profit. Paper-trade first.
* Educational tool, not investment advice.
