"""Recolhe fundamentais e histórico de preços (10 anos, mensal, em EUR) dos candidatos ao ETF."""
import json, time, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, yfinance as yf

OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
CANDIDATES = """MSFT GOOGL NVDA AVGO ASML.AS TSM SAP.DE CDNS SNPS ADBE
V MA SPGI MCO MSCI LSEG.L DB1.DE ICE CME BRK-B
LLY NOVO-B.CO ISRG TMO ZTS SYK NOVN.SW AZN.L EL.PA LONN.SW
COST MC.PA RMS.PA OR.PA NESN.SW PG KO RACE.MI ITX.MC
SU.PA ATCO-A.ST DSV.CO CPRT WM UNP AI.PA LIN SIKA.SW REL.L WKL.AS
6861.T 8035.T 0700.HK HDB MELI IBE.MC NEE PLD""".split()
BENCH = ["EUNL.DE", "SXR8.DE"]  # iShares Core MSCI World (EUR), iShares Core S&P 500 (EUR)
KEYS = ["shortName","sector","industry","country","currency","financialCurrency","marketCap","returnOnEquity",
        "returnOnAssets","grossMargins","operatingMargins","profitMargins","revenueGrowth","earningsGrowth",
        "debtToEquity","forwardPE","trailingPE","pegRatio","beta","freeCashflow","totalRevenue","ebitda",
        "totalDebt","totalCash","dividendYield","payoutRatio"]

def info(t):
    for a in range(3):
        try:
            i = yf.Ticker(t).info
            if i and i.get("shortName"):
                d = {k: i.get(k) for k in KEYS}; d["ticker"] = t
                try:
                    inc = yf.Ticker(t).income_stmt
                    rev = inc.loc["Total Revenue"].dropna().sort_index()
                    if len(rev) >= 3:
                        n = len(rev) - 1
                        d["rev_cagr_hist"] = (rev.iloc[-1] / rev.iloc[0]) ** (1 / n) - 1
                        d["rev_years"] = n
                except Exception:
                    pass
                return d
        except Exception:
            pass
        time.sleep(2 + 2 * a)
    return None

with ThreadPoolExecutor(6) as ex:
    rows = [r for r in ex.map(info, CANDIDATES) if r]
pd.DataFrame(rows).to_csv(OUT / "fundamentals.csv", index=False)
print("fundamentais:", len(rows), "/", len(CANDIDATES), file=sys.stderr)

tick = CANDIDATES + BENCH
px = yf.download(tick, period="11y", interval="1mo", auto_adjust=True, progress=False, threads=True)["Close"]
ccy = {r["ticker"]: r["currency"] for r in rows}
for b in BENCH: ccy[b] = "EUR"
need = {c for c in ccy.values() if c and c != "EUR"}
fx = {}
for c in need:
    major = "GBP" if c in ("GBp", "GBX") else c
    s = yf.download(f"EUR{major}=X", period="11y", interval="1mo", auto_adjust=True, progress=False)["Close"]
    fx[c] = s.squeeze() * (100 if c in ("GBp", "GBX") else 1)
eur = pd.DataFrame(index=px.index)
for t in tick:
    if t not in px: continue
    c = ccy.get(t, "EUR")
    eur[t] = px[t] if c == "EUR" else px[t] / fx[c].reindex(px.index).ffill()
eur.to_csv(OUT / "prices_eur_monthly.csv")
print("preços:", eur.shape, file=sys.stderr)
