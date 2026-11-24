"""Screener diário de ações abaixo do valor intrínseco (EUA + Europa).

Uso:  python -m screener.run [--limit N] [--out reports]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import universe
from .valuation import FINANCIAL_SECTORS, intrinsic_value, num, quality_score

MIN_MARGIN = 0.20    # margem de segurança mínima para "oportunidade"
WATCH_MARGIN = 0.10  # margem para a lista de observação
MIN_QUALITY = 55     # pontuação mínima de qualidade
MAX_FLAGS = 1        # nº máximo de alertas aceites

INFO_KEYS = [
    "shortName", "sector", "industry", "currency", "financialCurrency", "currentPrice",
    "regularMarketPrice", "sharesOutstanding", "trailingEps", "bookValue", "freeCashflow",
    "earningsGrowth", "revenueGrowth", "beta", "returnOnEquity", "profitMargins",
    "debtToEquity", "currentRatio", "payoutRatio", "52WeekChange", "recommendationMean",
    "targetMeanPrice", "numberOfAnalystOpinions", "marketCap", "dividendYield",
    "netIncomeToCommon", "impliedSharesOutstanding", "forwardEps",
]

_fx_cache: dict[str, float] = {}


def fx(src: str, dst: str) -> float | None:
    """Taxa de câmbio src→dst (1 src = x dst)."""
    if src == dst:
        return 1.0
    key = src + dst
    if key not in _fx_cache:
        rate = None
        for attempt in range(3):
            try:
                rate = num(yf.Ticker(f"{key}=X").fast_info["lastPrice"])
                if rate:
                    break
            except Exception:
                time.sleep(1 + attempt)
        _fx_cache[key] = rate
    return _fx_cache[key]


def normalize_ccy(ccy: str | None, price: float | None):
    """Converte cotações em subunidades (pence) para a unidade principal."""
    if ccy in ("GBp", "GBX"):
        return "GBP", (price / 100 if price is not None else None), 100.0
    if ccy in ("ZAc", "ILA"):
        major = {"ZAc": "ZAR", "ILA": "ILS"}[ccy]
        return major, (price / 100 if price is not None else None), 100.0
    return ccy, price, 1.0


def fetch(ticker: str) -> dict | None:
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info
            if info and (info.get("currentPrice") or info.get("regularMarketPrice")):
                d = {k: info.get(k) for k in INFO_KEYS}
                d["ticker"] = ticker
                return d
        except Exception:
            pass
        time.sleep(2 + 2 * attempt)  # resposta vazia/401 do Yahoo costuma ser temporária
    return None


def analyse(d: dict) -> dict | None:
    price_q = num(d.get("currentPrice")) or num(d.get("regularMarketPrice"))
    # impliedSharesOutstanding inclui todas as classes de ações (ex.: Berkshire A+B, Maersk A+B)
    shares = num(d.get("impliedSharesOutstanding")) or num(d.get("sharesOutstanding"))
    if not price_q or not shares:
        return None
    q_ccy, price_major, sub = normalize_ccy(d.get("currency"), price_q)
    f_ccy = d.get("financialCurrency") or q_ccy
    rate = fx(q_ccy, f_ccy)
    if rate is None:
        return None

    price_f = price_major * rate  # preço na moeda de reporte
    fcf = num(d.get("freeCashflow"))
    ni = num(d.get("netIncomeToCommon"))
    # Totais (lucro, FCF) vêm sempre na moeda de reporte -> per-share fiável
    eps_f = ni / shares if ni is not None else None
    d["fcf_ps"] = fcf / shares if fcf is not None else None

    # trailingEps/bookValue do Yahoo vêm às vezes na moeda de cotação, às vezes na de reporte.
    # Descobre o fator comparando trailingEps com lucro/ações e aplica-o ao valor contabilístico.
    t_eps, bv = num(d.get("trailingEps")), num(d.get("bookValue"))
    factor = 1.0
    if t_eps and eps_f and t_eps * eps_f > 0:
        factor = min((1.0, rate, rate / sub), key=lambda f: abs(math.log(t_eps * f / eps_f)))
    bvps = bv * factor if bv is not None else None
    if bvps is not None and not (0.02 < bvps / price_f < 20):
        bvps = None  # valor incoerente (ex.: book value por ação classe A)
    d["eps"] = eps_f if eps_f is not None else (t_eps * factor if t_eps else None)
    d["bvps"] = bvps
    f_eps, t_eps_raw = num(d.get("forwardEps")), num(d.get("trailingEps"))
    d["fwd_ratio"] = f_eps / t_eps_raw if f_eps and t_eps_raw and t_eps_raw > 0 else None

    iv_f, detail = intrinsic_value(d)
    quality, flags = quality_score(d)
    to_quote = sub / rate  # converte da moeda de reporte para a cotação original

    target = num(d.get("targetMeanPrice"))
    out = {
        "ticker": d["ticker"],
        "nome": d.get("shortName"),
        "regiao": universe.region(d["ticker"]),
        "setor": d.get("sector"),
        "moeda": d.get("currency"),
        "preco": price_q,
        "valor_intrinseco": iv_f * to_quote if iv_f else None,
        "margem_seguranca": (1 - price_f / iv_f) if iv_f else None,
        "qualidade": quality,
        "alertas": "; ".join(flags),
        "n_alertas": len(flags),
        "pe": price_f / d["eps"] if d["eps"] and d["eps"] > 0 else None,
        "fcf_yield": d["fcf_ps"] / price_f if d["fcf_ps"] else None,
        "roe": num(d.get("returnOnEquity")),
        "div_yield": num(d.get("dividendYield")),
        "cresc_usado": detail["g"],
        "taxa_desconto": detail["r"],
        "metodos": json.dumps({k: round(v * to_quote, 2) for k, v in detail["methods"].items()},
                              ensure_ascii=False),
        "alvo_analistas": target,
        "upside_analistas": (target / price_q - 1) if target else None,
        "n_analistas": d.get("numberOfAnalystOpinions"),
        "rec_analistas": num(d.get("recommendationMean")),
        "financeira": d.get("sector") in FINANCIAL_SECTORS,
    }
    out["score"] = (
        (out["margem_seguranca"] or 0) * 60 + quality * 0.4 if out["margem_seguranca"] else None
    )
    return out


def pct(x, nd=0):
    return "—" if x is None or pd.isna(x) else f"{x * 100:.{nd}f}%"


def money(x, ccy):
    return "—" if x is None or pd.isna(x) else f"{x:,.2f} {ccy}"


def select(df: pd.DataFrame):
    """Devolve (oportunidades, lista de observação), ordenadas por score."""
    eligible = df[
        (df.qualidade >= MIN_QUALITY) & (df.n_alertas <= MAX_FLAGS) & df.margem_seguranca.notna()
    ]
    picks = eligible[eligible.margem_seguranca >= MIN_MARGIN].sort_values("score", ascending=False)
    watch = eligible[
        (eligible.margem_seguranca >= WATCH_MARGIN) & (eligible.margem_seguranca < MIN_MARGIN)
    ].sort_values("score", ascending=False)
    return picks, watch


def build_report(df: pd.DataFrame, today: str, prev: pd.DataFrame | None, n_ok: int, n_total: int):
    picks, watch = select(df)

    prev_picks = set(prev.ticker) if prev is not None else set()
    cur = set(picks.ticker.head(15))
    entered = sorted(cur - prev_picks) if prev is not None else []
    left = sorted(prev_picks - cur) if prev is not None else []

    L = [f"# Oportunidades em bolsa — {today}", ""]
    L.append(
        f"Analisadas **{n_ok}** de {n_total} ações (EUA + Europa). "
        f"Critérios: margem de segurança ≥ {MIN_MARGIN:.0%}, qualidade ≥ {MIN_QUALITY}/100, "
        f"no máximo {MAX_FLAGS} alerta de risco."
    )
    L.append("")
    if prev is not None:
        L.append(f"**Entraram hoje:** {', '.join(entered) or 'nenhuma'}  ")
        L.append(f"**Saíram:** {', '.join(left) or 'nenhuma'}")
        L.append("")

    L.append(f"## Top oportunidades ({min(len(picks), 15)})")
    L.append("")
    if picks.empty:
        L.append("_Nenhuma ação cumpre todos os critérios hoje._")
    else:
        L.append("| # | Ticker | Empresa | Região | Preço | Valor intrínseco | Margem | Qualidade | P/E | FCF yield | Analistas (upside) | Alertas |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(picks.head(15).itertuples(), 1):
            L.append(
                f"| {i} | **{r.ticker}** | {r.nome} | {r.regiao} | {money(r.preco, r.moeda)} | "
                f"{money(r.valor_intrinseco, r.moeda)} | **{pct(r.margem_seguranca)}** | {r.qualidade} | "
                f"{'—' if pd.isna(r.pe) else f'{r.pe:.1f}'} | {pct(r.fcf_yield, 1)} | "
                f"{pct(r.upside_analistas)} | {r.alertas or '—'} |"
            )
        L.append("")
        L.append("### Detalhe da avaliação")
        L.append("")
        for r in picks.head(15).itertuples():
            methods = json.loads(r.metodos)
            mtxt = ", ".join(f"{k}: {v:,.2f}" for k, v in methods.items())
            L.append(
                f"- **{r.ticker}** ({r.setor}) — {mtxt}. Crescimento assumido {pct(r.cresc_usado, 1)}, "
                f"taxa de desconto {pct(r.taxa_desconto, 1)}, ROE {pct(r.roe)}, "
                f"dividendo {'—' if pd.isna(r.div_yield) else f'{r.div_yield:.1f}%'}."
            )
    L.append("")
    L.append(f"## Lista de observação (margem {WATCH_MARGIN:.0%}–{MIN_MARGIN:.0%})")
    L.append("")
    if watch.empty:
        L.append("_Vazia._")
    else:
        L.append("| Ticker | Empresa | Preço | Valor intrínseco | Margem | Qualidade |")
        L.append("|---|---|---|---|---|---|")
        for r in watch.head(10).itertuples():
            L.append(
                f"| {r.ticker} | {r.nome} | {money(r.preco, r.moeda)} | "
                f"{money(r.valor_intrinseco, r.moeda)} | {pct(r.margem_seguranca)} | {r.qualidade} |"
            )
    L.append("")
    traps = df[(df.margem_seguranca >= MIN_MARGIN) & (df.n_alertas > MAX_FLAGS)]
    if not traps.empty:
        L.append("## Baratas mas com sinais de risco (possíveis *value traps*)")
        L.append("")
        for r in traps.sort_values("margem_seguranca", ascending=False).head(8).itertuples():
            L.append(f"- **{r.ticker}** ({r.nome}) — margem {pct(r.margem_seguranca)}: {r.alertas}")
        L.append("")
    L += [
        "## Metodologia",
        "",
        "- **Valor intrínseco** = mediana de 3 métodos (mínimo 2 disponíveis): DCF de 10 anos sobre o "
        "free cash flow (bancos/seguradoras: P/B justificado pelo ROE), número de Graham e fórmula "
        "de crescimento de Graham.",
        "- Crescimento = o menor entre crescimento de lucros, de receitas e o EPS previsto pelos "
        "analistas, limitado a −2%…8% (energia e matérias-primas: máx. 3%, por serem cíclicas). "
        "Taxa de desconto = 4% + β × 5,5% (7,5%–12%). Crescimento terminal 2,5%.",
        "- **Qualidade** (0–100): ROE, margem líquida, dívida/capital, FCF positivo, liquidez, "
        "crescimento de receitas e lucros.",
        "- **Alertas**: queda > 30% em 12 meses, lucros a cair > 20%, receitas a cair > 5%, "
        "payout > 100%, dívida elevada, analistas negativos.",
        "- Dados: Yahoo Finance (podem ter atrasos ou erros). Valores convertidos para a moeda de "
        "reporte de cada empresa antes de comparar.",
        "",
        "> ⚠️ Isto é uma triagem quantitativa automática, não uma recomendação de investimento. "
        "Confirma sempre os relatórios da empresa, a tese e o teu perfil de risco antes de comprar.",
    ]
    return "\n".join(L), picks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="analisar só as primeiras N ações (teste)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    tickers = universe.ALL[: args.limit] if args.limit else universe.ALL
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        raw = [d for d in ex.map(fetch, tickers) if d]
    # Segunda passagem, sequencial, para tickers que falharam por limite de pedidos
    got = {d["ticker"] for d in raw}
    for t in [t for t in tickers if t not in got]:
        time.sleep(1)
        d = fetch(t)
        if d:
            raw.append(d)
    rows = []
    for d in raw:
        try:
            r = analyse(d)
            if r:
                rows.append(r)
        except Exception as e:  # um ticker com dados estranhos não deve parar o relatório
            print(f"[aviso] {d.get('ticker')}: {e}", file=sys.stderr)
    if not rows:
        print("Sem dados — verificar acesso ao Yahoo Finance.", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    prev_file = out / "latest_picks.csv"
    hist = out / "history.csv"
    prev = None
    if hist.exists():  # escolhas do último dia anterior a hoje (correr 2x no mesmo dia não estraga a comparação)
        h_all = pd.read_csv(hist)
        before = h_all[h_all.data < today]
        if not before.empty:
            prev = before[before.data == before.data.max()]

    report, picks = build_report(df, today, prev, len(df), len(tickers))
    (out / f"{today}.md").write_text(report, encoding="utf-8")
    (out / "latest.md").write_text(report, encoding="utf-8")
    df.sort_values("score", ascending=False).to_csv(out / "latest_full.csv", index=False)
    picks.head(15).to_csv(prev_file, index=False)

    snap_dir = out / "data" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values("score", ascending=False).to_csv(snap_dir / f"{today}.csv", index=False)

    h = picks.head(15)[["ticker", "nome", "preco", "moeda", "valor_intrinseco", "margem_seguranca", "qualidade"]].copy()
    h.insert(0, "data", today)
    if hist.exists():
        old = pd.read_csv(hist)
        h = pd.concat([old[old.data != today], h], ignore_index=True)
    h.to_csv(hist, index=False)

    from . import dashboard
    dashboard.build(out)

    print(report)
    print(f"\n[{len(df)}/{len(tickers)} ações em {time.time() - t0:.0f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
