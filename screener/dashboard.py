"""Painel histórico: agrega os snapshots diários em séries temporais, comparações e gráficos.

Gera (dentro de reports/):
- daily_summary.csv   indicadores por dia (nº oportunidades, margens EUA vs Europa, ...)
- performance.csv     desempenho de cada ação desde que entrou na lista vs. universo
- sectors_today.csv   oportunidades por setor hoje
- charts/*.png        gráficos
- DASHBOARD.md        painel com gráficos e tabelas (visível no GitHub)

Tudo é recalculado a partir de reports/data/snapshots/*.csv, por isso pode correr várias vezes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta (validada para daltonismo): EUA = azul, Europa = laranja, 3ª série = verde-água
C_US, C_EU, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "axes.axisbelow": True,
})


def _pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=None)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _date_axis(ax, fig):
    import matplotlib.dates as mdates

    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, round((hi - lo) / 8))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.autofmt_xdate()


def load_snapshots(out: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((out / "data" / "snapshots").glob("*.csv")):
        d = pd.read_csv(f)
        d["data"] = f.stem
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def daily_summary(snaps: pd.DataFrame) -> pd.DataFrame:
    from .run import select

    rows = []
    for day, d in snaps.groupby("data"):
        picks, watch = select(d)
        top = picks.head(15)
        row = {
            "data": day,
            "acoes_analisadas": len(d),
            "oportunidades": len(picks),
            "observacao": len(watch),
            "oportunidades_eua": int((picks.regiao == "EUA").sum()),
            "oportunidades_europa": int((picks.regiao == "Europa").sum()),
            "margem_mediana_universo": d.margem_seguranca.median(),
            "margem_mediana_eua": d[d.regiao == "EUA"].margem_seguranca.median(),
            "margem_mediana_europa": d[d.regiao == "Europa"].margem_seguranca.median(),
            "margem_media_top15": top.margem_seguranca.mean(),
            "qualidade_media_top15": top.qualidade.mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("data")


def performance(snaps: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Retorno de cada ação desde o 1º dia em que entrou no top 15, vs. média do universo."""
    if hist.empty:
        return pd.DataFrame()
    last_day = snaps.data.max()
    last = snaps[snaps.data == last_day].set_index("ticker")
    current = set(hist[hist.data == hist.data.max()].ticker)
    rows = []
    for tk, g in hist.groupby("ticker"):
        first = g.sort_values("data").iloc[0]
        if tk not in last.index:
            continue
        p0, p1 = first.preco, last.loc[tk, "preco"]
        # Benchmark: retorno médio (pesos iguais) de todo o universo no mesmo período
        base = snaps[snaps.data == first.data].set_index("ticker").preco
        univ = (last.preco.reindex(base.index) / base - 1).dropna()
        r = p1 / p0 - 1
        rows.append({
            "ticker": tk, "nome": first.nome, "moeda": first.moeda,
            "entrou_em": first.data, "preco_entrada": p0, "preco_atual": p1,
            "retorno": r, "retorno_universo": univ.mean(), "excesso": r - univ.mean(),
            "dias_no_top15": g.data.nunique(), "ainda_no_top15": tk in current,
        })
    return pd.DataFrame(rows).sort_values("retorno", ascending=False)


# ---------- gráficos ----------

def chart_top_margin(today: pd.DataFrame, path: Path):
    from .run import select

    picks = select(today)[0].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [C_US if r == "EUA" else C_EU for r in picks.regiao]
    ax.barh(picks.ticker, picks.margem_seguranca, color=colors, height=0.7)
    for y, (m, q) in enumerate(zip(picks.margem_seguranca, picks.qualidade)):
        ax.text(m + 0.005, y, f"{m:.0%}  ·  Q{q}", va="center", fontsize=8.5, color=INK2)
    _pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(picks.margem_seguranca.max() * 1.25, 0.3))
    ax.set_title("Top 15 hoje: margem de segurança (Q = qualidade 0-100)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_US), plt.Rectangle((0, 0), 1, 1, color=C_EU)],
              labels=["EUA", "Europa"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_price_vs_value(today: pd.DataFrame, path: Path):
    """Preço vs valor intrínseco, normalizado (preço = 100) para comparar moedas diferentes."""
    from .run import select

    picks = select(today)[0].head(15).iloc[::-1]
    iv_idx = 100 * picks.valor_intrinseco / picks.preco
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = range(len(picks))
    ax.hlines(y, 100, iv_idx, color=GRID, linewidth=2)
    ax.scatter([100] * len(picks), y, color=INK2, s=40, zorder=3, label="Preço atual (=100)")
    ax.scatter(iv_idx, y, color=C_3, s=60, zorder=3, edgecolor=SURFACE, linewidth=2,
               label="Valor intrínseco estimado")
    ax.set_yticks(list(y), picks.ticker)
    ax.grid(axis="y", visible=False)
    ax.set_title("Preço vs valor intrínseco (preço atual = 100)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_opportunities_over_time(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = pd.to_datetime(summary.data)
    ax.plot(x, summary.oportunidades_eua, color=C_US, linewidth=2, marker="o", markersize=5, label="EUA")
    ax.plot(x, summary.oportunidades_europa, color=C_EU, linewidth=2, marker="o", markersize=5, label="Europa")
    ax.set_title("Nº de oportunidades por dia")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5)
    _date_axis(ax, fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_market_margin(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = pd.to_datetime(summary.data)
    ax.plot(x, summary.margem_mediana_eua, color=C_US, linewidth=2, marker="o", markersize=5, label="EUA")
    ax.plot(x, summary.margem_mediana_europa, color=C_EU, linewidth=2, marker="o", markersize=5, label="Europa")
    ax.axhline(0, color=INK2, linewidth=0.8)
    _pct_axis(ax)
    ax.set_title("Margem de segurança mediana do mercado (negativo = caro)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5)
    _date_axis(ax, fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_sectors(sectors: pd.DataFrame, path: Path):
    s = sectors.sort_values("oportunidades")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(s.setor, s.oportunidades, color=C_US, height=0.65)
    for yv, v in enumerate(s.oportunidades):
        ax.text(v + 0.1, yv, str(v), va="center", fontsize=9, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Oportunidades por setor hoje")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_performance(perf: pd.DataFrame, path: Path):
    p = perf.sort_values("excesso")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(p) + 1)))
    ax.barh(p.ticker, p.retorno, color=C_US, height=0.38, label="Ação")
    ax.barh([i + 0.4 for i in range(len(p))], p.retorno_universo, color=GRID, height=0.38,
            label="Média do universo (mesmo período)")
    ax.axvline(0, color=INK2, linewidth=0.8)
    _pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Retorno desde que entrou no top 15")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_margin_evolution(snaps: pd.DataFrame, tickers: list[str], path: Path):
    d = snaps[snaps.ticker.isin(tickers)].pivot_table(index="data", columns="ticker", values="margem_seguranca")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(d.index)
    palette = [C_US, C_EU, C_3, "#eda100", "#e87ba4"]
    for i, tk in enumerate(tickers):
        if tk in d:
            ax.plot(x, d[tk], color=palette[i % len(palette)], linewidth=2, marker="o", markersize=5, label=tk)
    _pct_axis(ax)
    ax.set_title("Evolução da margem de segurança: top 5 de hoje")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=8.5)
    _date_axis(ax, fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- painel ----------

def _fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:+.1%}" if abs(x) < 10 else "—"


def build(out: Path):
    out = Path(out)
    snaps = load_snapshots(out)
    if snaps.empty:
        return
    hist_file = out / "history.csv"
    hist = pd.read_csv(hist_file) if hist_file.exists() else pd.DataFrame()
    charts = out / "charts"
    charts.mkdir(exist_ok=True)

    from .run import select

    last_day = snaps.data.max()
    today = snaps[snaps.data == last_day]
    picks = select(today)[0]

    summary = daily_summary(snaps)
    summary.to_csv(out / "daily_summary.csv", index=False)
    perf = performance(snaps, hist)
    perf.to_csv(out / "performance.csv", index=False)
    sectors = (picks.groupby("setor").agg(oportunidades=("ticker", "count"),
                                          margem_mediana=("margem_seguranca", "median"))
               .reset_index().sort_values("oportunidades", ascending=False))
    sectors.to_csv(out / "sectors_today.csv", index=False)

    chart_top_margin(today, charts / "top15_margem.png")
    chart_price_vs_value(today, charts / "preco_vs_valor.png")
    chart_opportunities_over_time(summary, charts / "oportunidades_por_dia.png")
    chart_market_margin(summary, charts / "margem_mercado.png")
    chart_sectors(sectors, charts / "setores.png")
    top5 = list(picks.ticker.head(5))
    chart_margin_evolution(snaps, top5, charts / "evolucao_top5.png")
    if not perf.empty:
        chart_performance(perf, charts / "desempenho.png")

    s = summary.iloc[-1]
    n_days = summary.data.nunique()
    L = [
        "# Painel de oportunidades em bolsa",
        "",
        f"Atualizado em **{last_day}** · {n_days} dia(s) de histórico · "
        f"[relatório do dia](latest.md) · dados em [`daily_summary.csv`](daily_summary.csv), "
        f"[`performance.csv`](performance.csv), [`history.csv`](history.csv)",
        "",
        "| Indicador | Hoje |",
        "|---|---|",
        f"| Ações analisadas | {s.acoes_analisadas} |",
        f"| Oportunidades (EUA / Europa) | {s.oportunidades} ({s.oportunidades_eua} / {s.oportunidades_europa}) |",
        f"| Lista de observação | {s.observacao} |",
        f"| Margem média do top 15 | {s.margem_media_top15:.0%} |",
        f"| Qualidade média do top 15 | {s.qualidade_media_top15:.0f}/100 |",
        f"| Margem mediana do mercado EUA / Europa | {s.margem_mediana_eua:.0%} / {s.margem_mediana_europa:.0%} |",
        "",
        "## Hoje",
        "",
        "![Top 15 por margem de segurança](charts/top15_margem.png)",
        "",
        "![Preço vs valor intrínseco](charts/preco_vs_valor.png)",
        "",
        "![Oportunidades por setor](charts/setores.png)",
        "",
        "## Ao longo do tempo",
        "",
        "![Oportunidades por dia](charts/oportunidades_por_dia.png)",
        "",
        "![Margem mediana do mercado](charts/margem_mercado.png)",
        "",
        "![Evolução da margem do top 5](charts/evolucao_top5.png)",
        "",
    ]
    if not perf.empty:
        L += [
            "## Desempenho das escolhas",
            "",
            "Retorno em moeda local desde o primeiro dia em que cada ação entrou no top 15, comparado com a "
            "média (pesos iguais) das ~280 ações analisadas no mesmo período. Não inclui dividendos.",
            "",
            f"Média das escolhas: **{perf.retorno.mean():+.1%}** vs universo **{perf.retorno_universo.mean():+.1%}** "
            f"({(perf.excesso > 0).mean():.0%} das escolhas acima do universo).",
            "",
            "![Desempenho](charts/desempenho.png)",
            "",
            "| Ticker | Entrou em | Preço entrada | Preço atual | Retorno | Universo | Diferença | Ainda no top 15 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in perf.itertuples():
            L.append(
                f"| {r.ticker} | {r.entrou_em} | {r.preco_entrada:,.2f} {r.moeda} | {r.preco_atual:,.2f} {r.moeda} | "
                f"{_fmt_pct(r.retorno)} | {_fmt_pct(r.retorno_universo)} | {_fmt_pct(r.excesso)} | "
                f"{'sim' if r.ainda_no_top15 else 'não'} |"
            )
        L.append("")
    L.append("> Triagem quantitativa automática, não é recomendação de investimento.")
    (out / "DASHBOARD.md").write_text("\n".join(L), encoding="utf-8")
