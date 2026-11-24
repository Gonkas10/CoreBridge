"""ETF pessoal "CoreBridge Quality Compounders": composição, métricas, backtest e DCA.

Uso: python etf/fetch.py && python etf/build.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = Path(__file__).parent
DATA, CH = D / "data", D / "charts"
CH.mkdir(exist_ok=True)

MONTHLY_DCA = 300   # € por mês na simulação
RF = 0.02           # taxa sem risco para o rácio de Sharpe
LIMITS = {"setor": 0.22, "regiao": 0.50, "posicao": 0.07}

# ticker: (peso, nome curto, setor, região, vantagem competitiva, principal risco)
HOLDINGS = {
    "MSFT":      (0.06, "Microsoft", "Tecnologia", "EUA",
                  "Azure + Office 365: software crítico com custos de mudança muito altos; distribuição de IA para empresas.",
                  "Investimento em IA pode não ter retorno; concorrência da AWS e Google Cloud."),
    "ASML.AS":   (0.05, "ASML", "Tecnologia", "Europa",
                  "Monopólio mundial em litografia EUV; nenhum concorrente consegue fazer estas máquinas.",
                  "Ciclo dos semicondutores e restrições de exportação para a China."),
    "TSM":       (0.05, "TSMC", "Tecnologia", "Ásia e emergentes",
                  "Fabrica ~90% dos chips mais avançados do mundo; anos de vantagem tecnológica e de escala.",
                  "Risco geopolítico Taiwan/China; novas fábricas fora de Taiwan com margens mais baixas."),
    "CDNS":      (0.03, "Cadence Design", "Tecnologia", "EUA",
                  "Duopólio com a Synopsys no software de desenho de chips; receita recorrente.",
                  "Avaliação exigente; dependência do ciclo de investimento em chips."),
    "6861.T":    (0.03, "Keyence", "Tecnologia", "Ásia e emergentes",
                  "Sensores e automação fabril com margens operacionais acima de 50% e zero dívida.",
                  "Ciclo industrial; crescimento recente mais lento."),
    "GOOGL":     (0.06, "Alphabet", "Comunicação", "EUA",
                  "Pesquisa, YouTube, Android e Google Cloud; dados e escala sem igual; TPUs próprios para IA.",
                  "Pesquisa ameaçada por assistentes de IA; processos de concorrência (antitrust)."),
    "V":         (0.05, "Visa", "Financeiro", "EUA",
                  "Rede de pagamentos com efeito de rede global; margens operacionais ~65%.",
                  "Regulação das comissões; pagamentos instantâneos e stablecoins."),
    "SPGI":      (0.04, "S&P Global", "Financeiro", "EUA",
                  "Ratings (duopólio com a Moody's), índices S&P 500 e dados financeiros.",
                  "Volume de emissões de dívida é cíclico."),
    "DB1.DE":    (0.04, "Deutsche Börse", "Financeiro", "Europa",
                  "Bolsa alemã, câmara de compensação Eurex e Clearstream; infraestrutura difícil de replicar.",
                  "Receitas de juros sobre depósitos de clientes caem se as taxas descerem."),
    "HDB":       (0.03, "HDFC Bank", "Financeiro", "Ásia e emergentes",
                  "Maior banco privado da Índia; aposta no crescimento do crédito e da classe média indiana.",
                  "Integração da fusão com a HDFC Ltd. pressiona margens; risco cambial (rupia)."),
    "LLY":       (0.04, "Eli Lilly", "Saúde", "EUA",
                  "Líder em medicamentos para obesidade/diabetes (tirzepatida) com pipeline forte.",
                  "Avaliação já descontada; concorrência e pressão nos preços dos medicamentos nos EUA."),
    "NOVN.SW":   (0.04, "Novartis", "Saúde", "Europa",
                  "Farmacêutica focada em medicamentos inovadores; balanço sólido e dividendo crescente.",
                  "Fim de patentes (Entresto) e risco de ensaios clínicos."),
    "ISRG":      (0.04, "Intuitive Surgical", "Saúde", "EUA",
                  "Líder em cirurgia robótica (da Vinci); base instalada gera receita recorrente de instrumentos.",
                  "Concorrência de Medtronic e J&J a chegar ao mercado; avaliação alta."),
    "AZN.L":     (0.04, "AstraZeneca", "Saúde", "Europa",
                  "Pipeline líder em oncologia; crescimento acima do setor.",
                  "Pressão de preços nos EUA e na China."),
    "COST":      (0.05, "Costco", "Consumo básico", "EUA",
                  "Modelo de quotas de sócio (renovação >90%) e preços baixos difíceis de igualar.",
                  "Avaliação muito alta (P/E > 40)."),
    "OR.PA":     (0.03, "L'Oréal", "Consumo básico", "Europa",
                  "Líder mundial em cosmética, com marcas fortes e distribuição global.",
                  "Abrandamento na China; concorrência de marcas digitais."),
    "KO":        (0.03, "Coca-Cola", "Consumo básico", "EUA",
                  "Marca e rede de distribuição global; defensiva em recessões; dividendo cresce há 60+ anos.",
                  "Crescimento baixo; tendência para menos açúcar."),
    "RMS.PA":    (0.03, "Hermès", "Consumo discricionário", "Europa",
                  "Luxo absoluto com escassez deliberada; poder de preço e margens >40%.",
                  "Avaliação alta; abrandamento do luxo na China."),
    "RACE.MI":   (0.03, "Ferrari", "Consumo discricionário", "Europa",
                  "Marca única, listas de espera de anos e poder de preço; ROE > 40%.",
                  "Transição para elétricos; avaliação de marca de luxo."),
    "MELI":      (0.03, "MercadoLibre", "Consumo discricionário", "Ásia e emergentes",
                  "Líder em e-commerce e fintech (Mercado Pago) na América Latina; crescimento > 30%.",
                  "Volatilidade de moedas latino-americanas; concorrência da Shopee e Temu."),
    "SU.PA":     (0.05, "Schneider Electric", "Indústria", "Europa",
                  "Gestão de energia e automação; beneficia da eletrificação e dos centros de dados.",
                  "Ciclo de construção e industrial."),
    "ATCO-A.ST": (0.03, "Atlas Copco", "Indústria", "Europa",
                  "Compressores e bombas de vácuo líderes; serviços pós-venda dão receita recorrente.",
                  "Ciclo industrial e de semicondutores."),
    "WM":        (0.03, "Waste Management", "Indústria", "EUA",
                  "Aterros sanitários são ativos quase impossíveis de licenciar de novo; preços sobem com a inflação.",
                  "Crescimento moderado; dívida."),
    "LIN":       (0.05, "Linde", "Materiais", "EUA",
                  "Oligopólio dos gases industriais; contratos de 15-20 anos com cláusulas de inflação.",
                  "Ciclo industrial; projetos de hidrogénio com retorno incerto."),
    "IBE.MC":    (0.04, "Iberdrola", "Utilities", "Europa",
                  "Maior utility renovável da Europa; redes elétricas reguladas nos EUA, Reino Unido e Espanha.",
                  "Sensível às taxas de juro e ao risco regulatório."),
}

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def holdings_df():
    h = pd.DataFrame([(t, *v) for t, v in HOLDINGS.items()],
                     columns=["ticker", "peso", "nome", "setor", "regiao", "moat", "risco"])
    assert abs(h.peso.sum() - 1) < 1e-9, f"pesos somam {h.peso.sum():.3f}"
    return h


def check_limits(h):
    out = []
    for col, lim in (("setor", LIMITS["setor"]), ("regiao", LIMITS["regiao"])):
        s = h.groupby(col).peso.sum()
        out.append((col, s.max(), lim, s.idxmax()))
    out.append(("posicao", h.peso.max(), LIMITS["posicao"], h.loc[h.peso.idxmax(), "ticker"]))
    for col, val, lim, who in out:
        assert val <= lim + 1e-9, f"limite {col} violado: {who} {val:.0%} > {lim:.0%}"
    return out


def perf_stats(series):
    s = series.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    yearly = s.resample("YE").last().pct_change().dropna()
    return {"CAGR": cagr, "Volatilidade": vol, "Queda máxima": (s / s.cummax() - 1).min(),
            "Sharpe": (cagr - RF) / vol, "Pior ano": yearly.min(), "Melhor ano": yearly.max()}


def backtest(prices, weights):
    """Carteira rebalanceada em janeiro de cada ano para os pesos-alvo."""
    rets = prices.pct_change().fillna(0)
    value, w = 1.0, weights.copy()
    vals = []
    for i, (d, r) in enumerate(rets.iterrows()):
        if i > 0:
            gross = (w * (1 + r)).sum()
            value *= gross
            w = w * (1 + r) / gross
            if d.month == 1:
                w = weights.copy()
        vals.append(value)
    return pd.Series(vals, index=prices.index)


def dca(index_series, monthly):
    """Investe `monthly` € no início de cada mês e devolve (valor, total investido)."""
    units, invested, vals, inv = 0.0, 0.0, [], []
    for p in index_series:
        units += monthly / p
        invested += monthly
        vals.append(units * p)
        inv.append(invested)
    return pd.Series(vals, index=index_series.index), pd.Series(inv, index=index_series.index)


def main():
    h = holdings_df()
    limits = check_limits(h)
    f = pd.read_csv(DATA / "fundamentals.csv").set_index("ticker")
    px = pd.read_csv(DATA / "prices_eur_monthly.csv", index_col=0, parse_dates=True)
    px = px[px.index >= px.index[-1] - pd.DateOffset(years=10)].ffill()

    tick = list(h.ticker)
    w = h.set_index("ticker").peso
    port = backtest(px[tick], w)
    world = px["EUNL.DE"] / px["EUNL.DE"].iloc[0]
    sp = px["SXR8.DE"] / px["SXR8.DE"].iloc[0]

    stats = pd.DataFrame({"ETF CoreBridge": perf_stats(port), "MSCI World (EUNL)": perf_stats(world),
                          "S&P 500 (SXR8)": perf_stats(sp)})
    # Robustez: e se as 3 melhores ações dos últimos 10 anos não tivessem sido escolhidas?
    tot = (px[tick].iloc[-1] / px[tick].iloc[0]).sort_values(ascending=False)
    top3 = list(tot.index[:3])
    w_ex = w.drop(top3) / w.drop(top3).sum()
    port_ex = backtest(px[w_ex.index], w_ex)
    stats["Sem as 3 melhores"] = perf_stats(port_ex)
    d_port, inv = dca(port, MONTHLY_DCA)
    d_world, _ = dca(world, MONTHLY_DCA)

    # Métricas fundamentais ponderadas
    m = f.reindex(tick)
    jp = m.index.str.endswith(".T")
    m.loc[jp, "forwardPE"] = np.nan  # P/E previsto do Yahoo para ações japonesas vem com moedas trocadas
    def wavg(col, clip=None):
        s = m[col].astype(float)
        if clip:
            s = s.clip(*clip)
        ok = s.notna()
        return (s[ok] * w[ok]).sum() / w[ok].sum()
    fund = {
        "Margem bruta": wavg("grossMargins"), "Margem operacional": wavg("operatingMargins"),
        "Margem líquida": wavg("profitMargins"), "ROE (limitado a 100%)": wavg("returnOnEquity", (0, 1)),
        "Crescimento de receitas (último ano)": wavg("revenueGrowth"),
        "Crescimento anual de receitas (histórico, ~3 anos)": wavg("rev_cagr_hist"),
        "P/E previsto (média ponderada)": wavg("forwardPE"), "Dividend yield (%)": wavg("dividendYield"),
    }
    corr = px[tick].pct_change().corr().values
    avg_corr = corr[np.triu_indices_from(corr, 1)].mean()

    ind = pd.DataFrame({t: perf_stats(px[t] / px[t].iloc[0]) for t in tick}).T
    table = h.set_index("ticker").join(ind).join(m[["returnOnEquity", "operatingMargins", "rev_cagr_hist", "forwardPE"]])
    table.to_csv(D / "holdings.csv")

    # ---------- gráficos ----------
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for s, lab, c in ((port, "ETF CoreBridge", PALETTE[0]), (world, "MSCI World", PALETTE[1]), (sp, "S&P 500", PALETTE[2])):
        ax.plot(s.index, 10000 * s, color=c, linewidth=2, label=f"{lab}: {10000 * s.iloc[-1]:,.0f} €")
    ax.set_title("10 000 € investidos há 10 anos (em euros, dividendos reinvestidos)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f} €"))
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(CH / "crescimento.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(inv.index, inv, color=INK2, linewidth=1.5, linestyle="--", label=f"Total investido: {inv.iloc[-1]:,.0f} €")
    ax.plot(d_port.index, d_port, color=PALETTE[0], linewidth=2, label=f"ETF CoreBridge: {d_port.iloc[-1]:,.0f} €")
    ax.plot(d_world.index, d_world, color=PALETTE[1], linewidth=2, label=f"MSCI World: {d_world.iloc[-1]:,.0f} €")
    ax.set_title(f"DCA de {MONTHLY_DCA} € por mês durante 10 anos")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f} €"))
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(CH / "dca.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for s, lab, c in ((port, "ETF CoreBridge", PALETTE[0]), (world, "MSCI World", PALETTE[1])):
        ax.fill_between(s.index, s / s.cummax() - 1, 0, color=c, alpha=0.25)
        ax.plot(s.index, s / s.cummax() - 1, color=c, linewidth=1.5, label=lab)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Quedas desde o máximo anterior")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(CH / "quedas.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in ((axes[0], "setor", "Peso por setor"), (axes[1], "regiao", "Peso por região")):
        s = h.groupby(col).peso.sum().sort_values()
        ax.barh(s.index, s.values, color=PALETTE[0] if col == "setor" else PALETTE[1], height=0.65)
        for y, v in enumerate(s.values):
            ax.text(v + 0.003, y, f"{v:.0%}", va="center", fontsize=9, color=INK2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(CH / "alocacao.png", dpi=150); plt.close(fig)

    # ---------- relatório ----------
    def pct(x, nd=1):
        return "—" if pd.isna(x) else f"{x:.{nd}%}"

    L = ["# ETF pessoal CoreBridge: Quality Compounders", ""]
    L += [
        f"Carteira de **{len(h)} empresas** de qualidade, com vantagens competitivas duradouras, pensada para "
        "investir regularmente (DCA) e manter durante muitos anos. Dados de mercado: Yahoo Finance, "
        f"até {px.index[-1]:%m/%Y}, todos os valores em euros.",
        "",
        "> ⚠️ Não é aconselhamento financeiro personalizado. O histórico abaixo tem **viés de retrospetiva**: as "
        "empresas foram escolhidas hoje, sabendo que correram bem. Os próximos 10 anos dificilmente serão tão "
        "bons. Lê a secção [Limitações](#limitações).",
        "",
        "## Composição",
        "",
        "| Empresa | Ticker | Peso | Setor | Região | Vantagem competitiva | Principal risco |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in h.sort_values("peso", ascending=False).itertuples():
        L.append(f"| **{r.nome}** | {r.ticker} | {r.peso:.0%} | {r.setor} | {r.regiao} | {r.moat} | {r.risco} |")
    L += ["", "![Alocação](charts/alocacao.png)", "",
          f"**Limites de diversificação** (verificados no código): nenhum setor acima de {LIMITS['setor']:.0%}, "
          f"nenhuma região acima de {LIMITS['regiao']:.0%}, nenhuma posição acima de {LIMITS['posicao']:.0%}. "
          f"O ETF MSCI World tem cerca de 70% em ações dos EUA; aqui os EUA pesam "
          f"{h[h.regiao == 'EUA'].peso.sum():.0%}. Correlação média entre as ações: {avg_corr:.2f}.", ""]

    L += ["## Qualidade dos negócios (média ponderada)", "", "| Indicador | ETF CoreBridge |", "|---|---|"]
    for k, v in fund.items():
        if "P/E" in k:
            L.append(f"| {k} | {v:.1f} |")
        elif "Dividend" in k:
            L.append(f"| {k} | {v:.1f}% |")
        else:
            L.append(f"| {k} | {pct(v)} |")
    L += ["", "Para comparação, o MSCI World tem uma margem líquida próxima de 12% e um P/E previsto entre 19 e 20. "
          "Paga-se um prémio por empresas com margens e rentabilidade bem acima da média, e é esse o risco "
          "principal da carteira: se o crescimento abrandar, os múltiplos podem descer.", ""]

    L += ["## Histórico (10 anos, em euros)", "",
          "| | " + " | ".join(stats.columns) + " |", "|---|" + "---|" * len(stats.columns)]
    for k in stats.index:
        row = [f"{v:.2f}" if k == "Sharpe" else pct(v) for v in stats.loc[k]]
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L += ["", f"A coluna **Sem as 3 melhores** retira {', '.join(top3)} (as que mais subiram) e redistribui o peso. "
          "Mostra quanto do resultado depende de ter acertado nos maiores vencedores: é uma estimativa mais "
          "realista do que esperar de uma carteira de qualidade, mas continua a sofrer de viés de retrospetiva.",
          "", "![Crescimento](charts/crescimento.png)", "", "![Quedas](charts/quedas.png)", "",
          f"### Simulação de DCA ({MONTHLY_DCA} €/mês)", "",
          f"Investindo {MONTHLY_DCA} € no início de cada mês durante 10 anos ({inv.iloc[-1]:,.0f} € no total), "
          f"o ETF CoreBridge valeria **{d_port.iloc[-1]:,.0f} €** e o MSCI World **{d_world.iloc[-1]:,.0f} €**.", "",
          "![DCA](charts/dca.png)", ""]

    L += ["## Métricas por empresa", "",
          "| Ticker | Peso | Retorno anual 10a | Volatilidade | Queda máxima | ROE | Margem operacional | Cresc. receitas (hist.) | P/E previsto |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, r in table.sort_values("peso", ascending=False).iterrows():
        pe = "n.d." if pd.isna(r.forwardPE) else f"{r.forwardPE:.1f}"
        roe = "n.d." if pd.isna(r.returnOnEquity) else ("> 100%" if r.returnOnEquity > 1 else pct(r.returnOnEquity, 0))
        L.append(f"| {t} | {r.peso:.0%} | {pct(r.CAGR)} | {pct(r.Volatilidade, 0)} | {pct(r['Queda máxima'], 0)} | "
                 f"{roe} | {pct(r.operatingMargins, 0)} | {pct(r.rev_cagr_hist)} | {pe} |")
    L += ["", "ROE acima de 100% resulta de capital próprio muito baixo por recompras de ações (ex.: Visa, Coca-Cola), "
          "não de lucros anormais.", ""]

    L += ["## Porque ficaram de fora", "",
          "- **Nvidia e Broadcom**: excelentes negócios, mas já pesam muito em qualquer ETF mundial; a exposição a IA "
          "vem via TSMC, ASML, Microsoft, Alphabet e Cadence, com menos risco de concentração.",
          "- **Novo Nordisk**: queda de 76% desde o máximo; a concorrência da Lilly e dos genéricos de semaglutido "
          "enfraqueceu a tese. A exposição à obesidade fica na Lilly.",
          "- **Tencent**: negócio forte, mas risco regulatório e geopolítico chinês difícil de avaliar.",
          "- **Adobe**: ameaça direta de ferramentas de IA generativa ao seu núcleo criativo.",
          "- **LVMH, Nestlé, Zoetis, Wolters Kluwer**: crescimento estagnado ou quedas superiores a 50% sem sinais "
          "claros de recuperação.", ""]

    L += ["## Como usar (DCA)", "",
          "1. **Corretora**: uma que permita frações de ações e acesso aos EUA, Europa e Japão (ex.: Interactive Brokers, "
          "Trade Republic, DEGIRO). As ações japonesas e a HDFC (ADR) podem não estar disponíveis em todas.",
          "2. **Reforços**: em cada reforço, investe nas 3-5 posições que estiverem **mais abaixo do peso-alvo**. "
          "Assim rebalanceias sem vender e compras automaticamente o que caiu.",
          "3. **Rebalanceamento**: uma vez por ano, só se uma posição se desviar mais de 50% do peso-alvo "
          "(ex.: 5% que passou a 7,5%). Vender gera imposto.",
          "4. **Revisão da tese**: uma vez por ano, confirma para cada empresa que a vantagem competitiva continua de pé. "
          "Sai se a tese se partir, não por o preço ter caído.",
          "5. **Custos e impostos (Portugal)**: mais-valias e dividendos são tributados a 28% (ou englobados). "
          "Os dividendos estrangeiros sofrem retenção na fonte; os EUA retêm 15% com o formulário W-8BEN. Com 25 posições, "
          "comissões fixas pesam: prefere reforços de pelo menos 500-1 000 € ou corretoras sem comissão.",
          "6. **Alternativa núcleo e satélite**: se preferires menos trabalho, 60% num ETF MSCI World e 40% nesta "
          "carteira dá diversificação máxima com um perfil de qualidade.", ""]

    L += ["## Limitações", "",
          "- **Viés de retrospetiva**: o backtest usa as empresas que hoje sabemos serem vencedoras. Uma carteira escolhida "
          "assim em 2016 teria incluído algumas que falharam. Trata o resultado histórico como um limite superior otimista.",
          "- **Avaliação**: estas empresas negociam com prémio face ao mercado. Retornos futuros dependem de o crescimento "
          "dos lucros justificar esses múltiplos.",
          "- **Moeda**: cerca de metade da carteira está em USD; um euro mais forte reduz o retorno em euros.",
          "- **Dados**: Yahoo Finance; alguns indicadores (ex.: P/E das ações japonesas) vêm com erros e foram excluídos.",
          "- As narrativas de vantagem competitiva são análise qualitativa e devem ser revistas anualmente.", "",
          "Ficheiros: [`holdings.csv`](holdings.csv) · código em [`build.py`](build.py) e [`fetch.py`](fetch.py) · "
          "para atualizar: `python etf/fetch.py && python etf/build.py`."]
    (D / "README.md").write_text("\n".join(L), encoding="utf-8")

    print(stats.round(3).to_string())
    print({k: round(v, 3) for k, v in fund.items()}, "corr", round(avg_corr, 2))
    print(h.groupby("setor").peso.sum().round(2).to_dict())
    print(h.groupby("regiao").peso.sum().round(2).to_dict())
    print("DCA", round(d_port.iloc[-1]), round(d_world.iloc[-1]), round(inv.iloc[-1]), limits)


if __name__ == "__main__":
    main()
