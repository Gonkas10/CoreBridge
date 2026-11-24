"""Modelos de valor intrínseco e de qualidade.

Todos os valores por ação estão na moeda em que a empresa reporta (financialCurrency),
para não misturar moedas (ex.: Shell cota em pence mas reporta em USD).
"""
from __future__ import annotations

import math
from statistics import median

RISK_FREE = 0.04        # taxa sem risco aproximada
EQUITY_PREMIUM = 0.055  # prémio de risco de mercado
TERMINAL_G = 0.025      # crescimento perpétuo
AAA_YIELD = 0.05        # rendibilidade obrigações AAA (fórmula de Graham)

FINANCIAL_SECTORS = {"Financial Services"}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def num(x):
    """Converte para float, devolvendo None para valores em falta/NaN."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def discount_rate(beta):
    beta = num(beta)
    beta = 1.0 if beta is None else clamp(beta, 0.6, 1.8)
    return clamp(RISK_FREE + beta * EQUITY_PREMIUM, 0.075, 0.12)


CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
MAX_G = 0.08          # crescimento máximo assumido
MAX_G_CYCLICAL = 0.03 # cíclicas: lucros atuais podem estar num pico


def growth_estimate(earnings_growth, revenue_growth, fwd_ratio=None, sector=None):
    """Crescimento conservador para os próximos 5 anos.

    Usa o menor entre crescimento de lucros, de receitas e o implícito no EPS previsto pelos
    analistas (forward/trailing), para não extrapolar picos de um só ano.
    """
    vals = [v for v in (num(earnings_growth), num(revenue_growth)) if v is not None]
    fr = num(fwd_ratio)
    if fr is not None and fr > 0:
        vals.append(fr - 1)
    g = min(vals) if vals else 0.03
    cap = MAX_G_CYCLICAL if sector in CYCLICAL_SECTORS else MAX_G
    return clamp(g, -0.02, cap)


def dcf_per_share(fcf_ps, g, r, years_high=5, years_fade=5):
    """DCF de 10 anos: 5 anos a g, 5 anos a convergir para o crescimento terminal."""
    if fcf_ps is None or fcf_ps <= 0 or r <= TERMINAL_G:
        return None
    value, cf = 0.0, fcf_ps
    total = years_high + years_fade
    for t in range(1, total + 1):
        if t <= years_high:
            gt = g
        else:
            k = (t - years_high) / years_fade
            gt = g + (TERMINAL_G - g) * k
        cf *= 1 + gt
        value += cf / (1 + r) ** t
    terminal = cf * (1 + TERMINAL_G) / (r - TERMINAL_G)
    value += terminal / (1 + r) ** total
    return value


def graham_number(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)


def graham_growth(eps, g):
    """Fórmula revista de Graham: V = EPS × (8.5 + 2g) × 4.4 / Y."""
    if eps is None or eps <= 0:
        return None
    g_pct = clamp(g * 100, 0, 7)
    return eps * (8.5 + 2 * g_pct) * 4.4 / (AAA_YIELD * 100)


def justified_pb(bvps, roe, r, g):
    """Para bancos/seguradoras: P/B justificado = (ROE - g) / (r - g)."""
    if bvps is None or roe is None or bvps <= 0 or roe <= 0:
        return None
    g = min(g, TERMINAL_G + 0.01)
    roe = min(roe, 0.25)
    if r <= g:
        return None
    return bvps * max(roe - g, 0) / (r - g)


def intrinsic_value(d: dict) -> tuple[float | None, dict]:
    """Devolve (valor intrínseco por ação, detalhe por método)."""
    r = discount_rate(d.get("beta"))
    g = growth_estimate(d.get("earningsGrowth"), d.get("revenueGrowth"),
                        d.get("fwd_ratio"), d.get("sector"))
    methods: dict[str, float] = {}
    is_fin = d.get("sector") in FINANCIAL_SECTORS

    if is_fin:
        v = justified_pb(d.get("bvps"), num(d.get("returnOnEquity")), r, g)
        if v:
            methods["P/B justificado"] = v
    else:
        v = dcf_per_share(d.get("fcf_ps"), g, r)
        if v:
            methods["DCF"] = v
    v = graham_number(d.get("eps"), d.get("bvps"))
    if v:
        methods["Graham nº"] = v
    v = graham_growth(d.get("eps"), g)
    if v:
        methods["Graham cresc."] = v

    # Exige pelo menos 2 métodos para evitar conclusões com base num só número
    if len(methods) < 2:
        return None, {"r": r, "g": g, "methods": methods}
    return median(methods.values()), {"r": r, "g": g, "methods": methods}


def quality_score(d: dict) -> tuple[int, list[str]]:
    """Pontuação 0-100 de qualidade financeira e lista de alertas (possíveis value traps)."""
    pts, maxpts = 0, 0
    is_fin = d.get("sector") in FINANCIAL_SECTORS

    def add(cond_hi, cond_lo, w):
        nonlocal pts, maxpts
        maxpts += w
        if cond_hi:
            pts += w
        elif cond_lo:
            pts += w / 2

    roe = num(d.get("returnOnEquity"))
    pm = num(d.get("profitMargins"))
    de = num(d.get("debtToEquity"))
    cr = num(d.get("currentRatio"))
    rg = num(d.get("revenueGrowth"))
    eg = num(d.get("earningsGrowth"))

    add(roe is not None and roe > 0.15, roe is not None and roe > 0.10, 20)
    add(pm is not None and pm > 0.15, pm is not None and pm > 0.08, 15)
    if not is_fin:
        add(de is not None and de < 50, de is not None and de < 100, 20)
        add((d.get("fcf_ps") or 0) > 0, False, 15)
        add(cr is not None and cr > 1.5, cr is not None and cr > 1.0, 10)
    add(rg is not None and rg > 0.03, rg is not None and rg > 0, 10)
    add(eg is not None and eg > 0.03, eg is not None and eg > 0, 10)

    flags = []
    chg = num(d.get("52WeekChange"))
    if chg is not None and chg < -0.30:
        flags.append(f"caiu {chg:.0%} em 12 meses")
    if eg is not None and eg < -0.20:
        flags.append(f"lucros a cair {eg:.0%}")
    if rg is not None and rg < -0.05:
        flags.append(f"receitas a cair {rg:.0%}")
    payout = num(d.get("payoutRatio"))
    if payout is not None and payout > 1.0:
        flags.append(f"payout {payout:.0%} (dividendo > lucros)")
    if not is_fin and de is not None and de > 150:
        flags.append(f"dívida elevada (D/E {de:.0f}%)")
    rec = num(d.get("recommendationMean"))
    if rec is not None and rec >= 3.0:
        flags.append("analistas negativos")

    score = round(100 * pts / maxpts) if maxpts else 0
    return score, flags
