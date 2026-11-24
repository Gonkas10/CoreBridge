# CoreBridge — Oportunidades diárias em bolsa

Screener automático que, todos os dias úteis, analisa ~280 grandes empresas dos **EUA e da Europa**
e identifica ações que negoceiam **abaixo do valor intrínseco estimado**, com boa qualidade
financeira e sem sinais claros de *value trap*.

- **ETF pessoal (carteira de 25 empresas para DCA): [`etf/README.md`](etf/README.md)**
- **Painel com gráficos e evolução histórica: [`reports/DASHBOARD.md`](reports/DASHBOARD.md)**
- Relatório do dia: [`reports/latest.md`](reports/latest.md)
- Histórico de relatórios: `reports/AAAA-MM-DD.md`
- Dados completos de todas as ações: `reports/latest_full.csv`
- Histórico das escolhas (para medir resultados): `reports/history.csv`
- Indicadores diários (EUA vs Europa, nº de oportunidades): `reports/daily_summary.csv`
- Desempenho de cada escolha vs. universo: `reports/performance.csv`
- Snapshot completo de cada dia: `reports/data/snapshots/AAAA-MM-DD.csv`

Os CSV são públicos e alimentam também a folha Google Sheets (via `IMPORTDATA`), que se
atualiza sozinha depois de cada corrida diária.

## Como correr

```bash
pip install -r requirements.txt
python -m screener.run            # gera reports/latest.md
python -m screener.run --limit 20 # teste rápido
```

## Como funciona

1. **Dados**: Yahoo Finance (`yfinance`). Lucros e free cash flow são convertidos para a moeda de
   reporte de cada empresa antes de comparar com o preço (evita erros como Shell em pence vs USD).
2. **Valor intrínseco** = mediana de três métodos (mínimo dois disponíveis):
   - DCF de 10 anos sobre o free cash flow (bancos e seguradoras: P/B justificado pelo ROE);
   - Número de Graham `√(22,5 × EPS × valor contabilístico)`;
   - Fórmula de crescimento de Graham `EPS × (8,5 + 2g) × 4,4 / Y`.
3. **Pressupostos conservadores**: crescimento = menor entre lucros, receitas e EPS previsto,
   limitado a −2%…8% (energia/matérias-primas: máx. 3%); taxa de desconto 4% + β×5,5%.
4. **Qualidade (0–100)**: ROE, margens, dívida, FCF, liquidez, crescimento.
5. **Alertas de risco**: quedas fortes, lucros/receitas a cair, payout > 100%, dívida alta,
   analistas negativos.
6. **Oportunidade** = margem de segurança ≥ 20%, qualidade ≥ 55 e no máximo 1 alerta.

Os limiares estão no topo de `screener/run.py` e os modelos em `screener/valuation.py`.

> ⚠️ Triagem quantitativa automática, não é recomendação de investimento. Os dados podem ter
> erros. Antes de comprar, lê os relatórios da empresa e confirma a tese.
