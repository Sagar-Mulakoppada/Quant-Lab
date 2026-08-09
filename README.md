# ⚡ AlphaSignal Engine

> **Systematic Equity Factor Research Platform** — Ingest raw OHLCV market data, compute 10 cross-sectional alpha factors, evaluate signals in an institutional-grade Factor Research Lab, and visualize results on an interactive dashboard.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-ff4b4b.svg)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10%2B-fff000.svg)](https://duckdb.org/)

---

## 📌 Overview

Most quantitative research code bases are fragmented across ad-hoc Jupyter Notebooks. **AlphaSignal Engine** unifies equity factor engineering into a structured, production-ready pipeline:

1. **Incremental Data Pipeline**: Autonomous OHLCV data ingestion via `yfinance` stored in local columnar `DuckDB`.
2. **Alpha Factor Library**: 10 cross-sectional factors (momentum, reversal, volatility, trend, volume, liquidity) calculated without lookahead bias.
3. **Factor Research Lab**: In-depth quantitative analytics including Spearman Information Coefficient (IC), ICIR, factor half-life decay, quintile performance, turnover, and monthly IC consistency heatmaps.
4. **Interactive Quant Dashboard**: A dark-themed Streamlit application with live factor analysis, interactive Plotly charts, and stock universe inspection.

---

## 🏗️ Architecture & Project Structure

```
alpha-signal-engine/
├── src/
│   ├── ingest/
│   │   └── market_data.py      # DuckDB pipeline & yfinance incremental loader
│   ├── factors/
│   │   ├── price_factors.py    # 10 cross-sectional alpha factor definitions
│   │   └── factor_lab.py       # ★ Quantitative evaluation & Factor Lab engine
│   ├── backtest/               # Custom backtesting engine (Planned)
│   ├── ml/                     # Machine learning meta-signals & SHAP (Planned)
│   └── optimizer/              # Portfolio optimization engine (Planned)
├── dashboard/
│   └── app.py                  # Dark-themed Streamlit research dashboard
├── notebooks/
│   └── research_report.ipynb   # Analytical research notebook
├── tests/                      # Unit test suites
├── data/                       # Local DuckDB store directory
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 🔬 Factor Research Lab Metrics

Every factor registered in the Lab undergoes rigorous cross-sectional signal evaluation:

| Metric | Description | Benchmark / Target |
|---|---|---|
| **IC (Information Coefficient)** | Spearman rank correlation between factor value $S_t$ and forward return $R_{t+1}$ | $> 0.04$ indicates strong predictive power |
| **ICIR** | Annualized Information Ratio of IC ($\frac{\text{Mean IC}}{\text{Std IC}} \times \sqrt{252}$) | $> 0.5$ indicates signal consistency |
| **IC Positive %** | Percentage of trading days where the signal correctly predicted directional ranking | $> 50\%$ |
| **Factor Decay** | IC measured across $1, 5, 10, 20, 60$ day forward horizons | Identifies signal decay & half-life |
| **L/S Sharpe** | Annualized Sharpe Ratio of a top-quintile minus bottom-quintile rebalanced portfolio | Evaluates tradeability & returns |
| **Average Turnover** | Mean percentage of long portfolio replaced during each rebalance period | Lower turnover = lower transaction drag |
| **Cross-Factor Correlation** | Pairwise Spearman rank correlation matrix across factors | Prevents signal redundancy |
| **Monthly IC Heatmap** | Calendar heat grid of monthly mean IC | Checks stability across macro regimes |

---

## 📊 Included Alpha Factors

| Factor Name | Category | Academic / Empirical Basis |
|---|---|---|
| **Momentum 12-1** | Price Momentum | Jegadeesh & Titman (1993) |
| **Short-Term Reversal** | Mean Reversion | Lehmann (1990) |
| **Low Volatility** | Risk Anomaly | Baker et al. (2011) |
| **RSI Oscillator** | Technical | Wilder (1978) |
| **RSI Divergence** | Signal Divergence | Technical divergence of price vs momentum |
| **MACD Histogram** | Trend / Momentum | Appel (1979) |
| **52-Week High Proximity** | Price Trend | George & Hwang (2004) |
| **Bollinger Band Position** | Volatility / Trend | Bollinger (1992) |
| **Price-Volume Trend (PVT)** | Volume Confirmation | Cumulative volume-weighted price change |
| **Amihud Liquidity** | Market Microstructure | Amihud (2002) |

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/your-username/alpha-signal-engine.git
cd alpha-signal-engine

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launch the Research Dashboard

Run the Streamlit dashboard app (market data is fetched and stored automatically on first run):

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser to inspect the interactive UI.

---

## 🧠 Tech Stack

- **Data Management**: `yfinance` for data fetching, `DuckDB` for serverless columnar SQL storage.
- **Quantitative Engine**: `pandas`, `numpy`, `scipy` for vectorized cross-sectional factor computations.
- **Analytics & ML**: `scikit-learn`, `XGBoost`, `SHAP`, `hmmlearn` *(planned integrations)*.
- **Visualization & UI**: `Streamlit`, `Plotly`, `Seaborn`.
- **Testing**: `pytest`.

---

## 🗺️ Development Roadmap

- [x] Columnar DuckDB data storage & incremental refresh pipeline
- [x] Cross-sectional price & volume factor library (10 factors)
- [x] Factor Research Lab evaluation engine (IC, ICIR, Decay, Quintiles)
- [x] Interactive dark-themed Streamlit dashboard with Plotly charts
- [ ] Portfolio backtesting engine with realistic slippage & transaction cost model
- [ ] ML Meta-Signal blending using XGBoost and SHAP feature importance
- [ ] Market regime detection using Hidden Markov Models (HMM)
- [ ] Portfolio optimization module (Mean-Variance & Risk Parity)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
