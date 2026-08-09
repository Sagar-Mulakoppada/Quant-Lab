"""
app.py — AlphaSignal Engine · Factor Research Lab Dashboard
============================================================
Run with:  streamlit run dashboard/app.py

Tabs
----
1. Factor Research Lab  (the star feature)
2. Backtest Results
3. Data Explorer
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.market_data import MarketDataPipeline, SP500_SUBSET
from src.factors import price_factors as pf
from src.factors.factor_lab import FactorLab

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AlphaSignal Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark quant aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .main { background-color: #0d0f14; }
  .stApp { background-color: #0d0f14; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111520 0%, #0d0f14 100%);
    border-right: 1px solid #1e2235;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #13162b;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 16px 20px;
  }
  [data-testid="stMetricLabel"] { color: #8892b0; font-size: 0.75rem; letter-spacing: 0.06em; }
  [data-testid="stMetricValue"] { color: #e6f1ff; font-size: 1.6rem; font-weight: 600; }
  [data-testid="stMetricDelta"] { font-size: 0.8rem; }

  /* Headers */
  h1, h2, h3 { color: #ccd6f6 !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] {
    color: #8892b0;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    color: #64ffda !important;
    border-bottom: 2px solid #64ffda;
  }

  /* Dataframe */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Selectbox */
  .stSelectbox label { color: #8892b0; }

  /* Status badges */
  .badge-green {
    background: #0d3326; color: #64ffda;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
  }
  .badge-red {
    background: #3d1515; color: #ff6b6b;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
  }

  /* Section card */
  .section-card {
    background: #13162b;
    border: 1px solid #1e2235;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
  }

  /* Divider */
  hr { border-color: #1e2235; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ AlphaSignal Engine")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    rebal_freq = st.selectbox("Rebalance Frequency", ["ME", "W"], index=0,
                               format_func=lambda x: {"ME": "Monthly", "W": "Weekly"}[x])
    tcost_bps = st.slider("Transaction Cost (bps)", 0, 30, 10)
    n_tickers = st.slider("Universe Size (tickers)", 20, 50, 30)

    st.markdown("---")
    st.markdown("### 🔄 Data")
    refresh_btn = st.button("Refresh Market Data", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#8892b0'>Built with ❤️ — AlphaSignal Engine v0.1</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_data(n_tickers: int):
    pipeline = MarketDataPipeline(db_path="data/alpha.db")
    tickers = SP500_SUBSET[:n_tickers]
    pipeline.refresh(tickers=tickers, start="2019-01-01")
    prices = pipeline.load_prices().dropna(axis=1, thresh=200)
    volumes = pipeline.load_volumes().reindex(columns=prices.columns).fillna(0)
    pipeline.close()
    return prices, volumes

if refresh_btn:
    st.cache_data.clear()

with st.spinner("Connecting to data pipeline..."):
    try:
        prices, volumes = load_data(n_tickers)
        data_ok = True
    except Exception as e:
        st.error(f"Data pipeline error: {e}")
        data_ok = False

# ---------------------------------------------------------------------------
# Factor computation (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Computing factors...")
def compute_factors(_prices, _volumes):
    return {
        "Momentum 12-1":     pf.momentum_12_1(_prices),
        "Short-Term Rev.":   pf.short_term_reversal(_prices),
        "Low Volatility":    pf.realized_volatility(_prices),
        "RSI":               pf.rsi(_prices),
        "RSI Divergence":    pf.rsi_divergence(_prices),
        "MACD":              pf.macd_signal(_prices),
        "52W High":          pf.price_to_52w_high(_prices),
        "Bollinger Pos.":    pf.bollinger_position(_prices),
        "PVT":               pf.price_volume_trend(_prices, _volumes),
        "Amihud Liquidity":  pf.amihud_illiquidity(_prices, _volumes),
    }

@st.cache_data(ttl=3600, show_spinner="Running Factor Research Lab...")
def run_lab(_prices, _factors, rebal_freq, tcost_bps):
    lab = FactorLab(
        prices=_prices,
        factors=_factors,
        rebal_freq=rebal_freq,
        tcost_bps=tcost_bps,
    )
    return lab.run()

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.markdown("# ⚡ AlphaSignal Engine")
st.markdown(
    "<p style='color:#8892b0;margin-top:-12px'>Systematic equity factor research platform</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not data_ok:
    st.stop()

factors = compute_factors(prices, volumes)
report = run_lab(prices, factors, rebal_freq, tcost_bps)

tab_lab, tab_backtest, tab_data = st.tabs([
    "🔬 Factor Research Lab",
    "📈 Backtest Results",
    "🗄️ Data Explorer",
])

# ===========================================================================
# TAB 1 — FACTOR RESEARCH LAB
# ===========================================================================

with tab_lab:
    st.markdown("## 🔬 Factor Research Lab")
    st.markdown(
        "<p style='color:#8892b0'>Full analytical profile for every registered factor -- "
        "IC, Sharpe, Turnover, Decay, and Cross-Factor Correlation.</p>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    st.markdown("### 📊 Factor Summary Table")

    summary = report.summary_table.copy()

    # Color-code the dataframe
    def color_ic(val):
        try:
            v = float(val)
            if v > 0.04: return "color: #64ffda; font-weight: 600"
            if v < 0: return "color: #ff6b6b"
        except: pass
        return ""

    def color_sharpe(val):
        try:
            v = float(val)
            if v > 0.8: return "color: #64ffda; font-weight: 600"
            if v < 0: return "color: #ff6b6b"
        except: pass
        return ""

    styled = (
        summary.style
        .map(color_ic, subset=["Mean IC", "IC@1d", "IC@5d", "IC@20d", "IC@60d"])
        .map(color_sharpe, subset=["L/S Sharpe"])
        .background_gradient(subset=["ICIR"], cmap="RdYlGn", vmin=-1, vmax=1)
        .format(precision=4)
    )
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Factor selector for detailed drill-down
    # -----------------------------------------------------------------------
    col_sel, col_meta = st.columns([1, 3])
    with col_sel:
        selected_factor = st.selectbox(
            "Select Factor for Deep-Dive",
            list(report.factor_stats.keys()),
        )

    fs = report.factor_stats[selected_factor]

    with col_meta:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mean IC", f"{fs.mean_ic:.4f}",
                  delta="Good" if fs.mean_ic > 0.04 else "Weak")
        m2.metric("ICIR", f"{fs.icir:.3f}",
                  delta="Good" if fs.icir > 0.5 else "Weak")
        m3.metric("IC Positive %", f"{fs.ic_positive_pct*100:.1f}%")
        m4.metric("L/S Sharpe", f"{fs.ls_sharpe:.3f}")
        m5.metric("Avg Turnover", f"{fs.avg_turnover*100:.1f}%")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Row 1: IC timeseries + IC distribution
    # -----------------------------------------------------------------------
    col_ic_ts, col_ic_hist = st.columns([2, 1])

    with col_ic_ts:
        st.markdown(f"#### Information Coefficient — {selected_factor}")
        ic_s = fs.ic_series.dropna()
        rolling_ic = ic_s.rolling(21).mean()

        # Build per-bar colors (Plotly requires 6-char hex or rgba())
        bar_colors = [
            "rgba(100,255,218,0.45)" if v >= 0 else "rgba(255,107,107,0.45)"
            for v in ic_s.values
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ic_s.index, y=ic_s.values,
            marker_color=bar_colors,
            name="Daily IC",
        ))
        fig.add_trace(go.Scatter(
            x=rolling_ic.index, y=rolling_ic.values,
            line=dict(color="#64ffda", width=2),
            name="21d Rolling IC",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#8892b0")
        fig.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6", legend=dict(bgcolor="#0d0f14"),
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_ic_hist:
        st.markdown(f"#### IC Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=ic_s.values, nbinsx=40,
            marker_color="#5c6bc0",
            marker_line_color="#64ffda", marker_line_width=0.5,
        ))
        fig2.add_vline(x=ic_s.mean(), line_color="#64ffda",
                       annotation_text=f"μ={ic_s.mean():.4f}", line_dash="dash")
        fig2.add_vline(x=0, line_color="rgba(255,107,107,0.4)", line_dash="dot")
        fig2.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------------
    # Row 2: Factor Decay + Quintile cumulative returns
    # -----------------------------------------------------------------------
    col_decay, col_quint = st.columns([1, 2])

    with col_decay:
        st.markdown("#### 📉 Factor Decay Curve")
        decay_df = pd.DataFrame(
            list(fs.decay.items()), columns=["Horizon (days)", "Mean IC"]
        ).sort_values("Horizon (days)")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=decay_df["Horizon (days)"],
            y=decay_df["Mean IC"],
            mode="lines+markers",
            marker=dict(color="#64ffda", size=8),
            line=dict(color="#64ffda", width=2),
            fill="tozeroy",
            fillcolor="rgba(100,255,218,0.13)",
        ))
        fig3.add_hline(y=0, line_dash="dot", line_color="#ff6b6b")
        fig3.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6",
            xaxis_title="Forward Horizon (days)",
            yaxis_title="Mean Spearman IC",
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_quint:
        st.markdown("#### 📈 Quintile Cumulative Returns")
        colors = ["#ff6b6b", "#ffa07a", "#6b7280", "#64b3f4", "#64ffda"]

        fig4 = go.Figure()
        for q_idx, (q, ret_series) in enumerate(sorted(fs.quintile_returns.items())):
            cum = (1 + ret_series.dropna()).cumprod()
            is_extreme = q in [1, 5]
            fig4.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                name=f"Q{q} ({'Top' if q == 5 else 'Bottom' if q == 1 else ''})".strip(),
                line=dict(color=colors[q_idx], width=2 if is_extreme else 1),
                opacity=1.0 if is_extreme else 0.5,
            ))
        fig4.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6", legend=dict(bgcolor="#0d0f14"),
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"),
            yaxis_title="Cumulative Return",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------------------------------------------------
    # Row 3: Factor correlation matrix  +  Long/Short equity curve
    # -----------------------------------------------------------------------
    col_corr, col_ls = st.columns([1, 2])

    with col_corr:
        st.markdown("#### 🔗 Factor Correlation Matrix")
        corr = report.correlation_matrix

        fig5 = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig5.update_traces(textfont_size=9)
        fig5.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=30, b=0), height=350,
            coloraxis_colorbar=dict(tickfont_color="#ccd6f6"),
            xaxis=dict(tickfont_size=9),
            yaxis=dict(tickfont_size=9),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col_ls:
        st.markdown(f"#### 📊 {selected_factor} — Long/Short Portfolio Equity Curve")

        ls_rets = fs.ls_returns.dropna()
        ls_cum = (1 + ls_rets).cumprod()

        # Rolling Sharpe (63-day)
        roll_sharpe = (
            ls_rets.rolling(63).mean() / ls_rets.rolling(63).std() * np.sqrt(252)
        )

        fig6 = go.Figure()

        # Drawdown fill
        rolling_max = ls_cum.cummax()
        drawdown = (ls_cum - rolling_max) / rolling_max

        fig6.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", fillcolor="rgba(255,107,107,0.13)",
            line=dict(color="rgba(255,107,107,0.2)", width=0.5),
            name="Drawdown", yaxis="y2",
        ))
        fig6.add_trace(go.Scatter(
            x=ls_cum.index, y=ls_cum.values,
            line=dict(color="#64ffda", width=2.5),
            name="L/S Portfolio",
        ))
        fig6.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6",
            legend=dict(bgcolor="#0d0f14"),
            margin=dict(l=0, r=0, t=30, b=0), height=350,
            xaxis=dict(gridcolor="#1e2235"),
            yaxis=dict(gridcolor="#1e2235", title="Cumulative Return"),
            yaxis2=dict(
                overlaying="y", side="right", showgrid=False,
                title="Drawdown", tickformat=".0%",
                tickfont=dict(color="rgba(255,107,107,0.5)"),
            ),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # -----------------------------------------------------------------------
    # Row 4: IC heatmap (month × year)
    # -----------------------------------------------------------------------
    st.markdown("#### 🔥 IC Heatmap — Monthly Performance")

    ic_monthly = fs.ic_series.resample("ME").mean().dropna()
    if not ic_monthly.empty:
        ic_df = ic_monthly.to_frame("IC")
        ic_df["Year"] = ic_df.index.year
        ic_df["Month"] = ic_df.index.strftime("%b")

        MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = ic_df.pivot(index="Year", columns="Month", values="IC").reindex(columns=MONTHS)

        fig7 = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            zmin=-0.1, zmax=0.1,
            text_auto=".3f",
            aspect="auto",
        )
        fig7.update_traces(textfont_size=9)
        fig7.update_layout(
            paper_bgcolor="#13162b", plot_bgcolor="#13162b",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=30, b=0), height=200,
        )
        st.plotly_chart(fig7, use_container_width=True)


# ===========================================================================
# TAB 2 — BACKTEST RESULTS
# ===========================================================================

with tab_backtest:
    st.markdown("## 📈 Backtest Results")
    st.info("Wire your Backtester class output here. Coming in Week 4.")

    # Placeholder with best factor's L/S curve
    best_factor = report.summary_table["L/S Sharpe"].idxmax()
    fs_best = report.factor_stats[best_factor]

    st.markdown(f"**Best factor by Sharpe:** `{best_factor}` — Sharpe {fs_best.ls_sharpe:.3f}")

    ls_cum = (1 + fs_best.ls_returns.dropna()).cumprod()

    # Benchmark (equal weight buy & hold)
    bh = prices.pct_change().mean(axis=1).dropna()
    bh_cum = (1 + bh).cumprod()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bh_cum.index, y=bh_cum.values,
                                line=dict(color="#8892b0", dash="dot"),
                                name="Equal-Weight Buy & Hold"))
    fig_bt.add_trace(go.Scatter(x=ls_cum.index, y=ls_cum.values,
                                line=dict(color="#64ffda", width=2),
                                name=f"{best_factor} L/S"))
    fig_bt.update_layout(
        paper_bgcolor="#13162b", plot_bgcolor="#13162b",
        font_color="#ccd6f6", height=400,
        legend=dict(bgcolor="#0d0f14"),
        xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"),
        yaxis_title="Cumulative Return",
    )
    st.plotly_chart(fig_bt, use_container_width=True)


# ===========================================================================
# TAB 3 — DATA EXPLORER
# ===========================================================================

with tab_data:
    st.markdown("## 🗄️ Data Explorer")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**Universe:** {len(prices.columns)} tickers")
        st.markdown(f"**Date range:** {prices.index[0].date()} -> {prices.index[-1].date()}")
        st.markdown(f"**Trading days:** {len(prices)}")
        st.markdown(f"**Factors loaded:** {len(factors)}")

    with col_right:
        ticker = st.selectbox("Inspect ticker", sorted(prices.columns))

    p = prices[ticker].dropna()
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=p.index, y=p.values,
                               line=dict(color="#64ffda"), fill="tozeroy",
                               fillcolor="rgba(100, 255, 218, 0.07)", name=ticker))
    fig_p.update_layout(
        paper_bgcolor="#13162b", plot_bgcolor="#13162b",
        font_color="#ccd6f6", height=300,
        xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("#### Factor Signals for Selected Ticker")
    factor_df = pd.DataFrame({k: v[ticker] for k, v in factors.items() if ticker in v.columns})
    st.line_chart(factor_df.dropna(), height=200)
