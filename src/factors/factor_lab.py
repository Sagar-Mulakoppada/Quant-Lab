"""
factor_lab.py
=============
Factor Research Lab — the analytical core of AlphaSignal Engine.

For every registered factor this module computes:
  - Information Coefficient (IC) and ICIR across multiple forward horizons
  - Factor Sharpe Ratio (long Q5 / short Q1 quintile spread)
  - Portfolio Turnover per rebalance
  - Factor Decay curve  (IC vs. horizon: 1, 5, 10, 20, 60 days)
  - Factor Correlation matrix (rank correlation between factor signals)

All analysis is purely cross-sectional and avoids lookahead bias by
aligning factor values at time t with returns starting at t+1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FactorStats:
    """Full analytics profile for a single factor."""
    name: str

    # IC metrics
    ic_series: pd.Series = field(default_factory=pd.Series)   # daily IC
    mean_ic: float = 0.0
    icir: float = 0.0                                           # IC / std(IC)
    ic_positive_pct: float = 0.0                               # % of days IC > 0

    # Decay profile  {horizon_days: mean_ic}
    decay: Dict[int, float] = field(default_factory=dict)

    # Long/short quintile portfolio metrics
    ls_sharpe: float = 0.0
    ls_annual_return: float = 0.0
    ls_max_drawdown: float = 0.0
    ls_returns: pd.Series = field(default_factory=pd.Series)

    # Turnover
    avg_turnover: float = 0.0                                  # fraction 0–1

    # Quintile cumulative returns {1..5: pd.Series}
    quintile_returns: Dict[int, pd.Series] = field(default_factory=dict)


@dataclass
class LabReport:
    """Aggregated output across all factors."""
    factor_stats: Dict[str, FactorStats] = field(default_factory=dict)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class FactorLab:
    """
    Factor Research Lab.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices. Shape (T, N) — rows=dates, cols=tickers.
    factors : Dict[str, pd.DataFrame]
        Each value is a factor signal with the same shape as `prices`.
        Values at date t will be aligned with returns from t+1.
    decay_horizons : list of int
        Forward return horizons (in trading days) to compute IC decay.
    n_quintiles : int
        Number of portfolio buckets (default 5).
    rebal_freq : str
        Pandas offset alias for rebalancing ('W'=weekly, 'M'=monthly).
    tcost_bps : float
        One-way transaction cost in basis points.
    """

    DECAY_HORIZONS: List[int] = [1, 5, 10, 20, 60]

    def __init__(
        self,
        prices: pd.DataFrame,
        factors: Dict[str, pd.DataFrame],
        decay_horizons: Optional[List[int]] = None,
        n_quintiles: int = 5,
        rebal_freq: str = "M",
        tcost_bps: float = 10.0,
    ):
        self.prices = prices.copy()
        self.factors = {k: v.copy() for k, v in factors.items()}
        self.decay_horizons = decay_horizons or self.DECAY_HORIZONS
        self.n_quintiles = n_quintiles
        self.rebal_freq = "ME" if rebal_freq == "M" else rebal_freq
        self.tcost_bps = tcost_bps / 10_000

        # Pre-compute forward returns for every horizon we need
        # Always include horizon=1 for the primary IC series
        self._fwd_returns: Dict[int, pd.DataFrame] = {}
        all_horizons = sorted(set(self.decay_horizons + [1, 5, 10, 20, 60]))
        for h in all_horizons:
            self._fwd_returns[h] = self._compute_forward_returns(h)

        # Rebalance dates
        self._rebal_dates = self._get_rebal_dates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> LabReport:
        """Run full analysis across all registered factors."""
        print(f"[FactorLab] Analysing {len(self.factors)} factors ...")

        stats_map: Dict[str, FactorStats] = {}
        for name, signal in self.factors.items():
            print(f"  -> {name}")
            stats_map[name] = self._analyse_factor(name, signal)

        corr_matrix = self._factor_correlation_matrix()
        summary = self._build_summary_table(stats_map)

        report = LabReport(
            factor_stats=stats_map,
            correlation_matrix=corr_matrix,
            summary_table=summary,
        )
        print("[FactorLab] Done.")
        return report

    # ------------------------------------------------------------------
    # Per-factor analysis
    # ------------------------------------------------------------------

    def _analyse_factor(self, name: str, signal: pd.DataFrame) -> FactorStats:
        fs = FactorStats(name=name)

        # 1. IC series (horizon=1 day)
        fs.ic_series = self._compute_ic_series(signal, horizon=1)
        fs.mean_ic = fs.ic_series.mean()
        ic_std = fs.ic_series.std()
        fs.icir = fs.mean_ic / ic_std * np.sqrt(252) if ic_std > 0 else 0.0
        fs.ic_positive_pct = (fs.ic_series > 0).mean()

        # 2. Decay curve
        fs.decay = {
            h: self._compute_ic_series(signal, horizon=h).mean()
            for h in self.decay_horizons
        }

        # 3. Quintile portfolio analysis
        q_rets = self._quintile_returns(signal)
        fs.quintile_returns = q_rets

        ls = q_rets[self.n_quintiles] - q_rets[1]   # top minus bottom
        ls = ls.dropna()
        fs.ls_returns = ls

        daily_mean = ls.mean()
        daily_std = ls.std()
        fs.ls_sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
        fs.ls_annual_return = (1 + daily_mean) ** 252 - 1
        fs.ls_max_drawdown = self._max_drawdown(ls)

        # 4. Turnover
        fs.avg_turnover = self._compute_turnover(signal)

        return fs

    # ------------------------------------------------------------------
    # IC computation
    # ------------------------------------------------------------------

    def _compute_ic_series(
        self, signal: pd.DataFrame, horizon: int = 1
    ) -> pd.Series:
        """
        Compute daily cross-sectional Spearman rank IC between
        signal(t) and forward_return(t, horizon).
        """
        fwd = self._fwd_returns[horizon]

        common_dates = signal.index.intersection(fwd.index)
        sig = signal.loc[common_dates]
        ret = fwd.loc[common_dates]

        ic_values = []
        for date in common_dates:
            s = sig.loc[date].dropna()
            r = ret.loc[date].dropna()
            both = s.index.intersection(r.index)
            if len(both) < 10:
                ic_values.append(np.nan)
                continue
            rho, _ = stats.spearmanr(s[both], r[both])
            ic_values.append(rho)

        return pd.Series(ic_values, index=common_dates, name=f"IC_{horizon}d")

    # ------------------------------------------------------------------
    # Quintile portfolio returns
    # ------------------------------------------------------------------

    def _quintile_returns(
        self, signal: pd.DataFrame
    ) -> Dict[int, pd.Series]:
        """
        At each rebalance date, rank stocks into N quintiles by signal.
        Return the equal-weight daily return of each quintile bucket.
        """
        daily_returns = self.prices.pct_change()
        quintile_cumrets: Dict[int, List] = {q: [] for q in range(1, self.n_quintiles + 1)}
        index_dates: List = []

        for i, start in enumerate(self._rebal_dates[:-1]):
            end = self._rebal_dates[i + 1]

            if start not in signal.index:
                continue
            sig_t = signal.loc[start].dropna()
            if len(sig_t) < self.n_quintiles * 2:
                continue

            # Rank into quintiles
            labels = pd.qcut(sig_t.rank(method="first"), self.n_quintiles,
                             labels=False) + 1  # 1..N

            period_dates = daily_returns.loc[start:end].index[1:]  # exclude rebal day
            if len(period_dates) == 0:
                continue

            for q in range(1, self.n_quintiles + 1):
                tickers = labels[labels == q].index
                tickers = [t for t in tickers if t in daily_returns.columns]
                if not tickers:
                    quintile_cumrets[q].extend([np.nan] * len(period_dates))
                else:
                    period = daily_returns.loc[period_dates, tickers]
                    eq_weight = period.mean(axis=1)
                    # Apply transaction cost on entry (one-way)
                    if len(eq_weight) > 0:
                        eq_weight.iloc[0] -= self.tcost_bps
                    quintile_cumrets[q].extend(eq_weight.tolist())

            index_dates.extend(period_dates.tolist())

        # Build series
        result = {}
        for q in range(1, self.n_quintiles + 1):
            length = min(len(index_dates), len(quintile_cumrets[q]))
            result[q] = pd.Series(
                quintile_cumrets[q][:length],
                index=index_dates[:length],
            ).fillna(0)

        return result

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    def _compute_turnover(self, signal: pd.DataFrame) -> float:
        """
        Average fraction of the long portfolio (top quintile) that
        changes between consecutive rebalance dates.
        """
        turnovers = []
        prev_long: set = set()

        for date in self._rebal_dates:
            if date not in signal.index:
                continue
            sig_t = signal.loc[date].dropna()
            if len(sig_t) < self.n_quintiles * 2:
                continue
            n_long = max(1, len(sig_t) // self.n_quintiles)
            curr_long = set(sig_t.nlargest(n_long).index)

            if prev_long:
                exits = prev_long - curr_long
                turnover = len(exits) / len(prev_long)
                turnovers.append(turnover)

            prev_long = curr_long

        return float(np.mean(turnovers)) if turnovers else 0.0

    # ------------------------------------------------------------------
    # Factor correlation matrix
    # ------------------------------------------------------------------

    def _factor_correlation_matrix(self) -> pd.DataFrame:
        """
        Cross-sectional rank correlation between factor signals.
        For each date, compute pairwise Spearman rho, then average.
        """
        names = list(self.factors.keys())
        N = len(names)
        corr_accum = np.zeros((N, N))
        count = 0

        common_dates = self.factors[names[0]].index
        for n in names[1:]:
            common_dates = common_dates.intersection(self.factors[n].index)

        for date in common_dates:
            ranked = {}
            for name in names:
                row = self.factors[name].loc[date].dropna()
                if len(row) < 10:
                    break
                ranked[name] = row.rank()
            else:
                # All factors valid on this date
                all_tickers = list(
                    set.intersection(*[set(v.index) for v in ranked.values()])
                )
                if len(all_tickers) < 10:
                    continue

                mat = np.array([[ranked[n][all_tickers].values for n in names]])
                mat = mat[0]  # shape (N, n_tickers)
                rho = np.corrcoef(mat)
                corr_accum += rho
                count += 1

        if count == 0:
            return pd.DataFrame(np.eye(N), index=names, columns=names)

        return pd.DataFrame(
            corr_accum / count, index=names, columns=names
        ).round(3)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def _build_summary_table(
        self, stats_map: Dict[str, FactorStats]
    ) -> pd.DataFrame:
        rows = []
        for name, fs in stats_map.items():
            rows.append({
                "Factor": name,
                "Mean IC": round(fs.mean_ic, 4),
                "ICIR": round(fs.icir, 3),
                "IC Positive %": round(fs.ic_positive_pct * 100, 1),
                "L/S Sharpe": round(fs.ls_sharpe, 3),
                "L/S Ann. Return %": round(fs.ls_annual_return * 100, 2),
                "Max Drawdown %": round(fs.ls_max_drawdown * 100, 2),
                "Avg Turnover %": round(fs.avg_turnover * 100, 1),
                "IC@1d": round(fs.decay.get(1, np.nan), 4),
                "IC@5d": round(fs.decay.get(5, np.nan), 4),
                "IC@20d": round(fs.decay.get(20, np.nan), 4),
                "IC@60d": round(fs.decay.get(60, np.nan), 4),
            })
        df = pd.DataFrame(rows).set_index("Factor")
        return df.sort_values("ICIR", ascending=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_forward_returns(self, horizon: int) -> pd.DataFrame:
        """Price return from t+1 to t+horizon (avoids lookahead)."""
        prices = self.prices
        fwd = prices.shift(-horizon) / prices.shift(-1) - 1
        return fwd

    def _get_rebal_dates(self) -> List[pd.Timestamp]:
        """Rebalance schedule aligned to business-month-end (or week-end)."""
        idx = self.prices.resample(self.rebal_freq).last().index
        return [d for d in idx if d in self.prices.index]

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        cumret = (1 + returns).cumprod()
        rolling_max = cumret.cummax()
        drawdown = (cumret - rolling_max) / rolling_max
        return float(drawdown.min())
