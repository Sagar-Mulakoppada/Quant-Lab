"""
market_data.py
==============
Data ingestion pipeline — pulls price + volume data and stores in DuckDB.
Supports incremental refresh (only fetches missing dates).

Usage
-----
    from src.ingest.market_data import MarketDataPipeline
    pipeline = MarketDataPipeline(db_path="data/alpha.db")
    pipeline.refresh(tickers=SP500_SUBSET, start="2019-01-01")
    prices = pipeline.load_prices()
    volumes = pipeline.load_volumes()
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# S&P 500 representative subset (free, no API key needed)
# Extend this list or pull dynamically via Wikipedia
# ---------------------------------------------------------------------------

SP500_SUBSET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "KO", "LLY", "AVGO", "TMO", "MCD", "COST", "CSCO", "ABT", "DHR",
    "ACN", "BAC", "WMT", "NEE", "RTX", "UNH", "QCOM", "LIN", "IBM",
    "AMGN", "SBUX", "GS", "CAT", "DE", "MMM", "SCHW", "AXP", "BKNG",
    "TGT", "MDLZ", "CVS", "CI", "SO",
]


class MarketDataPipeline:
    """
    Manages OHLCV data in a local DuckDB database.
    """

    TABLE_PRICES = "prices"
    TABLE_VOLUMES = "volumes"

    def __init__(self, db_path: str = "data/alpha.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(
        self,
        tickers: Optional[List[str]] = None,
        start: str = "2019-01-01",
        end: Optional[str] = None,
        batch_size: int = 25,
        sleep_secs: float = 1.0,
    ) -> None:
        """
        Fetch missing data for all tickers and upsert into DuckDB.
        Batches requests to avoid rate limits.
        """
        tickers = tickers or SP500_SUBSET
        end = end or datetime.date.today().isoformat()

        existing_end = self._latest_date()
        fetch_start = existing_end or start

        if existing_end and existing_end >= end:
            print(f"[DataPipeline] Data is up to date (latest: {existing_end})")
            return

        print(f"[DataPipeline] Fetching {len(tickers)} tickers from {fetch_start} to {end} ...")

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i: i + batch_size]
            print(f"  batch {i // batch_size + 1}: {batch[:3]} ... ({len(batch)} tickers)")
            self._fetch_batch(batch, fetch_start, end)
            time.sleep(sleep_secs)

        print(f"[DataPipeline] Refresh complete. Latest date: {self._latest_date()}")

    def load_prices(self) -> pd.DataFrame:
        """Return wide DataFrame: rows=dates, cols=tickers (adjusted close)."""
        return self._load_wide(self.TABLE_PRICES)

    def load_volumes(self) -> pd.DataFrame:
        """Return wide DataFrame: rows=dates, cols=tickers (volume)."""
        return self._load_wide(self.TABLE_VOLUMES)

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self.con.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_PRICES} (
                date    DATE    NOT NULL,
                ticker  VARCHAR NOT NULL,
                close   DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        self.con.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_VOLUMES} (
                date    DATE    NOT NULL,
                ticker  VARCHAR NOT NULL,
                volume  DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)

    def _fetch_batch(
        self, tickers: List[str], start: str, end: str
    ) -> None:
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"    [WARN] yfinance error: {e}")
            return

        if raw.empty:
            return

        # Handle single vs multi-ticker response
        if isinstance(raw.columns, pd.MultiIndex):
            close_wide = raw["Close"] if "Close" in raw else raw["Adj Close"]
            vol_wide = raw["Volume"]
        else:
            # Single ticker — yfinance returns flat columns
            ticker = tickers[0]
            col = "Close" if "Close" in raw.columns else "Adj Close"
            close_wide = raw[[col]].rename(columns={col: ticker})
            vol_wide = raw[["Volume"]].rename(columns={"Volume": ticker})

        self._upsert_wide(close_wide, self.TABLE_PRICES, "close")
        self._upsert_wide(vol_wide, self.TABLE_VOLUMES, "volume")

    def _upsert_wide(
        self, wide: pd.DataFrame, table: str, value_col: str
    ) -> None:
        """Melt wide DataFrame and upsert into DuckDB table."""
        long = wide.reset_index().melt(
            id_vars=["Date"], var_name="ticker", value_name=value_col
        )
        long = long.rename(columns={"Date": "date"})
        long = long.dropna(subset=[value_col])
        long["date"] = pd.to_datetime(long["date"]).dt.date

        # Register as an in-memory view for DuckDB SQL
        self.con.register("_tmp_upsert", long)
        self.con.execute(f"""
            INSERT OR REPLACE INTO {table}
            SELECT date, ticker, {value_col} FROM _tmp_upsert
        """)
        self.con.unregister("_tmp_upsert")

    def _load_wide(self, table: str) -> pd.DataFrame:
        val_col = "close" if table == self.TABLE_PRICES else "volume"
        long = self.con.execute(
            f"SELECT date, ticker, {val_col} FROM {table} ORDER BY date"
        ).df()
        wide = long.pivot(index="date", columns="ticker", values=val_col)
        wide.index = pd.to_datetime(wide.index)
        wide.sort_index(inplace=True)
        return wide

    def _latest_date(self) -> Optional[str]:
        try:
            result = self.con.execute(
                f"SELECT MAX(date) FROM {self.TABLE_PRICES}"
            ).fetchone()
            if result and result[0]:
                return str(result[0])
        except Exception:
            pass
        return None
