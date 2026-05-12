"""FRED (Federal Reserve Economic Data) macro data fetcher.

Free, unlimited tier — primary source for CPI, GDP, unemployment, FOMC dates.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.data.config_loader import load_secrets
from src.data.fetcher import DataFetcher
from src.data.cache import DataCache
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)

# Canonical FRED series IDs for key macro indicators
FRED_SERIES: dict[str, str] = {
    "cpi": "CPIAUCSL",            # CPI All Urban Consumers (monthly)
    "core_cpi": "CPILFESL",       # Core CPI (ex food/energy)
    "unemployment": "UNRATE",      # Unemployment rate (monthly)
    "gdp": "GDP",                  # Gross Domestic Product (quarterly)
    "fed_funds": "FEDFUNDS",       # Effective Federal Funds Rate
    "industrial_production": "INDPRO",  # Industrial Production Index
    "retail_sales": "RSAFS",      # Retail Sales
    "ism_pmi": "NAPM",            # ISM Manufacturing PMI
}


class FREDFetcher(DataFetcher):
    """Fetch macro data from FRED API.

    Free tier: 120 requests/minute, key required.
    """

    source_name: str = "fred"

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        secrets = load_secrets()
        self._api_key = secrets.get("fred", {}).get("api_key", "")
        if not self._api_key:
            logger.warning("FRED API key not configured — fetcher will fail")
        self._cache = cache or DataCache()

    def fetch(
        self,
        series: str | List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch one or more FRED series.

        Args:
            series: Series name (e.g. "cpi") or FRED ID (e.g. "CPIAUCSL").
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            DataFrame with DatetimeIndex tz="America/New_York",
            columns = series names.
        """
        if isinstance(series, str):
            series = [series]

        frames: dict[str, pd.Series] = {}
        for s in series:
            fred_id = FRED_SERIES.get(s, s)
            ts = self._fetch_series(fred_id, start, end)
            frames[s] = ts

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(MARKET_TIMEZONE)
        df = df.sort_index()
        return self.validate(df)

    def _fetch_series(
        self, fred_id: str, start: str, end: str
    ) -> pd.Series:
        """Fetch a single FRED series via the API."""
        import requests

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": fred_id,
            "observation_start": start,
            "observation_end": end,
            "api_key": self._api_key,
            "file_type": "json",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("observations", [])
        dates: list[pd.Timestamp] = []
        values: list[float] = []
        for obs in records:
            val = obs.get("value")
            if val is None or val == ".":
                continue
            try:
                dates.append(pd.Timestamp(obs["date"]))
                values.append(float(val))
            except (ValueError, KeyError):
                continue

        if not dates:
            from src.utils.validation import DataError
            raise DataError(
                f"[fred] no data for series {fred_id} ({start} → {end})"
            )

        return pd.Series(values, index=pd.DatetimeIndex(dates), name=fred_id)

    def fetch_macro_calendar(
        self, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch a calendar of major macro releases as event markers."""
        series_ids = list(FRED_SERIES.values())
        df = self.fetch(series_ids, start, end)

        # Mark release dates (non-NaN rows) as event dates
        events: list[dict] = []
        for col in df.columns:
            non_null = df[col].dropna()
            for dt in non_null.index:
                events.append({
                    "date": dt,
                    "series": col,
                    "value": non_null[dt],
                    "event_type": "macro_release",
                })

        result = pd.DataFrame(events)
        if result.empty:
            return result
        result = result.sort_values("date").reset_index(drop=True)
        return result
