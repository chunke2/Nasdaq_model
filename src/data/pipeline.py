"""Unified data pipeline with fallback chain and batch fetching.

Priority: Polygon > Alpha Vantage > synthetic (offline fallback).
Reads tickers and parameters from config/settings.yaml.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.data.cache import DataCache
from src.data.config_loader import load_settings
from src.data.fetcher import DataFetcher
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)


class DataPipeline:
    """Unified data fetching with automatic fallback.

    Usage:
        pipe = DataPipeline()
        df = pipe.fetch_prices(["AAPL", "NVDA"], "2023-01-01", "2025-12-31")
        # or batch all config tickers:
        df = pipe.fetch_default_universe()
    """

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        self._cache = cache or DataCache()
        self._fetchers: dict[str, DataFetcher] = {}
        self._init_fetchers()

    def _init_fetchers(self) -> None:
        """Lazy-init fetchers in priority order."""
        # P0: Polygon (most capable)
        try:
            from src.data.polygon import PolygonFetcher
            self._fetchers["polygon"] = PolygonFetcher(cache=self._cache)
        except Exception as e:
            logger.debug("Polygon unavailable: %s", e)

        # P1: Alpha Vantage (limited but reliable)
        try:
            from src.data.alpha_vantage import AlphaVantageFetcher
            self._fetchers["alpha_vantage"] = AlphaVantageFetcher(
                cache=self._cache
            )
        except Exception as e:
            logger.debug("Alpha Vantage unavailable: %s", e)

        # P2: Synthetic (always available, offline)
        try:
            from src.data.synthetic import SyntheticPriceFetcher
            self._fetchers["synthetic"] = SyntheticPriceFetcher()
        except Exception as e:
            logger.debug("Synthetic unavailable: %s", e)

    @property
    def available_sources(self) -> list[str]:
        return list(self._fetchers.keys())

    def fetch_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch price data for tickers, trying fetchers in priority order.

        Each ticker is fetched from the highest-priority source that has
        it cached; new tickers are fetched from the top available source.

        Returns:
            DataFrame with DatetimeIndex tz=America/New_York,
            columns: [ticker, open, high, low, close, adj_close, volume].
        """
        s = pd.Timestamp(start, tz=MARKET_TIMEZONE)
        e = pd.Timestamp(end, tz=MARKET_TIMEZONE)

        frames: list[pd.DataFrame] = []
        missing: list[str] = []

        for ticker in tickers:
            cached = self._load_cached(ticker, s, e)
            if cached is not None and not force_refresh:
                frames.append(cached)
            else:
                missing.append(ticker)

        if missing:
            new_data = self._fetch_missing(missing, start, end)
            for ticker in missing:
                if ticker in new_data:
                    frames.append(new_data[ticker])

        if not frames:
            from src.utils.validation import DataError
            raise DataError("No data fetched for any ticker")

        result = pd.concat(frames)
        result = result.sort_index()
        return result

    def fetch_default_universe(self) -> pd.DataFrame:
        """Fetch prices for all tickers in config/settings.yaml."""
        settings = load_settings()
        tickers = (
            settings.get("data", {})
            .get("tickers", {})
            .get("nasdaq_sample", [])
        )
        start = (
            settings.get("data", {}).get("default_start", "2020-01-01")
        )
        end = settings.get("data", {}).get("default_end", "2025-12-31")

        if not tickers:
            raise ValueError("No tickers configured in settings.yaml")

        logger.info("Fetching default universe: %d tickers (%s → %s)",
                     len(tickers), start, end)
        return self.fetch_prices(tickers, start, end)

    def _load_cached(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Try to satisfy request from cache. Returns data covering full
        [start, end] range, or None if cache insufficient."""
        for source_name in self._fetchers:
            cached = self._cache.load(source_name, ticker.upper())
            if cached is not None and not cached.empty:
                c_min, c_max = cached.index.min(), cached.index.max()
                if c_min <= start and c_max >= end:
                    return cached.loc[start:end]
                # Partial hit — merge with what we fetch
                if c_min <= end and c_max >= start:
                    # Use what we have, mark what's needed
                    pass
        return None

    def _fetch_missing(
        self, tickers: list[str], start: str, end: str
    ) -> dict[str, pd.DataFrame]:
        """Fetch missing tickers from the best available source."""
        result: dict[str, pd.DataFrame] = {}

        for source_name, fetcher in self._fetchers.items():
            remaining = [t for t in tickers if t not in result]
            if not remaining:
                break

            logger.info("Trying %s for %d tickers", source_name, len(remaining))
            try:
                if source_name == "synthetic":
                    # Synthetic is batch-capable
                    df = fetcher.fetch(remaining, start, end)
                    for ticker, group in df.groupby("ticker"):
                        if ticker.upper() not in result:
                            result[ticker.upper()] = group
                            logger.info(
                                "  %s: %s fetched via synthetic (%d rows)",
                                ticker, ticker, len(group),
                            )
                else:
                    for ticker in remaining:
                        try:
                            df = fetcher.fetch(ticker, start, end)
                            result[ticker.upper()] = df
                            logger.info(
                                "  %s: fetched via %s (%d rows)",
                                ticker, source_name, len(df),
                            )
                        except Exception as exc:
                            logger.warning(
                                "  %s: %s failed (%s), trying next",
                                ticker, source_name, exc,
                            )
            except Exception as exc:
                logger.warning("%s batch failed: %s", source_name, exc)

        return result

    def refresh_cache(self, tickers: Optional[List[str]] = None) -> int:
        """Force-refresh all cached data. Returns count of tickers refreshed."""
        if tickers is None:
            settings = load_settings()
            tickers = (
                settings.get("data", {})
                .get("tickers", {})
                .get("nasdaq_sample", [])
            )

        start = "2020-01-01"
        end = pd.Timestamp.now(tz=MARKET_TIMEZONE).strftime("%Y-%m-%d")

        count = 0
        for ticker in tickers:
            try:
                self.fetch_prices([ticker], start, end, force_refresh=True)
                count += 1
            except Exception as exc:
                logger.warning("Refresh failed for %s: %s", ticker, exc)

        logger.info("Refreshed %d/%d tickers", count, len(tickers))
        return count
