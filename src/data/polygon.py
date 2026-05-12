"""Polygon.io price data fetcher.

Free tier: 5 requests/minute, unlimited historical daily bars.
Docs: https://polygon.io/docs/stocks/
"""

from __future__ import annotations

import time
from typing import List, Optional, Union

import pandas as pd

from src.data.config_loader import load_secrets
from src.data.fetcher import DataFetcher
from src.data.cache import DataCache
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)


class PolygonFetcher(DataFetcher):
    """Fetch OHLCV from Polygon.io.

    Uses /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}
    Free tier: 5 calls/min, full historical access.
    """

    source_name: str = "polygon"

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        secrets = load_secrets()
        self._api_key = secrets.get("polygon", {}).get("api_key", "")
        if not self._api_key:
            logger.warning("Polygon.io API key not configured")
        self._cache = cache or DataCache()
        self._last_call: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce 5 calls/min (12s between calls)."""
        elapsed = time.time() - self._last_call
        if elapsed < 12.0 and self._last_call > 0:
            wait = 12.0 - elapsed
            logger.debug("Polygon rate limit: waiting %.1fs", wait)
            time.sleep(wait)
        self._last_call = time.time()

    def fetch(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for one or more tickers."""
        if isinstance(tickers, str):
            tickers = [tickers]

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self._fetch_single(ticker, start, end)
            frames.append(df)

        result = pd.concat(frames)
        return self.validate(result)

    def _fetch_single(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch one ticker from Polygon, using cache if available."""
        cache_key = ticker.upper()
        if not self._cache.stale(self.source_name, cache_key):
            cached = self._cache.load(self.source_name, cache_key)
            if cached is not None:
                s = pd.Timestamp(start, tz=MARKET_TIMEZONE)
                e = pd.Timestamp(end, tz=MARKET_TIMEZONE)
                if cached.index.min() <= s and cached.index.max() >= e:
                    return cached.loc[s:e]
                # Cache exists but doesn't cover range — force full refetch
                logger.info("Cache miss for range, force-fetching %s", ticker)

        return self._force_fetch(ticker, start, end, cache_key)

    def _force_fetch(
        self, ticker: str, start: str, end: str, cache_key: str
    ) -> pd.DataFrame:
        """Fetch from API directly (no cache check)."""
        self._rate_limit()

        import requests

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/"
            f"range/1/day/{start}/{end}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self._api_key,
        }

        logger.info("Polygon: fetching %s (%s → %s)", ticker, start, end)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "ERROR":
            from src.utils.validation import DataError
            raise DataError(
                f"[polygon] API error for {ticker}: {data.get('error')}"
            )

        results = data.get("results", [])
        if not results:
            from src.utils.validation import DataError
            raise DataError(
                f"[polygon] no results for {ticker} ({start} → {end})"
            )

        records = [
            {
                "open": float(r["o"]),
                "high": float(r["h"]),
                "low": float(r["l"]),
                "close": float(r["c"]),
                "adj_close": float(r["c"]),
                "volume": int(r.get("v", 0)),
                "ticker": ticker.upper(),
            }
            for r in results
        ]

        df = pd.DataFrame(records)
        df.index = pd.to_datetime(
            [r["t"] for r in results], unit="ms", utc=True
        )
        df.index = df.index.tz_convert(MARKET_TIMEZONE)
        df = df.sort_index()

        self._cache.update(df, self.source_name, cache_key)
        return df


    def fetch_earnings_calendar(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch historical earnings dates for a ticker.

        Free tier: limited access. May return empty if not available.
        """
        self._rate_limit()

        import requests

        url = (
            f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        )
        params = {"apiKey": self._api_key}

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", {})
        return pd.DataFrame([results]) if results else pd.DataFrame()
