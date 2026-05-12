"""Alpha Vantage price data fetcher.

Free tier: 25 requests/day. Use as fallback when yfinance is rate-limited.
Docs: https://www.alphavantage.co/documentation/
"""

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd

from src.data.config_loader import load_secrets
from src.data.fetcher import DataFetcher
from src.data.cache import DataCache
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)


class AlphaVantageFetcher(DataFetcher):
    """Fetch OHLCV data from Alpha Vantage.

    Uses TIME_SERIES_DAILY_ADJUSTED endpoint.
    Rate limit: 25 calls/day on free tier.
    """

    source_name: str = "alpha_vantage"

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        secrets = load_secrets()
        self._api_key = secrets.get("alpha_vantage", {}).get("api_key", "")
        if not self._api_key:
            logger.warning("Alpha Vantage API key not configured")
        self._cache = cache or DataCache()

    def fetch(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily adjusted OHLCV for one or more tickers.

        Note: each ticker counts as 1 API call. On free tier (25/day),
        batch carefully.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self._fetch_single(ticker)
            frames.append(df)

        result = pd.concat(frames)
        # Filter to requested date range
        s = pd.Timestamp(start, tz=MARKET_TIMEZONE)
        e = pd.Timestamp(end, tz=MARKET_TIMEZONE)
        filtered = result.loc[s:e] if not result.empty else result
        if filtered.empty:
            logger.warning(
                "Requested range %s→%s outside available data; "
                "returning full compact set (%s→%s)",
                start, end,
                result.index.min(), result.index.max(),
            )
            return self.validate(result)
        return self.validate(filtered)

    def _fetch_single(self, ticker: str) -> pd.DataFrame:
        """Fetch a single ticker from Alpha Vantage."""
        import requests

        cache_key = ticker.upper()
        if not self._cache.stale(self.source_name, cache_key):
            cached = self._cache.load(self.source_name, cache_key)
            if cached is not None:
                return cached

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self._api_key,
            "datatype": "json",
        }

        logger.info("Alpha Vantage: fetching %s", ticker)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Check for rate-limit message
        if "Note" in data or "Information" in data:
            from src.utils.validation import DataError
            msg = data.get("Note") or data.get("Information", "unknown error")
            raise DataError(f"[alpha_vantage] API message: {msg}")

        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            from src.utils.validation import DataError
            raise DataError(
                f"[alpha_vantage] unexpected response for {ticker}: "
                f"{list(data.keys())}"
            )

        records: list[dict] = []
        ts = data[ts_key]
        for date_str, values in ts.items():
            c = float(values["4. close"])
            records.append({
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": c,
                "adj_close": c,  # unadjusted — no adj close on free tier
                "volume": int(values["5. volume"]),
                "ticker": ticker.upper(),
            })

        df = pd.DataFrame(records)

        # Parse dates
        df.index = pd.to_datetime(
            [d for d in ts.keys()], format="%Y-%m-%d"
        )
        df.index = df.index.tz_localize(MARKET_TIMEZONE)
        df = df.sort_index()

        self._cache.update(df, self.source_name, cache_key)
        return df
