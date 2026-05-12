"""Price/volume data fetcher via yfinance."""

from __future__ import annotations

import time
from typing import List, Optional, Union

import pandas as pd
import yfinance as yf

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)

PRICE_COLUMNS: list[str] = [
    "open", "high", "low", "close", "adj_close", "volume"
]

META_COLUMNS: list[str] = [
    "ticker", "sector", "industry", "market_cap"
]

_RETRY_DELAYS: list[float] = [2.0, 5.0, 15.0]  # seconds, exponential-ish



class PriceFetcher(DataFetcher):
    """Fetch OHLCV data from Yahoo Finance with local caching."""

    source_name: str = "yfinance_price"

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        self._cache = cache or DataCache()

    def fetch(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch price data for one or more tickers.

        Args:
            tickers: Single ticker (str) or list of tickers.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: If True and cache is fresh, return cached data.
            force_refresh: If True, skip cache entirely.

        Returns:
            DataFrame with DatetimeIndex tz="America/New_York",
            columns: [ticker, open, high, low, close, adj_close, volume].
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self._fetch_single(
                ticker, start, end, use_cache=use_cache, force_refresh=force_refresh
            )
            frames.append(df)

        result = pd.concat(frames)
        return self.validate(result)

    def _fetch_single(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        use_cache: bool,
        force_refresh: bool,
    ) -> pd.DataFrame:
        cache_key = ticker.upper()

        if use_cache and not force_refresh and not self._cache.stale(self.source_name, cache_key):
            cached = self._cache.load(self.source_name, cache_key)
            if cached is not None:
                return self._slice_cached(cached, start, end, ticker)

        logger.info("Fetching %s from Yahoo Finance (%s → %s)", ticker, start, end)
        raw = self._download_with_retry(ticker, start, end)

        if raw.empty:
            from src.utils.validation import DataError

            raise DataError(f"[{self.source_name}] no data returned for {ticker}")

        df = self._normalize(raw, ticker)
        self._cache.update(df, self.source_name, cache_key)
        return df

    def _download_with_retry(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Download with retry on rate-limit errors.

        Tries yf.download first, falls back to Ticker.history(),
        with exponential backoff between retries.
        """
        last_err: Exception | None = None
        for i, delay in enumerate([0.0, 5.0, 15.0, 30.0]):
            if i > 0:
                logger.warning(
                    "Retry %d for %s in %.0fs",
                    i, ticker, delay,
                )
                time.sleep(delay)
            try:
                # Try download first, fall back to ticker-based fetch
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                )
                if raw.empty:
                    logger.info(
                        "download empty, trying Ticker.history for %s", ticker
                    )
                    raw = yf.Ticker(ticker).history(
                        start=start, end=end, auto_adjust=False
                    )
                if not raw.empty:
                    return raw
            except Exception as exc:
                last_err = exc
        raise last_err or RuntimeError(
            f"Failed to download {ticker} after retries"
        )

    def _normalize(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Convert yfinance output to standardized format."""
        df = raw.copy()

        # Handle multi-level columns (yfinance >= 0.2.0 with multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            cols = df.columns.get_level_values(0)
            if ticker in cols:
                df = df[ticker]
            else:
                df = df.iloc[:, 0]

        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)

        # Keep only standard columns
        for col in PRICE_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[PRICE_COLUMNS]

        # Add ticker column
        df["ticker"] = ticker.upper()

        # Timezone: yfinance returns naive timestamps in exchange local time (ET)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(
                MARKET_TIMEZONE, ambiguous=True
            )

        # Drop rows where index didn't parse
        df = df[df.index.notna()]
        return df

    def _slice_cached(
        self, cached: pd.DataFrame, start: str, end: str, ticker: str
    ) -> pd.DataFrame:
        """Return cache slice; refetch if range extends beyond cache."""
        s = pd.Timestamp(start, tz=MARKET_TIMEZONE)
        e = pd.Timestamp(end, tz=MARKET_TIMEZONE)

        if cached.index.min() <= s and cached.index.max() >= e:
            return cached.loc[s:e]

        # Need more data — refetch full range
        return self._fetch_single(
            ticker, start, end, use_cache=False, force_refresh=True
        )

    def fetch_meta(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch ticker metadata (sector, industry, market cap)."""
        records: list[dict] = []
        for t in tickers:
            info = yf.Ticker(t).info
            records.append({
                "ticker": t.upper(),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
            })

        df = pd.DataFrame(records)
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        return df
