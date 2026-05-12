"""Synthetic price data generator for offline development and testing.

Generates realistic OHLCV data with known factor effects, enabling
end-to-end model pipeline testing without external API dependencies.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.utils.logging_config import get_logger
from src.utils.seed import get_rng
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)

PRICE_COLUMNS: list[str] = [
    "open", "high", "low", "close", "adj_close", "volume"
]


class SyntheticPriceFetcher(DataFetcher):
    """Generate synthetic price data with controllable factor effects.

    Prices follow geometric Brownian motion with an added AR(1) drift
    term for known predictability. Event shocks can be injected at
    specified dates with known magnitudes.
    """

    source_name: str = "synthetic_price"

    def __init__(
        self,
        base_price: float = 100.0,
        annual_vol: float = 0.25,
        annual_drift: float = 0.07,
        tick_size: float = 0.01,
    ) -> None:
        self.base_price = base_price
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.tick_size = tick_size
        self._rng = get_rng()

    def fetch(
        self,
        tickers: str | List[str],
        start: str,
        end: str,
        *,
        events: Optional[dict[str, pd.Series]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic price paths.

        Args:
            tickers: Single ticker or list of tickers.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            events: Optional dict of ticker -> pd.Series of event shocks
                    (index=date, value=log-return shock on that day).
            seed: Override RNG seed for reproducibility.

        Returns:
            DataFrame with DatetimeIndex tz="America/New_York",
            columns: [ticker, open, high, low, close, adj_close, volume].
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self._rng

        if isinstance(tickers, str):
            tickers = [tickers]

        dates = pd.date_range(
            start=start, end=end, freq="B", tz=MARKET_TIMEZONE
        )
        n = len(dates)
        if n == 0:
            from src.utils.validation import DataError
            raise DataError("No business days in date range")

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            ticker_upper = ticker.upper()
            shocks = np.zeros(n)
            if events and ticker_upper in events:
                evt = events[ticker_upper]
                for d, v in evt.items():
                    d_ts = pd.Timestamp(d, tz=MARKET_TIMEZONE)
                    mask = dates == d_ts
                    if mask.any():
                        shocks[mask] = v

            df = self._generate_path(
                ticker_upper, dates, shocks, rng
            )
            frames.append(df)

        result = pd.concat(frames)
        return self.validate(result)

    def _generate_path(
        self,
        ticker: str,
        dates: pd.DatetimeIndex,
        shocks: np.ndarray,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate one price path with GBM + shock injections."""
        n = len(dates)
        dt = 1 / 252
        drift = self.annual_drift * dt
        vol = self.annual_vol * np.sqrt(dt)

        # Daily log returns with known AR(1) structure → momentum factor
        raw_ret = rng.normal(drift, vol, size=n)
        # Inject shock on event day, spread reversal over t+2:t+4
        # (post-earnings drift decays over several days)
        ret = np.zeros(n)
        for i in range(n):
            ret[i] = raw_ret[i] + shocks[i]
            # Multi-day reversal: signal persists for factor to capture
            # factor at t+1 (= shock_t) can predict returns at t+2, t+3, t+4
            if i >= 3:
                ret[i] -= 0.25 * shocks[i - 2]   # t+2: reversal peak
                ret[i] -= 0.15 * shocks[i - 3]   # t+3: continuation
                ret[i] -= 0.07 * shocks[i - 4]   # t+4: residual

        # Price path
        price = self.base_price * np.exp(np.cumsum(ret))
        # Round to tick size
        price = np.round(price / self.tick_size) * self.tick_size

        # OHLC from close + intraday noise
        intraday_range = np.abs(ret) * 0.5 * price
        close = price
        open_ = close - ret * price + rng.normal(0, 0.002, n) * price
        high = np.maximum(close, open_) + np.abs(intraday_range) * rng.uniform(0.1, 1.0, n)
        low = np.minimum(close, open_) - np.abs(intraday_range) * rng.uniform(0.1, 1.0, n)
        adj_close = close
        # Base volume, elevated on event days (5x spike = realistic earnings surge)
        base_vol = rng.integers(10_000_000, 100_000_000, n)
        vol_multiplier = np.ones(n)
        vol_multiplier[shocks != 0] = 5.0  # volume spike on event days
        volume = (base_vol * vol_multiplier).astype(int)

        df = pd.DataFrame(
            {
                "open": np.round(open_, 2),
                "high": np.round(high, 2),
                "low": np.round(low, 2),
                "close": np.round(close, 2),
                "adj_close": np.round(adj_close, 2),
                "volume": volume,
                "ticker": ticker,
            },
            index=dates,
        )
        return df
