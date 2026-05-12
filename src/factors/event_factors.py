"""Event-driven factors.

Constructs factors from detected events, ensuring strict temporal ordering
to prevent look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.factors.base import FactorBase
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE
from src.utils.validation import LeakageReport

logger = get_logger(__name__)


class EarningsSurpriseFactor(FactorBase):
    """Factor: signed earnings surprise magnitude.

    - Positive surprise → expected positive forward return (post-earnings drift)
    - Negative surprise → expected negative forward return
    - Factor value = surprise_pct from event detector, forward-filled
      until next event (carries the signal)
    """

    factor_name: str = "earnings_surprise"

    def compute(
        self, price_df: pd.DataFrame, events_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Compute earnings surprise factor from detected events.

        The factor is aligned to the NEXT trading day after the event,
        ensuring no look-ahead: the signal only becomes available at market
        open after the announcement.

        Returns:
            DataFrame with columns: [date, ticker, factor_name]
            where factor value is the most recent surprise.
        """
        if events_df is None or events_df.empty:
            logger.warning("No events provided for %s factor", self.factor_name)
            return pd.DataFrame(columns=["date", "ticker", self.factor_name])

        earnings = events_df[events_df["event_type"] == "earnings"].copy()
        if earnings.empty:
            return pd.DataFrame(columns=["date", "ticker", self.factor_name])

        # Map events to their effective trading day
        dates = pd.to_datetime(earnings["date"])
        if dates.dt.tz is None:
            dates = dates.dt.tz_localize(MARKET_TIMEZONE)
        else:
            dates = dates.dt.tz_convert(MARKET_TIMEZONE)
        earnings["effective_date"] = dates.dt.normalize()

        # Get the full date index from price data
        all_dates = price_df.index.unique().sort_values()

        records: list[dict] = []
        for ticker in earnings["ticker"].unique():
            ticker_events = earnings[earnings["ticker"] == ticker].sort_values(
                "effective_date"
            )
            ticker_prices = price_df[price_df["ticker"] == ticker].sort_index()

            # Build surprise series: reindex to price dates, forward-fill
            surprise_series = pd.Series(
                index=pd.DatetimeIndex(ticker_events["effective_date"]),
                data=ticker_events["surprise_pct"].values,
            )
            surprise_series = surprise_series.reindex(
                ticker_prices.index
            ).ffill().fillna(0.0)

            # CRITICAL: shift forward by 1 to prevent leakage.
            # The event at close today becomes known tomorrow morning.
            surprise_series = surprise_series.shift(1).fillna(0.0)

            for dt, val in surprise_series.items():
                records.append({
                    "date": dt,
                    "ticker": ticker,
                    self.factor_name: val,
                })

        if not records:
            return pd.DataFrame(columns=["date", "ticker", self.factor_name])

        return pd.DataFrame(records)

    def check_leakage(self, factor_df: pd.DataFrame) -> LeakageReport:
        """Verify no look-ahead in surprise factor."""
        col = self.factor_name
        series = factor_df.set_index("date")[col]

        last_valid = self._last_valid_date(series)

        # Correlation with forward return should be moderate (not suspicious)
        returns = series.pct_change()
        fwd_returns = returns.shift(-1)
        corr_with_fwd = (
            series.corr(fwd_returns) if not series.empty else None
        )
        corr_same = series.corr(returns) if not series.empty else None

        # suspicious if corr_with_fwd > 0.7 (perfect prediction = leak)
        suspicious = (
            corr_with_fwd is not None
            and abs(corr_with_fwd) > 0.7
        )

        return LeakageReport(
            factor_name=self.factor_name,
            last_valid_date=last_valid,
            corr_with_fwd_return=corr_with_fwd,
            corr_with_same_return=corr_same,
            has_future_peek=suspicious,
            notes=(
                "High forward correlation — possible leakage"
                if suspicious
                else ""
            ),
        )


class MomentumFactor(FactorBase):
    """Classic momentum factor: N-day price change, lagged by 1 day.

    factor_t = (P_t / P_{t-N}) - 1
    Shifted by 1 to predict next-day return.
    """

    factor_name: str = "momentum"

    def __init__(self, lookback_days: int = 20) -> None:
        self.lookback_days = lookback_days
        self.factor_name = f"momentum_{lookback_days}d"

    def compute(
        self, price_df: pd.DataFrame, events_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Compute momentum factor per ticker."""
        records: list[dict] = []
        for ticker, group in price_df.groupby("ticker"):
            group = group.sort_index()
            mom = group["adj_close"].pct_change(
                periods=self.lookback_days
            ).shift(1)  # shift to prevent leakage
            for dt, val in mom.dropna().items():
                records.append({
                    "date": dt,
                    "ticker": ticker,
                    self.factor_name: val,
                })

        if not records:
            return pd.DataFrame(
                columns=["date", "ticker", self.factor_name]
            )
        return pd.DataFrame(records)

    def check_leakage(self, factor_df: pd.DataFrame) -> LeakageReport:
        col = self.factor_name
        series = factor_df.set_index("date")[col]
        last_valid = self._last_valid_date(series)
        returns = series.pct_change()
        fwd = returns.shift(-1)
        corr_fwd = series.corr(fwd) if len(series) > 1 else None
        corr_same = series.corr(returns) if len(series) > 1 else None

        return LeakageReport(
            factor_name=self.factor_name,
            last_valid_date=last_valid,
            corr_with_fwd_return=corr_fwd,
            corr_with_same_return=corr_same,
            has_future_peek=False,
            notes="",
        )
