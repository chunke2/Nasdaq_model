"""Earnings event detector.

Primary: real earnings calendar (Alpha Vantage EARNINGS endpoint).
Fallback: proxy detection via large returns + volume spikes.
"""

from __future__ import annotations

import pandas as pd

from src.events.base import EventDetector
from src.utils.logging_config import get_logger
from src.utils.timezone import get_trading_day

logger = get_logger(__name__)


class EarningsSurpriseDetector(EventDetector):
    """Detect earnings events — real calendar preferred, proxy as fallback.

    On init:
        real_calendar: pd.DataFrame from fetch_earnings_calendar().
        If None or empty, falls back to proxy heuristics.
    """

    event_type: str = "earnings"

    def __init__(
        self,
        return_threshold_pct: float = 3.0,
        volume_multiple: float = 1.5,
        lookback_days: int = 20,
        real_calendar: pd.DataFrame | None = None,
    ) -> None:
        self.return_threshold_pct = return_threshold_pct
        self.volume_multiple = volume_multiple
        self.lookback_days = lookback_days
        self._real_calendar = real_calendar
        self._used_real: bool = False

    @property
    def used_real_calendar(self) -> bool:
        """Whether real earnings calendar was used (vs proxy fallback)."""
        return self._used_real

    def detect(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Detect earnings events.

        Returns DataFrame: [date, ticker, event_type, description,
                            impact_estimate, surprise_pct]
        """
        real = self._detect_from_calendar(price_df)
        if not real.empty:
            self._used_real = True
            logger.info(
                "Using real earnings calendar: %d events", len(real)
            )
            return real

        self._used_real = False
        logger.info(
            "No real calendar available — using proxy detection"
        )
        return self._detect_proxy(price_df)

    def _detect_from_calendar(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Filter real calendar to events with available price data."""
        if self._real_calendar is None or self._real_calendar.empty:
            return pd.DataFrame()

        cal = self._real_calendar.copy()
        cal["date"] = pd.to_datetime(cal["date"])

        # Keep only events within price data range
        p_min, p_max = price_df.index.min(), price_df.index.max()
        cal = cal[(cal["date"] >= p_min) & (cal["date"] <= p_max)]

        if cal.empty:
            return pd.DataFrame()

        # Build output with real EPS data
        records = []
        for _, row in cal.iterrows():
            ticker = row["ticker"]
            evt_date = row["date"]
            surprise_pct_float = float(row.get("surprise_pct", 0.0) or 0.0)

            records.append({
                "date": evt_date,
                "ticker": ticker,
                "event_type": "earnings",
                "description": str(row.get("description", f"Earnings: {ticker}")),
                "impact_estimate": float(row.get("impact_estimate", 0.0) or 0.0),
                "surprise_pct": surprise_pct_float / 100.0,
            })

        df = pd.DataFrame(records)
        return df.sort_values("date").reset_index(drop=True)

    def _detect_proxy(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: detect via large returns + volume spikes."""
        events: list[dict] = []

        for ticker, group in price_df.groupby("ticker"):
            group = group.sort_index()
            returns = group["adj_close"].pct_change()
            avg_vol = group["volume"].rolling(self.lookback_days).mean()
            vol_ratio = group["volume"] / avg_vol

            for i in range(1, len(group)):
                ret_val = returns.iloc[i]
                vr = vol_ratio.iloc[i]
                if (
                    abs(ret_val) >= self.return_threshold_pct / 100
                    and vr >= self.volume_multiple
                ):
                    dt = get_trading_day(group.index[i])
                    direction = "beat" if ret_val > 0 else "miss"
                    events.append({
                        "date": dt,
                        "ticker": ticker,
                        "event_type": "earnings",
                        "description": f"Earnings proxy: {direction} ({ret_val:.1%})",
                        "impact_estimate": ret_val,
                        "surprise_pct": ret_val,
                    })

        if not events:
            logger.warning(
                "No proxy events (threshold=%.1f%%, vol_mult=%.1fx)",
                self.return_threshold_pct, self.volume_multiple,
            )
            return pd.DataFrame(
                columns=[
                    "date", "ticker", "event_type", "description",
                    "impact_estimate", "surprise_pct",
                ]
            )

        df = pd.DataFrame(events)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
