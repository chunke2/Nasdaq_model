"""Earnings event detector.

Detects earnings announcement dates and computes earnings surprise
(synthetic: uses large price moves as proxies for earnings events).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.events.base import EventDetector
from src.utils.logging_config import get_logger
from src.utils.timezone import get_trading_day

logger = get_logger(__name__)


class EarningsSurpriseDetector(EventDetector):
    """Detect earnings events from price data using proxy heuristics.

    In production, this would use actual earnings calendar data (Polygon
    or Finnhub). For development, we use:
    - Large absolute daily returns (top 5% by magnitude) as earnings proxy
    - Volume spike > 2x 20-day average as confirmation

    The detector marks each event with an estimated surprise magnitude
    (signed return on the event day) that serves as the core factor.
    """

    event_type: str = "earnings"

    def __init__(
        self,
        return_threshold_pct: float = 3.0,
        volume_multiple: float = 1.5,
        lookback_days: int = 20,
    ) -> None:
        self.return_threshold_pct = return_threshold_pct
        self.volume_multiple = volume_multiple
        self.lookback_days = lookback_days

    def detect(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Detect likely earnings events from price/volume data.

        Args:
            price_df: Price DataFrame with [adj_close, volume, ticker] columns
                      and DatetimeIndex.

        Returns:
            DataFrame: [date, ticker, event_type, description, impact_estimate, surprise_pct]
        """
        events: list[dict] = []

        for ticker, group in price_df.groupby("ticker"):
            group = group.sort_index()
            returns = group["adj_close"].pct_change()
            avg_vol = group["volume"].rolling(self.lookback_days).mean()
            vol_ratio = group["volume"] / avg_vol

            for i in range(1, len(group)):
                ret = returns.iloc[i]
                vr = vol_ratio.iloc[i]
                if abs(ret) >= self.return_threshold_pct / 100 and vr >= self.volume_multiple:
                    dt = get_trading_day(group.index[i])
                    direction = "beat" if ret > 0 else "miss"
                    events.append({
                        "date": dt,
                        "ticker": ticker,
                        "event_type": "earnings",
                        "description": f"Earnings proxy: {direction} ({ret:.1%})",
                        "impact_estimate": ret,  # signed return = surprise proxy
                        "surprise_pct": ret,
                    })

        if not events:
            logger.warning(
                "No earnings events detected (threshold=%.1f%%, vol_mult=%.1fx)",
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
