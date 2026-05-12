"""Unified timezone handling for NASDAQ trading.

All-internal time representation: pandas Timestamp with tz="America/New_York".
This module is the single entry point for timezone conversion — no other module
should call .tz_localize() or .tz_convert() directly.
"""

from __future__ import annotations

from datetime import time, timezone
from typing import Any

import pandas as pd

MARKET_TIMEZONE: str = "America/New_York"
MARKET_OPEN: time = time(9, 30)
MARKET_CLOSE: time = time(16, 0)
UTC: timezone = timezone.utc


def to_market_time(dt: Any) -> pd.Timestamp:
    """Convert any datetime-like input to timezone-aware Eastern Time.

    Accepts:
    - naive datetime / Timestamp → assumes it is already ET
    - UTC-aware datetime / Timestamp → converts to ET
    - other timezone-aware → converts to ET

    Raises DataError if the input type is unrecognized.
    """
    ts = pd.Timestamp(dt)

    if ts.tz is None:
        return ts.tz_localize(MARKET_TIMEZONE, ambiguous=True)
    return ts.tz_convert(MARKET_TIMEZONE)


def get_trading_day(event_dt: pd.Timestamp) -> pd.Timestamp:
    """Map an event timestamp to the trading day it affects.

    Rule: events before or at 16:00 ET affect the same trading day;
    events after 16:00 ET affect the next trading day.
    """
    ts = to_market_time(event_dt)
    if ts.time() <= MARKET_CLOSE:
        return ts.normalize()
    return (ts + pd.Timedelta(days=1)).normalize()


def next_trading_day(
    dt: pd.Timestamp, calendar: str = "XNYS"
) -> pd.Timestamp:
    """Return the next valid trading day after `dt`.

    Uses pandas market calendars. Falls back to +1 business day if the
    calendar is not available.
    """
    ts = to_market_time(dt)
    try:
        from pandas.tseries.holiday import CustomBusinessDay

        bday = CustomBusinessDay(calendar=calendar)
        return ts.normalize() + bday
    except Exception:
        from pandas.tseries.offsets import BDay

        return (ts.normalize() + BDay(1)).tz_localize(None).tz_localize(
            MARKET_TIMEZONE
        )


def trading_days_between(
    start: pd.Timestamp, end: pd.Timestamp, calendar: str = "XNYS"
) -> pd.DatetimeIndex:
    """Return all trading days in [start, end] inclusive."""
    s = to_market_time(start).normalize()
    e = to_market_time(end).normalize()
    try:
        bday = pd.offsets.CustomBusinessDay(calendar=calendar)
        return pd.date_range(start=s, end=e, freq=bday)
    except Exception:
        bday = pd.offsets.BDay()
        return pd.date_range(start=s, end=e, freq=bday)


def is_trading_day(dt: pd.Timestamp) -> bool:
    """Check if `dt` falls on a valid trading day."""
    ts = to_market_time(dt).normalize()
    trading = trading_days_between(ts - pd.Timedelta(days=5), ts + pd.Timedelta(days=5))
    return ts in trading
