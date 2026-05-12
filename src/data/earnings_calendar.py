"""Real earnings calendar fetcher.

Tries multiple sources in priority order:
1. Alpha Vantage EARNINGS endpoint (free: 25 calls/day, full history)
2. yfinance calendar (rate-limited but sometimes works)
3. Falls back to proxy detection
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.data.cache import DataCache
from src.data.config_loader import load_secrets
from src.utils.logging_config import get_logger
from src.utils.timezone import MARKET_TIMEZONE

logger = get_logger(__name__)


def fetch_earnings_alpha_vantage(
    ticker: str, api_key: str
) -> pd.DataFrame:
    """Fetch historical earnings dates from Alpha Vantage.

    Returns DataFrame with columns: [date, ticker, reported_eps, estimated_eps, surprise, surprise_pct]
    """
    import requests

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS",
        "symbol": ticker,
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    quarterly = data.get("quarterlyEarnings", [])
    if not quarterly:
        logger.debug("Alpha Vantage: no earnings data for %s", ticker)
        return pd.DataFrame()

    records = []
    for e in quarterly:
        reported = e.get("reportedDate", "")
        if not reported or reported == "None":
            continue
        try:
            reported_eps = float(e.get("reportedEPS", 0) or 0)
            estimated_eps = float(e.get("estimatedEPS", 0) or 0)
            surprise_amt = float(e.get("surprise", 0) or 0)
            surprise_pct_val = float(e.get("surprisePercentage", 0) or 0)
        except (ValueError, TypeError):
            reported_eps = 0.0
            estimated_eps = 0.0
            surprise_amt = 0.0
            surprise_pct_val = 0.0

        records.append({
            "date": pd.Timestamp(reported),
            "ticker": ticker.upper(),
            "reported_eps": reported_eps,
            "estimated_eps": estimated_eps,
            "surprise": surprise_amt,
            "surprise_pct": surprise_pct_val,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize(MARKET_TIMEZONE)
    return df.sort_values("date").reset_index(drop=True)


def fetch_earnings_calendar(
    tickers: List[str],
    start: str,
    end: str,
    cache: Optional[DataCache] = None,
) -> pd.DataFrame:
    """Fetch real earnings dates for multiple tickers.

    Priority: Alpha Vantage → returns empty if API key missing or fails.

    Returns:
        DataFrame: [date, ticker, event_type="earnings", description,
                     impact_estimate, surprise_pct, reported_eps, estimated_eps]
    """
    secrets = load_secrets()
    av_key = secrets.get("alpha_vantage", {}).get("api_key", "")

    cache_obj = cache or DataCache()
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        t = ticker.upper()
        df = pd.DataFrame()

        # Try Alpha Vantage
        if av_key:
            cache_key_av = f"earnings_av__{t}"
            if not cache_obj.stale("earnings_av", cache_key_av):
                cached = cache_obj.load("earnings_av", cache_key_av)
                if cached is not None and not cached.empty:
                    df = cached

            if df.empty:
                try:
                    df = fetch_earnings_alpha_vantage(t, av_key)
                    if not df.empty:
                        cache_obj.store(df, "earnings_av", cache_key_av)
                        logger.info(
                            "Alpha Vantage earnings: %s → %d quarters",
                            t, len(df),
                        )
                except Exception as exc:
                    logger.warning("Alpha Vantage earnings failed for %s: %s", t, exc)

        if not df.empty:
            # Add standardized event columns
            s = pd.Timestamp(start, tz=MARKET_TIMEZONE)
            e = pd.Timestamp(end, tz=MARKET_TIMEZONE)
            df = df[(df["date"] >= s) & (df["date"] <= e)]

            if not df.empty:
                df["event_type"] = "earnings"
                df["description"] = df.apply(
                    lambda r: f"EPS: {r.reported_eps:.2f} vs est {r.estimated_eps:.2f} "
                              f"(surprise: {r.surprise_pct:+.1f}%)",
                    axis=1,
                )
                df["impact_estimate"] = df["surprise_pct"] / 100.0
                all_frames.append(df)

    if not all_frames:
        logger.warning("No real earnings data fetched for any ticker")
        return pd.DataFrame(
            columns=[
                "date", "ticker", "event_type", "description",
                "impact_estimate", "surprise_pct", "reported_eps", "estimated_eps",
            ]
        )

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)
    logger.info(
        "Earnings calendar: %d events across %d tickers",
        len(result), result["ticker"].nunique(),
    )
    return result
