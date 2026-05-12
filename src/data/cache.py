"""Local Parquet-based cache for financial data.

Caches fetched data to avoid repeated API calls. Handles incremental updates
by storing snapshots keyed by (source, symbol, date_range).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CACHE_ROOT: Path = Path("data")


class DataCache:
    """Parquet-backed cache for DataFrames, keyed by (source, symbol)."""

    def __init__(self, root: Path = CACHE_ROOT) -> None:
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)

    def _key(self, source: str, symbol: str) -> str:
        return f"{source}__{symbol}.parquet"

    def load(self, source: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data if it exists. Returns None on miss."""
        path = self.raw / self._key(source, symbol)
        if not path.exists():
            return None
        logger.debug("Cache hit: %s / %s", source, symbol)
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def store(self, df: pd.DataFrame, source: str, symbol: str) -> None:
        """Write DataFrame to raw cache."""
        path = self.raw / self._key(source, symbol)
        df.to_parquet(path, compression="snappy")
        logger.info("Cached: %s / %s → %s (%d rows)", source, symbol, path, len(df))

    def update(
        self, df_new: pd.DataFrame, source: str, symbol: str
    ) -> pd.DataFrame:
        """Merge new data with cached data, deduplicate by index, store."""
        existing = self.load(source, symbol)
        if existing is not None:
            combined = pd.concat([existing, df_new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = df_new.sort_index()
        self.store(combined, source, symbol)
        return combined

    def stale(
        self, source: str, symbol: str, max_age_hours: int = 24
    ) -> bool:
        """Check if cached data is older than max_age_hours."""
        path = self.raw / self._key(source, symbol)
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 3600
        return age > max_age_hours
