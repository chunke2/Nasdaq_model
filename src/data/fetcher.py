"""Abstract base for data fetchers.

Every data fetcher must subclass DataFetcher and implement fetch().
The base class handles caching and validation dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataFetcher(ABC):
    """Abstract base class for all data fetchers.

    Subclasses must implement:
    - source_name: str classvar — human-readable source identifier
    - fetch(*args, **kwargs) -> pd.DataFrame
    """

    source_name: str = "unknown"

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic validations applied after every fetch.

        Subclasses can override to add source-specific checks, but must
        call super().validate() first.
        """
        if df is None or df.empty:
            from src.utils.validation import DataError

            raise DataError(f"[{self.source_name}] fetched DataFrame is empty")

        if not isinstance(df.index, pd.DatetimeIndex):
            from src.utils.validation import DataError

            raise DataError(
                f"[{self.source_name}] DataFrame index must be DatetimeIndex, "
                f"got {type(df.index).__name__}"
            )

        # Ensure time is monotonic
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        return df

    def validate_columns(
        self, df: pd.DataFrame, required: list[str]
    ) -> None:
        """Check that all required columns are present."""
        missing = set(required) - set(df.columns)
        if missing:
            from src.utils.validation import DataError

            raise DataError(
                f"[{self.source_name}] missing required columns: {missing}"
            )

    @abstractmethod
    def fetch(self, *args: object, **kwargs: object) -> pd.DataFrame:
        """Fetch data from the source. Must return a DataFrame with DatetimeIndex."""
        ...
