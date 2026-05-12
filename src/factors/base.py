"""Abstract base class for all factor constructors.

Every factor MUST implement check_leakage() per the Anti-Leakage rules
defined in CLAUDE.md section 2.5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.utils.validation import LeakageReport


class FactorBase(ABC):
    """Abstract factor with mandatory leakage check contract.

    Subclasses must implement:
    - compute(price_df, events_df) -> pd.DataFrame
    - check_leakage(factor_df) -> LeakageReport
    """

    factor_name: str = "unknown"

    @abstractmethod
    def compute(
        self, price_df: pd.DataFrame, events_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Compute factor values.

        Args:
            price_df: Price DataFrame with DatetimeIndex, columns include adj_close.
            events_df: Optional event DataFrame from EventDetector.detect().

        Returns:
            DataFrame with DatetimeIndex, columns = [factor_name] per ticker,
            or MultiIndex (date, ticker).
        """
        ...

    @abstractmethod
    def check_leakage(self, factor_df: pd.DataFrame) -> LeakageReport:
        """Check for look-ahead bias in factor construction.

        Must verify:
        - factor at time t uses only info available at or before t
        - no shift(-1) or center=True in rolling windows
        """
        ...

    @staticmethod
    def _last_valid_date(series: pd.Series) -> pd.Timestamp | None:
        """Return the last date with a non-NaN value."""
        valid = series.dropna()
        if valid.empty:
            return None
        return valid.index[-1]  # type: ignore[return-value]
