"""Abstract base class for all event detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class EventDetector(ABC):
    """Detect and classify market-impacting events from structured data.

    Subclasses must implement detect().
    """

    event_type: str = "unknown"

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect events in the input data.

        Args:
            data: Source-specific DataFrame (price, fundamentals, news, etc.)

        Returns:
            DataFrame with columns: [date, ticker, event_type, description, impact_estimate]
        """
        ...
