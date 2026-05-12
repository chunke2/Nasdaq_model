"""Pydantic v2 data validation models.

All external data entering the system (API responses, DataFrames, config)
must be validated through Pydantic schemas defined here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── exception taxonomy ──────────────────────────────────────────────────────

class DataError(Exception):
    """Raised when input data fails validation."""


class ModelError(Exception):
    """Raised when a model encounters an invalid state."""


class ConfigError(Exception):
    """Raised when configuration is missing or malformed."""


# ── data schemas ─────────────────────────────────────────────────────────────

class PriceRecord(BaseModel):
    """Validated single-row price bar."""

    ticker: str = Field(min_length=1, max_length=10)
    date: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    adj_close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @field_validator("high")
    @classmethod
    def high_ge_low(cls, v: float, info: Any) -> float:
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError(f"high ({v}) < low ({low})")
        return v


class EventRecord(BaseModel):
    """Validated single event."""

    ticker: str = Field(min_length=1, max_length=10)
    event_date: datetime  # date the event occurred / was announced
    event_type: Literal[
        "earnings",
        "dividend",
        "split",
        "fomc",
        "cpi",
        "employment",
        "gdp",
        "news",
    ]
    description: str = Field(default="", max_length=500)
    source: str = Field(default="", max_length=200)
    impact_estimate: float | None = Field(
        default=None, description="Pre-event consensus estimate, if applicable"
    )


class LeakageReport(BaseModel):
    """Output of factor check_leakage() — records potential look-ahead bias."""

    factor_name: str
    last_valid_date: datetime | None = Field(
        default=None, description="Latest date with a non-NaN factor value"
    )
    corr_with_fwd_return: float | None = Field(
        default=None,
        description="Spearman correlation with forward return — suspicious if very high",
    )
    corr_with_same_return: float | None = Field(
        default=None,
        description="Spearman correlation with contemporaneous return — baseline comparison",
    )
    has_future_peek: bool = Field(
        default=False, description="True if factor construction used any future-looking ops"
    )
    notes: str = Field(default="")
