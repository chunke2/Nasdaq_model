"""Abstract base class for all models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ModelBase(ABC):
    """Abstract model with fit/predict contract.

    Subclasses implement:
    - fit(X, y) -> self
    - predict(X) -> pd.Series
    """

    model_type: str = "unknown"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        log_experiment: bool = False,
    ) -> "ModelBase":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...
