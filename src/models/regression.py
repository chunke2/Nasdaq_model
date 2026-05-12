"""Multi-factor regression model for return direction prediction.

Combines earnings surprise, momentum, and other factors into a single
predictive model. Supports:
- Logistic regression: binary direction (POSITIVE/NEGATIVE)
- Linear regression: continuous return prediction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import ModelBase
from src.utils.logging_config import get_logger
from src.utils.experiment_logger import ExperimentLogger

logger = get_logger(__name__)


class MultiFactorModel(ModelBase):
    """Predict return direction from multiple factors.

    Usage:
        model = MultiFactorModel(model_type="logistic")
        model.fit(factor_df, price_df)  # builds X, y internally
        direction = model.predict(new_factor_df, new_price_df)
    """

    model_type: str = "multi_factor"

    def __init__(
        self,
        model_type: str = "logistic",
        factor_names: list[str] | None = None,
        target_horizon: int = 1,
    ) -> None:
        """
        Args:
            model_type: "logistic" for direction, "linear" for return magnitude.
            factor_names: List of factor column names to use. Auto-detected if None.
            target_horizon: Days ahead to predict (1 = next day).
        """
        self._model_type = model_type
        self._factor_names = factor_names or []
        self._target_horizon = target_horizon
        self._scaler = StandardScaler()
        self._model = (
            LogisticRegression(max_iter=1000, class_weight="balanced")
            if model_type == "logistic"
            else LinearRegression()
        )
        self._fitted = False
        self._feature_names: list[str] = []
        self._coefficients: dict[str, float] = {}
        self._exp_logger = ExperimentLogger()

    def fit(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        *,
        log_experiment: bool = False,
    ) -> "MultiFactorModel":
        """Fit regression model on factor data.

        factor_df must have: [date, ticker, <factor columns>]
        price_df must have: [ticker, adj_close] with DatetimeIndex.

        Target: forward return direction (POSITIVE if return>0 else NEGATIVE).
        """
        X, y = self._build_xy(factor_df, price_df)
        if X.empty:
            logger.warning("No training samples after alignment")
            return self

        self._feature_names = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)

        self._model.fit(X_scaled, y)
        self._fitted = True

        # Extract coefficients for attribution
        if self._model_type == "logistic":
            coefs = self._model.coef_[0]
        else:
            coefs = self._model.coef_
        self._coefficients = dict(zip(self._feature_names, coefs))

        logger.info(
            "MultiFactorModel fitted: %d samples, %d features, "
            "class balance: pos=%.1f%%",
            len(X),
            len(self._feature_names),
            float((y == "POSITIVE").mean()) * 100,
        )

        if log_experiment:
            self._exp_logger.start(
                "regression",
                {
                    "model_type": self._model_type,
                    "factors_used": self._feature_names,
                    "n_samples": len(X),
                    "coefficients": self._coefficients,
                },
            )
            self._exp_logger.log_factor_attribution(
                {k: {"coefficient": float(v)} for k, v in self._coefficients.items()}
            )
            self._exp_logger.finalize(
                f"# Multi-Factor Regression\n\n"
                f"**Model**: {self._model_type}\n"
                f"**Features**: {self._feature_names}\n"
                f"**Samples**: {len(X)}\n"
                f"**Coefficients**: {self._coefficients}\n"
            )

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict direction or return for each row.

        If fitted as logistic: returns pd.Series of "POSITIVE"/"NEGATIVE".
        If fitted as linear: returns pd.Series of predicted returns.
        """
        if not self._fitted:
            return pd.Series(["NEUTRAL"] * len(X), index=X.index)

        # Ensure columns match training
        X_aligned = X[self._feature_names].copy()
        X_scaled = self._scaler.transform(X_aligned)

        if self._model_type == "logistic":
            preds = self._model.predict(X_scaled)
            return pd.Series(preds, index=X.index)
        else:
            preds = self._model.predict(X_scaled)
            return pd.Series(preds, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (logistic only)."""
        if not self._fitted or self._model_type != "logistic":
            return np.zeros((len(X), 2))
        X_aligned = X[self._feature_names].copy()
        X_scaled = self._scaler.transform(X_aligned)
        return self._model.predict_proba(X_scaled)

    def get_factor_attribution(self) -> dict:
        """Return factor name → coefficient mapping."""
        return self._coefficients

    def _build_xy(
        self, factor_df: pd.DataFrame, price_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build aligned X (factors) and y (forward return direction)."""
        if self._factor_names:
            factor_cols = self._factor_names
        else:
            # Auto-detect factor columns (exclude date, ticker)
            exclude = {"date", "ticker"}
            factor_cols = [c for c in factor_df.columns if c not in exclude]

        if not factor_cols:
            return pd.DataFrame(), pd.Series(dtype=str)

        # Pivot factor_df: index=date, columns=(ticker, factor)
        samples: list[dict] = []
        targets: list[str] = []

        for ticker in factor_df["ticker"].unique():
            f_sub = factor_df[factor_df["ticker"] == ticker].copy()
            p_sub = price_df[price_df["ticker"] == ticker].copy()

            if f_sub.empty or p_sub.empty:
                continue

            f_sub = f_sub.set_index("date").sort_index()
            p_sub = p_sub.sort_index()

            # Compute forward return direction
            returns = p_sub["adj_close"].pct_change().shift(
                -self._target_horizon
            )
            direction = returns.apply(
                lambda r: "POSITIVE" if r > 0 else ("NEGATIVE" if r < 0 else None)
            )

            for date in f_sub.index:
                if date not in direction.index:
                    continue
                d = direction.loc[date]
                if d is None:
                    continue
                row = {"date": date, "ticker": ticker}
                for col in factor_cols:
                    if col in f_sub.columns:
                        row[col] = float(f_sub.loc[date, col])
                samples.append(row)
                targets.append(d)

        if not samples:
            return pd.DataFrame(), pd.Series(dtype=str)

        X = pd.DataFrame(samples)
        X = X.drop(columns=["date", "ticker"], errors="ignore")
        X = X.fillna(0.0)
        y = pd.Series(targets, name="direction")

        # Remove rows where target is NaN or None
        valid_mask = y.notna() & (y != "None") & (y != None)
        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()

        return X, y
