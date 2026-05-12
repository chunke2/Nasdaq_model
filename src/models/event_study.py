"""Event Study methodology — the core Phase 1 model.

Classic event study following MacKinlay (1997):
1. Estimate market model parameters over estimation window
2. Compute abnormal returns (AR) over event window
3. Aggregate to cumulative abnormal returns (CAR)
4. Test statistical significance
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.base import ModelBase
from src.utils.logging_config import get_logger
from src.utils.experiment_logger import ExperimentLogger

logger = get_logger(__name__)


class EventStudyModel(ModelBase):
    """Event study with market-model benchmark.

    For each event:
    - Estimation window: [-N_est, -1] days before event
    - Event window: [0, N_event] days after event
    - Benchmark: R_it = α + β * R_mt + ε
    - AR_t = R_actual_t - R_expected_t
    - CAR = sum of AR over event window

    Output: direction prediction (POSITIVE/NEGATIVE) based on CAR sign.
    """

    model_type: str = "event_study"

    def __init__(
        self,
        estimation_window: int = 60,
        event_window: int = 5,
        min_events: int = 3,
    ) -> None:
        self.estimation_window = estimation_window  # trading days before event
        self.event_window = event_window            # trading days after event
        self.min_events = min_events
        self._results: dict = {}
        self._exp_logger = ExperimentLogger()

    def fit(
        self,
        price_df: pd.DataFrame,
        events_df: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
        *,
        log_experiment: bool = False,
    ) -> "EventStudyModel":
        """Run event study on price data with detected events.

        Args:
            price_df: Price DataFrame with [ticker, adj_close] and DatetimeIndex.
            events_df: Detected events with [date, ticker, event_type, surprise_pct].
            benchmark_returns: Daily returns of the benchmark (e.g. ^IXIC).
                               If None, uses equal-weighted average of all tickers.

        Returns:
            self, with results stored in self._results.
        """
        if events_df.empty:
            logger.warning("No events to study")
            return self

        # Prepare benchmark returns
        if benchmark_returns is None:
            benchmark_returns = self._equal_weight_benchmark(price_df)

        all_ars: list[float] = []
        event_results: list[dict] = []

        for _, event in events_df.iterrows():
            ticker = event["ticker"]
            event_date = pd.Timestamp(event["date"])

            ticker_prices = price_df[
                price_df["ticker"] == ticker
            ].sort_index()
            if ticker_prices.empty:
                continue

            ars = self._run_single_event(
                ticker_prices, benchmark_returns, event_date
            )
            if ars is not None:
                all_ars.extend(ars)
                car = float(np.sum(ars))
                event_results.append({
                    "date": event_date,
                    "ticker": ticker,
                    "car": car,
                    "direction": "POSITIVE" if car > 0 else "NEGATIVE",
                    "surprise_pct": event.get("surprise_pct", 0.0),
                    "n_abnormal_returns": len(ars),
                })

        self._results = {
            "events": event_results,
            "mean_car": float(np.mean([e["car"] for e in event_results]))
                       if event_results else 0.0,
            "mean_ar": float(np.mean(all_ars)) if all_ars else 0.0,
            "n_events": len(event_results),
            "car_t_stat": self._car_t_test(all_ars, event_results),
        }

        if log_experiment:
            import json
            self._exp_logger.start(
                "event_study",
                {
                    "model_type": self.model_type,
                    "estimation_window": self.estimation_window,
                    "event_window": self.event_window,
                    "n_events": self._results["n_events"],
                },
            )
            self._exp_logger.log_metrics({
                "mean_car": self._results["mean_car"],
                "n_events": float(self._results["n_events"]),
            })
            self._exp_logger.finalize(
                f"# Event Study Results\n\n"
                f"Mean CAR: {self._results['mean_car']:.4%}\n"
                f"CAR t-stat: {self._results.get('car_t_stat', 'N/A')}\n"
                f"Events: {self._results['n_events']}\n"
            )

        return self

    def _run_single_event(
        self,
        ticker_prices: pd.DataFrame,
        benchmark_returns: pd.Series,
        event_date: pd.Timestamp,
    ) -> list[float] | None:
        """Compute AR for one event. Returns list of AR values or None."""
        # Find event date index in ticker prices
        ticker_prices = ticker_prices.sort_index()
        try:
            loc = ticker_prices.index.get_loc(event_date)
        except KeyError:
            return None

        if isinstance(loc, slice):
            return None
        loc = int(loc)

        # Estimation window: [-estimation_window, -1]
        est_start = max(0, loc - self.estimation_window - 1)
        est_end = max(0, loc - 1)
        if est_end - est_start < 10:
            return None  # not enough data for estimation

        # Event window: [0, event_window]
        evt_end = min(len(ticker_prices) - 1, loc + self.event_window)
        if evt_end <= loc:
            return None

        # Estimation data
        est_prices = ticker_prices.iloc[est_start:est_end + 1]
        est_returns = est_prices["adj_close"].pct_change().dropna()

        # Align benchmark returns
        bm_est = benchmark_returns.reindex(est_returns.index).dropna()
        if len(bm_est) < 10:
            return None

        common_idx = est_returns.index.intersection(bm_est.index)
        y = est_returns.loc[common_idx].values
        X = bm_est.loc[common_idx].values

        # Market model: r_i = α + β * r_m
        X_sm = sm.add_constant(X)
        try:
            ols = sm.OLS(y, X_sm).fit()
        except Exception:
            return None

        alpha, beta = ols.params[0], ols.params[1]

        # Event window returns
        evt_prices = ticker_prices.iloc[loc:evt_end + 1]
        evt_returns = evt_prices["adj_close"].pct_change().dropna()
        if evt_returns.empty:
            return None

        bm_evt = benchmark_returns.reindex(evt_returns.index).dropna()
        common_evt = evt_returns.index.intersection(bm_evt.index)
        actual = evt_returns.loc[common_evt].values
        bm_vals = bm_evt.loc[common_evt].values

        # AR = actual - (α + β * r_m)
        expected = alpha + beta * bm_vals
        ar = (actual - expected).tolist()
        return [float(x) for x in ar]

    def predict(self, events_df: pd.DataFrame) -> pd.Series:
        """Predict direction for each event based on CAR sign from study.

        Returns Series of "POSITIVE" / "NEGATIVE" per event index.
        """
        if not self._results or not self._results.get("events"):
            return pd.Series(["NEUTRAL"] * len(events_df), index=events_df.index)

        # Map past event dates to directions
        past_dirs: dict[Tuple[str, pd.Timestamp], str] = {}
        for e in self._results["events"]:
            key = (e["ticker"], e["date"])
            past_dirs[key] = e["direction"]

        predictions: list[str] = []
        for _, evt in events_df.iterrows():
            key = (evt["ticker"], pd.Timestamp(evt["date"]))
            predictions.append(past_dirs.get(key, "NEUTRAL"))

        return pd.Series(predictions, index=events_df.index)

    @property
    def results(self) -> dict:
        """Return the event study results."""
        return self._results

    def _equal_weight_benchmark(self, price_df: pd.DataFrame) -> pd.Series:
        """Compute equal-weighted average return across all tickers."""
        returns_by_ticker: list[pd.Series] = []
        for _, group in price_df.groupby("ticker"):
            r = group.sort_index()["adj_close"].pct_change()
            returns_by_ticker.append(r)
        if not returns_by_ticker:
            return pd.Series(dtype=float)
        combined = pd.concat(returns_by_ticker, axis=1)
        return combined.mean(axis=1)

    @staticmethod
    def _car_t_test(
        all_ars: list[float], event_results: list[dict]
    ) -> float | None:
        """Cross-sectional t-test: is mean CAR significantly non-zero?"""
        cars = [e["car"] for e in event_results]
        if len(cars) < 3:
            return None
        n = len(cars)
        mean_car = np.mean(cars)
        if np.std(cars, ddof=1) == 0:
            return None
        se = np.std(cars, ddof=1) / np.sqrt(n)
        return float(mean_car / se)
