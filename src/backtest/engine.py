"""Event-driven backtest engine.

Executes trades based on model predictions at event dates.
Strict temporal ordering: prediction uses only data available before the event.
"""

from __future__ import annotations

import pandas as pd

from src.backtest import metrics as bt_metrics
from src.models.base import ModelBase
from src.utils.experiment_logger import ExperimentLogger
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """Event-driven backtest engine.

    For each detected event:
    1. Model predicts direction (POSITIVE/NEGATIVE)
    2. Enter position on the event's effective trading day
    3. Hold for `hold_days` trading days
    4. Exit and record PnL

    Anti-leakage: model prediction at event_date uses only data
    available on or before that date.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        hold_days: int = 5,
        transaction_cost_bps: float = 10.0,
        max_position_pct: float = 0.20,
        log_experiment: bool = True,
    ) -> None:
        self.initial_capital = initial_capital
        self.hold_days = hold_days
        self.transaction_cost_bps = transaction_cost_bps
        self.max_position_pct = max_position_pct
        self._log_experiment = log_experiment
        self._exp_logger = ExperimentLogger() if log_experiment else None

        # Results populated after run()
        self.equity: pd.Series | None = None
        self.trades: pd.DataFrame | None = None
        self.metrics: dict = {}

    def run(
        self,
        model: ModelBase,
        price_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> dict:
        """Execute event-driven backtest.

        Args:
            model: A fitted model with predict() method.
            price_df: Price data with [ticker, adj_close] and DatetimeIndex.
            events_df: Detected events with [date, ticker, event_type].

        Returns:
            dict with keys: equity (pd.Series), trades (pd.DataFrame), metrics (dict).
        """
        if events_df.empty:
            logger.warning("No events to backtest")
            return {"equity": pd.Series(), "trades": pd.DataFrame(), "metrics": {}}

        events = events_df.sort_values("date").copy()
        events["effective_date"] = pd.to_datetime(events["date"]).dt.normalize()

        trades, equity_curve = self._simulate(events, price_df, model)

        self.equity = equity_curve
        self.trades = trades
        self.metrics = bt_metrics.compute_all(equity_curve, trades)

        if self._log_experiment and self._exp_logger:
            self._record_experiment()

        return {"equity": equity_curve, "trades": trades, "metrics": self.metrics}

    def _simulate(
        self,
        events: pd.DataFrame,
        price_df: pd.DataFrame,
        model: ModelBase,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Walk forward through events, executing trades."""
        cash = self.initial_capital
        positions: list[dict] = []  # open positions
        trades_list: list[dict] = []
        equity_log: list[dict] = []

        # Build a price access structure
        price_by_ticker: dict[str, pd.DataFrame] = {}
        for ticker, group in price_df.groupby("ticker"):
            price_by_ticker[ticker] = group.sort_index()

        # Get all unique trading days
        all_dates = price_df.index.unique().sort_values()

        # Track daily mark-to-market
        for day in all_dates:
            # Mark open positions
            pnl_today = 0.0
            closed = []
            for i, pos in enumerate(positions):
                if day >= pos["exit_date"]:
                    # Close position
                    exit_px = self._get_price(price_by_ticker, pos["ticker"], day)
                    if exit_px is None:
                        exit_px = pos["entry_price"]
                    gross_pnl = (
                        (exit_px - pos["entry_price"]) / pos["entry_price"]
                        * pos["capital"]
                    )
                    if pos["direction"] == "NEGATIVE":
                        gross_pnl = -gross_pnl
                    cost = pos["capital"] * 2 * self.transaction_cost_bps / 10000
                    net_pnl = gross_pnl - cost
                    pnl_today += net_pnl
                    cash += pos["capital"] + net_pnl
                    trades_list.append({
                        "entry_date": pos["entry_date"],
                        "exit_date": day,
                        "ticker": pos["ticker"],
                        "direction": pos["direction"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_px,
                        "capital": pos["capital"],
                        "pnl": net_pnl,
                        "pnl_pct": net_pnl / pos["capital"],
                    })
                    closed.append(i)
                else:
                    # Mark to market
                    mkt_px = self._get_price(price_by_ticker, pos["ticker"], day)
                    if mkt_px is not None:
                        unrealized = (
                            (mkt_px - pos["entry_price"]) / pos["entry_price"]
                            * pos["capital"]
                        )
                        if pos["direction"] == "NEGATIVE":
                            unrealized = -unrealized
                        pnl_today += unrealized

            for i in reversed(closed):
                positions.pop(i)

            # Check for new events on this day
            day_events = events[events["effective_date"] == day]
            for _, evt in day_events.iterrows():
                ticker = evt["ticker"]
                # Predict direction using model
                prediction = self._safe_predict(model, evt)
                if prediction in ("POSITIVE", "NEGATIVE"):
                    # Size position
                    pos_capital = cash * self.max_position_pct
                    entry_px = self._get_price(price_by_ticker, ticker, day)
                    if entry_px is None:
                        continue
                    exit_date = day + pd.Timedelta(days=self.hold_days * 2)
                    # Cost to enter
                    cost = pos_capital * self.transaction_cost_bps / 10000
                    cash -= cost
                    positions.append({
                        "entry_date": day,
                        "exit_date": exit_date,
                        "ticker": ticker,
                        "direction": prediction,
                        "entry_price": entry_px,
                        "capital": pos_capital,
                    })

            # Record equity
            position_value = sum(p["capital"] for p in positions)
            equity_log.append({"date": day, "equity": cash + position_value})

        equity_curve = pd.DataFrame(equity_log).set_index("date")["equity"]
        trades_df = pd.DataFrame(trades_list)
        return trades_df, equity_curve

    def _safe_predict(self, model: ModelBase, event: pd.Series) -> str:
        """Predict direction for a single event. Returns NEUTRAL on failure."""
        try:
            mini_df = pd.DataFrame([{
                "date": event["date"],
                "ticker": event["ticker"],
                "event_type": event.get("event_type", "earnings"),
                "surprise_pct": event.get("surprise_pct", 0.0),
            }])
            pred = model.predict(mini_df)
            return str(pred.iloc[0]) if len(pred) > 0 else "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    @staticmethod
    def _get_price(
        price_by_ticker: dict[str, pd.DataFrame],
        ticker: str,
        day: pd.Timestamp,
    ) -> float | None:
        """Get adj_close for ticker on or before day. Returns None if not found."""
        df = price_by_ticker.get(ticker)
        if df is None:
            return None
        available = df[df.index <= day]
        if available.empty:
            return None
        return float(available["adj_close"].iloc[-1])

    def _record_experiment(self) -> None:
        """Log backtest results to experiments/."""
        if self._exp_logger is None:
            return
        cfg = {
            "model_type": "event_study",
            "initial_capital": self.initial_capital,
            "hold_days": self.hold_days,
            "transaction_cost_bps": self.transaction_cost_bps,
            "max_position_pct": self.max_position_pct,
            "factors_used": ["earnings_surprise"],
        }
        exp_id = self._exp_logger.start("backtest", cfg)
        self._exp_logger.log_metrics({
            "sharpe_ratio": self.metrics.get("sharpe_ratio"),
            "max_drawdown": self.metrics.get("max_drawdown"),
            "hit_rate": self.metrics.get("hit_rate"),
            "profit_factor": self.metrics.get("profit_factor"),
            "annual_return": self.metrics.get("annual_return"),
            "annual_vol": self.metrics.get("annual_vol"),
            "n_trades": float(self.metrics.get("n_trades", 0)),
        })
        self._exp_logger.finalize(
            f"# Backtest Results — {exp_id}\n\n"
            f"**Sharpe**: {self.metrics.get('sharpe_ratio')}\n"
            f"**Max DD**: {self.metrics.get('max_drawdown')}\n"
            f"**Hit Rate**: {self.metrics.get('hit_rate')}\n"
            f"**Annual Return**: {self.metrics.get('annual_return')}\n"
            f"**Trades**: {self.metrics.get('n_trades')}\n"
        )
