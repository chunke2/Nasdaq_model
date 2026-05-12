"""NASDAQ Event-Factor Model — backtesting layer.

Submodules:
- engine: backtest engine (walk-forward, event-triggered)
- metrics: performance metrics (Sharpe, MaxDD, hit rate, profit factor)
"""

from src.backtest.engine import BacktestEngine
from src.backtest import metrics

__all__ = ["BacktestEngine", "metrics"]

