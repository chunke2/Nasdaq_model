"""Performance metrics for backtest evaluation.

All metrics computed from equity curve and trade log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series, rf_annual: float = 0.04, periods: int = 252
) -> float:
    """Annualized Sharpe ratio. Returns 0.0 if std is zero."""
    excess = returns - rf_annual / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.15 = 15% loss)."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def annual_return(equity: pd.Series, periods: int = 252) -> float:
    """Annualized return from equity curve."""
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods
    if years == 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annual_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility."""
    return float(returns.std() * np.sqrt(periods))


def hit_rate(trades: pd.DataFrame) -> float:
    """Fraction of winning trades. Trades must have 'pnl' column."""
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit / gross loss. inf if no losses."""
    if trades.empty:
        return 0.0
    gross_win = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def calmar_ratio(equity: pd.Series, periods: int = 252) -> float:
    """Annualized return / max drawdown (absolute value)."""
    ret = annual_return(equity, periods)
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    return ret / dd


def compute_all(
    equity: pd.Series,
    trades: pd.DataFrame,
    benchmark_equity: pd.Series | None = None,
) -> dict:
    """Compute a full metrics summary."""
    rets = equity.pct_change().dropna()
    metrics: dict = {
        "sharpe_ratio": round(sharpe_ratio(rets), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "annual_return": round(annual_return(equity), 4),
        "annual_vol": round(annual_volatility(rets), 4),
        "calmar_ratio": round(calmar_ratio(equity), 4),
        "hit_rate": round(hit_rate(trades), 4),
        "profit_factor": round(profit_factor(trades), 4),
        "n_trades": len(trades),
        "start_date": str(equity.index[0].date()),
        "end_date": str(equity.index[-1].date()),
    }

    if benchmark_equity is not None:
        bm_rets = benchmark_equity.pct_change().dropna()
        common = rets.index.intersection(bm_rets.index)
        if len(common) > 1:
            excess = rets.loc[common] - bm_rets.loc[common]
            metrics["information_ratio"] = round(
                float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0, 4
            )
            # Beta
            cov = np.cov(rets.loc[common], bm_rets.loc[common])
            if cov[0, 1] != 0:
                metrics["beta"] = round(float(cov[0, 1] / bm_rets.loc[common].var()), 4)
            metrics["benchmark_return"] = round(annual_return(benchmark_equity), 4)

    return metrics
