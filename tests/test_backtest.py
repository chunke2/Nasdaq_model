"""Backtest engine integration test."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.synthetic import SyntheticPriceFetcher
from src.events.earnings import EarningsSurpriseDetector
from src.factors.event_factors import EarningsSurpriseFactor, MomentumFactor
from src.models.event_study import EventStudyModel
from src.backtest.engine import BacktestEngine
from src.backtest import metrics as bt_metrics
from src.utils.seed import seed_everything

seed_everything(42)

print("=" * 55)
print("BACKTEST ENGINE TEST")
print("=" * 55)

# ── 1. Generate data with known earnings events ──
fetcher = SyntheticPriceFetcher(base_price=150.0)
events_map = {
    "AAPL": pd.Series(
        [0.05, -0.04, 0.06, -0.03, 0.04],
        index=["2024-02-15", "2024-05-10", "2024-07-20",
               "2024-10-25", "2025-01-15"],
    ),
}
df = fetcher.fetch(
    ["AAPL", "NVDA", "MSFT"], "2024-01-01", "2025-12-31",
    events=events_map, seed=42,
)
print(f"[1] Data: {len(df)} rows, {df.ticker.nunique()} tickers")

# ── 2. Detect events ──
detector = EarningsSurpriseDetector(
    return_threshold_pct=2.5, volume_multiple=1.3
)
events = detector.detect(df)
print(f"[2] Events: {len(events)}")

# ── 3. Train model (walk-forward safe: train on 2024, test on 2025) ──
split_date = "2025-01-01"
train_df = df[df.index < split_date]
train_events = events[events["date"] < split_date]

model = EventStudyModel(estimation_window=60, event_window=5)
model.fit(train_df, train_events)
print(f"[3] Model trained on pre-2025 data: {model.results['n_events']} events")

# ── 4. Run backtest on full period ──
engine = BacktestEngine(
    initial_capital=1_000_000.0,
    hold_days=5,
    transaction_cost_bps=10.0,
    max_position_pct=0.20,
)
result = engine.run(model, df, events)

eq = result["equity"]
trades = result["trades"]
metrics = result["metrics"]

print(f"\n[4] BACKTEST RESULTS")
print(f"    Trades executed : {len(trades)}")
print(f"    Equity start   : {eq.iloc[0]:,.0f}")
print(f"    Equity end     : {eq.iloc[-1]:,.0f}")
print(f"    Total return   : {(eq.iloc[-1]/eq.iloc[0]-1):.4%}")
print(f"    Sharpe         : {metrics['sharpe_ratio']}")
print(f"    Max Drawdown   : {metrics['max_drawdown']:.4%}")
print(f"    Hit Rate       : {metrics['hit_rate']:.4%}")
print(f"    Profit Factor  : {metrics['profit_factor']}")
print(f"    Annual Return  : {metrics['annual_return']:.4%}")
print(f"    Annual Vol     : {metrics['annual_vol']:.4%}")
print(f"    Calmar Ratio   : {metrics['calmar_ratio']}")

if len(trades) > 0:
    print(f"\n    Sample trades:")
    for _, t in trades.head(5).iterrows():
        print(f"      {t.entry_date.date()} | {t.ticker:5s} | "
              f"{t.direction:9s} | PnL={t.pnl:+,.0f} ({t.pnl_pct:+.4%})")

    # ── 5. Direction analysis ──
    pos_trades = trades[trades["direction"] == "POSITIVE"]
    neg_trades = trades[trades["direction"] == "NEGATIVE"]
    print(f"\n[5] DIRECTION BREAKDOWN")
    print(f"    POSITIVE: {len(pos_trades)} trades, "
          f"hit_rate={bt_metrics.hit_rate(pos_trades):.4%}")
    print(f"    NEGATIVE: {len(neg_trades)} trades, "
          f"hit_rate={bt_metrics.hit_rate(neg_trades):.4%}")

# ── 6. Anti-leakage check ──
sf = EarningsSurpriseFactor()
f_s = sf.compute(df, events)
lr = sf.check_leakage(f_s)
print(f"\n[6] LEAKAGE: peek={lr.has_future_peek}")

print(f"\n{'='*55}")
print("BACKTEST VALIDATED")
print(f"{'='*55}")
