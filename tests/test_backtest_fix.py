"""Walk-forward backtest with MultiFactorModel — validates real trades."""
import sys
sys.path.insert(0, ".")

import pandas as pd

from src.data.synthetic import SyntheticPriceFetcher
from src.events.earnings import EarningsSurpriseDetector
from src.factors.event_factors import EarningsSurpriseFactor, MomentumFactor, ShortTermReversalFactor
from src.models.regression import MultiFactorModel
from src.backtest.engine import BacktestEngine
from src.backtest import metrics as bt_metrics
from src.utils.seed import seed_everything

seed_everything(42)

print("=" * 55)
print("BACKTEST FIX: WALK-FORWARD WITH MULTIFACTORMODEL")
print("=" * 55)

# ── 1. Generate data with known earnings events ──
fetcher = SyntheticPriceFetcher(base_price=150.0)
events_map = {
    "AAPL": pd.Series(
        [0.05, -0.04, 0.06, -0.03, 0.04, 0.05, -0.03, 0.06],
        index=["2024-02-15", "2024-05-10", "2024-07-20",
               "2024-10-25", "2025-01-15", "2025-04-15",
               "2025-07-20", "2025-10-20"],
    ),
    "NVDA": pd.Series(
        [0.03, -0.05, 0.04, -0.03, 0.06, -0.04],
        index=["2024-03-10", "2024-06-15", "2024-09-20",
               "2025-01-10", "2025-05-15", "2025-09-10"],
    ),
    "MSFT": pd.Series(
        [0.04, -0.03, 0.05, -0.04, 0.03, -0.05],
        index=["2024-01-20", "2024-05-05", "2024-08-15",
               "2025-02-01", "2025-06-10", "2025-11-15"],
    ),
}
df = fetcher.fetch(
    ["AAPL", "NVDA", "MSFT"], "2024-01-01", "2025-12-31",
    events=events_map, seed=42,
)
print(f"[1] Data: {len(df)} rows, {df.ticker.nunique()} tickers")

# ── 2. Detect events ──
detector = EarningsSurpriseDetector(
    return_threshold_pct=3.0, volume_multiple=1.5
)
events = detector.detect(df)
print(f"[2] Events: {len(events)}")

# ── 3. Factor builder function (used by walk-forward) ──
def build_factors(prices: pd.DataFrame, evts: pd.DataFrame) -> pd.DataFrame:
    sf = EarningsSurpriseFactor(alpha=0.4)
    f_surprise = sf.compute(prices, evts)
    mom = MomentumFactor(20)
    f_momentum = mom.compute(prices)
    rev = ShortTermReversalFactor(2)
    f_reversal = rev.compute(prices)
    merged = f_surprise.merge(f_momentum, on=["date", "ticker"], how="outer")
    merged = merged.merge(f_reversal, on=["date", "ticker"], how="outer")
    return merged.fillna(0.0)

# ── 3a. Verify factors on full data ──
full_factors = build_factors(df, events)
print(f"[3] Factors: {len(full_factors)} rows, cols={list(full_factors.columns)}")

# ── 4. Walk-forward backtest ──
engine = BacktestEngine(
    initial_capital=1_000_000,
    hold_days=5,
    transaction_cost_bps=10,
    max_position_pct=0.20,
)

result = engine.run_walk_forward(
    model_class=lambda: MultiFactorModel(model_type="logistic"),
    price_df=df,
    events_df=events,
    train_days=252,
    test_days=126,
    min_train_events=5,
    factor_builder=build_factors,
)

trades = result.get("trades", pd.DataFrame())
eq = result.get("equity", pd.Series())
m = result.get("metrics", {})
windows = result.get("windows", [])

print(f"\n[4] WALK-FORWARD RESULTS")
print(f"    Windows : {len(windows)}")
print(f"    Trades  : {len(trades)}")
if len(eq) > 1:
    print(f"    Equity  : {eq.iloc[0]:,.0f} → {eq.iloc[-1]:,.0f}")
print(f"    Sharpe  : {m.get('sharpe_ratio', 'N/A'):.4f}" if isinstance(m.get('sharpe_ratio'), float) else f"    Sharpe  : {m.get('sharpe_ratio', 'N/A')}")
print(f"    Hit Rate: {m.get('hit_rate', 'N/A'):.4f}" if isinstance(m.get('hit_rate'), float) else f"    Hit Rate: {m.get('hit_rate', 'N/A')}")
print(f"    Max DD  : {m.get('max_drawdown', 'N/A'):.4f}" if isinstance(m.get('max_drawdown'), float) else f"    Max DD  : {m.get('max_drawdown', 'N/A')}")
print(f"    PnL     : {trades['pnl'].sum():,.0f}" if not trades.empty else "    PnL     : N/A")

for w in windows:
    print(f"    Win: Train={w['train_start']}→{w['train_end']} | "
          f"Test={w['test_start']}→{w['test_end']} | "
          f"Ev={w['train_events']}/{w['test_events']} | Tr={w['test_trades']}")

# ── 5. Anti-leakage check (full data) ──
sf = EarningsSurpriseFactor()
lr = sf.check_leakage(full_factors)
print(f"\n[5] LEAKAGE: peek={lr.has_future_peek}")

# ── 6. Verify core assertion ──
print(f"\n[6] ASSERTION: trades={len(trades)}")
if len(trades) > 0:
    print("    PASS: Walk-forward produced actual trades")
else:
    print("    FAIL: No trades produced — chain still broken")

print(f"\n{'='*55}")
if len(trades) > 0:
    print("BACKTEST FIX VALIDATED")
else:
    print("BACKTEST FIX FAILED")
print(f"{'='*55}")
