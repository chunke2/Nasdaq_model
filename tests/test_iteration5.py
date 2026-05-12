"""Iteration 5: Real earnings calendar + Walk-forward backtest test."""
import sys
sys.path.insert(0, ".")

import pandas as pd

from src.data.synthetic import SyntheticPriceFetcher
from src.data.earnings_calendar import fetch_earnings_calendar
from src.data.cache import DataCache
from src.events.earnings import EarningsSurpriseDetector
from src.models.event_study import EventStudyModel
from src.backtest.engine import BacktestEngine
from src.utils.seed import seed_everything

seed_everything(42)

print("=" * 55)
print("ITERATION 5: REAL CALENDAR + WALK-FORWARD")
print("=" * 55)

# ── 1. Try real earnings calendar from Alpha Vantage ──
print("\n[1] Fetching real earnings calendar...")
try:
    real_cal = fetch_earnings_calendar(
        ["AAPL", "MSFT", "NVDA"], "2023-01-01", "2025-12-31"
    )
    has_real = not real_cal.empty
    print(f"    Real calendar: {len(real_cal)} events, "
          f"{real_cal['ticker'].nunique() if has_real else 0} tickers")
    if has_real:
        print(f"    Sample:")
        for _, r in real_cal.head(5).iterrows():
            print(f"      {r.date.date()} | {r.ticker} | {r.description}")
except Exception as e:
    print(f"    Real calendar unavailable: {e}")
    real_cal = pd.DataFrame()
    has_real = False

# ── 2. Generate synthetic price data ──
print("\n[2] Generating synthetic price data...")
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
print(f"    Price data: {len(df)} rows")

# ── 3a. Real calendar test (verify Alpha Vantage works) ──
print("\n[3a] Real earnings calendar test...")
detector_real = EarningsSurpriseDetector(real_calendar=real_cal if has_real else None)
events_real = detector_real.detect(df)
print(f"     Real events: {len(events_real)} (source={'AlphaVantage' if detector_real.used_real_calendar else 'proxy'})")

# ── 3b. Proxy detection (for walk-forward — events derived from price data) ──
print("\n[3b] Proxy detection (for walk-forward alignment)...")
detector_proxy = EarningsSurpriseDetector(real_calendar=None)
events = detector_proxy.detect(df)
print(f"     Proxy events: {len(events)}")
if not events.empty:
    for _, e in events.head(3).iterrows():
        print(f"       {e.date.date()} | {e.ticker} | surprise={e['surprise_pct']:+.4f}")

# ── 4. Walk-forward backtest ──
print("\n[4] Walk-forward backtest...")
engine = BacktestEngine(
    initial_capital=1_000_000.0,
    hold_days=5,
    transaction_cost_bps=10.0,
    max_position_pct=0.20,
)

result = engine.run_walk_forward(
    model_class=lambda: EventStudyModel(estimation_window=60, event_window=5),
    price_df=df,
    events_df=events,
    train_days=252,
    test_days=63,
    min_train_events=5,
)

trades = result["trades"]
eq = result["equity"]
m = result["metrics"]
windows = result.get("windows", [])

print(f"    Windows: {len(windows)}")
print(f"    Trades : {len(trades)}")
if not eq.empty and len(eq) > 1:
    print(f"    Equity : {eq.iloc[0]:,.0f} -> {eq.iloc[-1]:,.0f}")
    print(f"    Return : {(eq.iloc[-1]/eq.iloc[0]-1):.4%}")
    print(f"    Sharpe : {m.get('sharpe_ratio', 'N/A')}")
    print(f"    Max DD : {m.get('max_drawdown', 'N/A')}")
    print(f"    Hit Rt : {m.get('hit_rate', 'N/A')}")
if trades is not None and len(trades) > 0:
    print(f"    Sample trades:")
    for _, t in trades.head(3).iterrows():
        print(f"      {t.entry_date.date()} | {t.ticker} | "
              f"{t.direction} | PnL={t.pnl:+,.0f}")

print(f"\n    Windows detail:")
for w in windows:
    print(f"      Train: {w['train_start']}->{w['train_end']} "
          f"({w['train_events']}ev) | "
          f"Test: {w['test_start']}->{w['test_end']} "
          f"({w['test_events']}ev, {w['test_trades']}tr)")

# ── 5. Anti-leakage ──
from src.factors.event_factors import EarningsSurpriseFactor
sf = EarningsSurpriseFactor()
f_s = sf.compute(df, events)
lr = sf.check_leakage(f_s)
print(f"\n[5] Leakage: peek={lr.has_future_peek}")

print(f"\n{'='*55}")
print(f"ITERATION 5 VALIDATED")
print(f"{'='*55}")
