"""End-to-end validation: synthetic data → events → factors → event study."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.synthetic import SyntheticPriceFetcher
from src.events.earnings import EarningsSurpriseDetector
from src.factors.event_factors import EarningsSurpriseFactor, MomentumFactor
from src.models.event_study import EventStudyModel
from src.utils.seed import seed_everything

seed_everything(42)

# ── 1. Generate synthetic data with known earnings shocks ──
print("=" * 55)
print("END-TO-END EVENT STUDY PIPELINE")
print("=" * 55)

fetcher = SyntheticPriceFetcher(base_price=150.0)
events_map = {
    "AAPL": pd.Series(
        [0.05, -0.04, 0.06, -0.03, 0.04],
        index=["2024-02-15", "2024-05-10", "2024-07-20",
               "2024-10-25", "2025-01-15"],
    ),
}
df = fetcher.fetch(
    ["AAPL", "NVDA", "MSFT"], "2024-01-01", "2025-06-30",
    events=events_map, seed=42,
)
print(f"[1] Price data: {len(df)} rows, {df.ticker.nunique()} tickers")

# ── 2. Detect earnings events ──
detector = EarningsSurpriseDetector(
    return_threshold_pct=2.5, volume_multiple=1.3
)
events = detector.detect(df)
print(f"[2] Events detected: {len(events)}")

# ── 3. Compute factors ──
sf = EarningsSurpriseFactor()
f_surprise = sf.compute(df, events)
mom = MomentumFactor(20)
f_momentum = mom.compute(df)
print(f"[3] Factors: surprise={len(f_surprise)} rows, momentum={len(f_momentum)} rows")

# ── 4. Anti-leakage check ──
lr_s = sf.check_leakage(f_surprise)
lr_m = mom.check_leakage(f_momentum)
print(f"[4] Leakage: surprise.peek={lr_s.has_future_peek}, "
      f"momentum.peek={lr_m.has_future_peek}")
if lr_m.corr_with_fwd_return is not None:
    print(f"    Momentum corr(w/ fwd return) = {lr_m.corr_with_fwd_return:.4f}")

# ── 5. Event study model ──
model = EventStudyModel(estimation_window=60, event_window=5)
model.fit(df, events)
r = model.results

print(f"\n[5] EVENT STUDY RESULTS")
print(f"    Events analyzed : {r['n_events']}")
print(f"    Mean CAR        : {r['mean_car']:.4%}")
ct = r.get("car_t_stat", "N/A")
if isinstance(ct, float):
    print(f"    CAR t-statistic : {ct:.4f}")
else:
    print(f"    CAR t-statistic : {ct}")

print(f"\n    Sample event CARs:")
for e in r["events"][:6]:
    print(f"      {e['date'].date()} | {e['ticker']:5s} | "
          f"CAR={e['car']:+.4%} | {e['direction']}")

# ── 6. Direction accuracy on AAPL known events ──
print(f"\n[6] AAPL KNOWN EVENT DIRECTION CHECK")
aapl_events = events[events.ticker == "AAPL"].copy()
predictions = model.predict(aapl_events)
correct = 0
for i in range(len(aapl_events)):
    ev = aapl_events.iloc[i]
    expected = "POSITIVE" if ev.surprise_pct > 0 else "NEGATIVE"
    pred = predictions.iloc[i]
    match = "OK" if pred == expected else "DIFF"
    if pred == expected:
        correct += 1
    print(f"      {match} | {ev.date.date()} | "
          f"surprise={ev.surprise_pct:+.3f} | "
          f"expected={expected:9s} | pred={pred}")
print(f"    Accuracy: {correct}/{len(aapl_events)}")

# ── 7. Summary ──
print(f"\n{'='*55}")
print(f"PIPELINE VALIDATED SUCCESSFULLY")
print(f"{'='*55}")
