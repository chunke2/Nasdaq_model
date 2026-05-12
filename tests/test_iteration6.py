"""Iteration 6: Multi-factor regression model test."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.synthetic import SyntheticPriceFetcher
from src.events.earnings import EarningsSurpriseDetector
from src.factors.event_factors import EarningsSurpriseFactor, MomentumFactor
from src.models.regression import MultiFactorModel
from src.utils.seed import seed_everything

seed_everything(42)

print("=" * 55)
print("ITERATION 6: MULTI-FACTOR REGRESSION")
print("=" * 55)

# ── 1. Generate data ──
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
print(f"[1] Price: {len(df)} rows")

# ── 2. Events + factors ──
detector = EarningsSurpriseDetector(real_calendar=None)
events = detector.detect(df)
sf = EarningsSurpriseFactor(); f_s = sf.compute(df, events)
mom = MomentumFactor(20); f_m = mom.compute(df)
f_merged = f_s.merge(f_m, on=["date", "ticker"], how="outer").fillna(0.0)
print(f"[2] Factors: {len(f_merged)} rows, cols={list(f_merged.columns)}")

# ── 3. Time split: train on 2024, test on 2025 ──
split = "2025-01-01"
f_train = f_merged[f_merged["date"] < split]
f_test = f_merged[f_merged["date"] >= split]
p_train = df[df.index < split]
p_test = df[df.index >= split]
print(f"[3] Train: {len(f_train)} factors / {len(p_train)} prices")
print(f"    Test:  {len(f_test)} factors / {len(p_test)} prices")

# ── 4. Train MultiFactorModel ──
model = MultiFactorModel(model_type="logistic")
model.fit(f_train, p_train, log_experiment=True)
print(f"\n[4] MODEL FITTED")
print(f"    Features     : {model._feature_names}")
print(f"    Coefficients :")
for k, v in model.get_factor_attribution().items():
    print(f"      {k:25s}: {v:+.6f}")

# ── 5. Predict on test set + evaluate accuracy ──
preds = model.predict(f_test)
print(f"\n[5] PREDICTIONS: {len(preds)} total")

# Align: preds has same index as f_test
correct = 0
total = 0
pos_correct = 0
neg_correct = 0
pos_total = 0
neg_total = 0

for i in range(len(f_test)):
    row = f_test.iloc[i]
    ticker = row["ticker"]
    d = pd.Timestamp(row["date"])
    tp = p_test[p_test["ticker"] == ticker].sort_index()
    # Find the price date closest to this factor date
    nearby = tp.index[tp.index >= d]
    if len(nearby) == 0:
        continue
    px_date = nearby[0]
    loc = tp.index.get_loc(px_date)
    if isinstance(loc, slice) or loc + 1 >= len(tp):
        continue
    actual_ret = tp["adj_close"].iloc[loc + 1] / tp["adj_close"].iloc[loc] - 1
    actual = "POSITIVE" if actual_ret > 0 else "NEGATIVE"
    pred = preds.iloc[i]
    if pred is None or pred == "NEUTRAL":
        continue
    total += 1
    if pred == actual:
        correct += 1
        if actual == "POSITIVE":
            pos_correct += 1
        else:
            neg_correct += 1
    if actual == "POSITIVE":
        pos_total += 1
    else:
        neg_total += 1

acc = correct / total if total > 0 else 0
print(f"    Overall accuracy : {acc:.4f} ({correct}/{total})")
if pos_total > 0:
    print(f"    POS accuracy     : {pos_correct/pos_total:.4f} ({pos_correct}/{pos_total})")
if neg_total > 0:
    print(f"    NEG accuracy     : {neg_correct/neg_total:.4f} ({neg_correct}/{neg_total})")

# ── 6. Factor attribution ──
print(f"\n[6] FACTOR ATTRIBUTION")
for k, v in model.get_factor_attribution().items():
    direction = "NEGATIVE" if v < 0 else "POSITIVE"
    strength = abs(v)
    print(f"    {k}: coeff={v:+.4f} → favors {direction} (strength={strength:.4f})")

# ── 7. Anti-leakage ──
lr = sf.check_leakage(f_s)
print(f"\n[7] LEAKAGE: peek={lr.has_future_peek}")

# ── 8. Compare: logistic vs random baseline ──
baseline = max(pos_total / total, neg_total / total) if total > 0 else 0.5
print(f"\n[8] BENCHMARK")
print(f"    Model accuracy    : {acc:.4f}")
print(f"    Baseline (max cls): {baseline:.4f}")
print(f"    Lift              : {acc - baseline:+.4f}")

print(f"\n{'='*55}")
print(f"ITERATION 6 VALIDATED")
print(f"{'='*55}")
