"""Diagnostic: trace factor→target signal chain in synthetic data.

Hypothesis: The -0.3 * shock reversal at t+1 in synthetic data is misaligned
with the factor shift(1), producing factor_t that predicts return_{t+2} when
the reversal already happened at t+1. The signal leaks out before the factor
can capture it.

Fix: Move reversal to t+2 so factor_{t+1}(=shock_t) predicts return_{t+2}.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.synthetic import SyntheticPriceFetcher
from src.events.earnings import EarningsSurpriseDetector
from src.factors.event_factors import EarningsSurpriseFactor, MomentumFactor, ShortTermReversalFactor
from src.models.regression import MultiFactorModel
from src.utils.seed import seed_everything

seed_everything(42)

print("=" * 60)
print("DIAGNOSTIC: FACTOR→TARGET SIGNAL CHAIN")
print("=" * 60)

# ── 1. Generate data with a SINGLE known event ──
print("\n── 1. Single event test ──")
fetcher = SyntheticPriceFetcher(base_price=150.0)
events_map = {
    "AAPL": pd.Series([0.10], index=["2024-06-14"]),  # big +10% shock (Friday)
}
df = fetcher.fetch(["AAPL"], "2024-01-01", "2025-12-31", events=events_map, seed=42)
print(f"  Data: {len(df)} rows, index={df.index[0].date()} to {df.index[-1].date()}")

# Find the event date and surrounding returns
event_idx = df.index.get_loc(pd.Timestamp("2024-06-14", tz="America/New_York"))
print(f"  Event at 2024-06-14 is row index {event_idx}")

# Show returns around the event
for offset in range(-3, 6):
    idx = event_idx + offset
    if 0 <= idx < len(df):
        row = df.iloc[idx]
        if idx > 0 and idx < len(df) - 1:
            ret_t = (row["adj_close"] - df.iloc[idx-1]["adj_close"]) / df.iloc[idx-1]["adj_close"]
            ret_t1 = (df.iloc[idx+1]["adj_close"] - row["adj_close"]) / row["adj_close"]
        else:
            ret_t = float('nan')
            ret_t1 = float('nan')
        print(f"  row {idx} | {row.name.date()} | "
              f"price={row['adj_close']:.2f} | "
              f"ret_t={ret_t:+.4%} | ret_t+1={ret_t1:+.4%}")

# ── 2. Detect events ──
detector = EarningsSurpriseDetector(return_threshold_pct=3.0, volume_multiple=1.5)
events = detector.detect(df)
print(f"\n── 2. Events detected: {len(events)}")
for _, e in events.iterrows():
    print(f"  {e['date'].date()} | surprise={e['surprise_pct']:+.4f}")

# ── 3. Factor values around event ──
sf = EarningsSurpriseFactor(alpha=0.4)
f_surprise = sf.compute(df, events)
mom = MomentumFactor(20)
f_momentum = mom.compute(df)
rev = ShortTermReversalFactor(2)
f_reversal = rev.compute(df)
f_merged = f_surprise.merge(f_momentum, on=["date", "ticker"], how="outer")
f_merged = f_merged.merge(f_reversal, on=["date", "ticker"], how="outer").fillna(0.0)

print(f"\n── 3. Factor values around event ──")
for offset in range(-3, 6):
    idx = event_idx + offset
    if 0 <= idx < len(df):
        d = df.index[idx]
        f_row = f_merged[(f_merged["date"] == d) & (f_merged["ticker"] == "AAPL")]
        if not f_row.empty:
            es = f_row["earnings_surprise"].values[0]
            mom_val = f_row["momentum_20d"].values[0]
            rev_val = f_row["reversal_2d"].values[0]
            print(f"  {d.date()} | es={es:+.6f} | mom={mom_val:+.6f} | rev={rev_val:+.6f}")

# ── 4. Compute factor→target correlation manually ──
print(f"\n── 4. Manual factor→target alignment ──")
p_sub = df[df["ticker"] == "AAPL"].sort_index()
returns = p_sub["adj_close"].pct_change()
fwd_returns = returns.shift(-1)  # return from t to t+1

# Align: factor at t, target = return t→t+1
f_aapl = f_merged[f_merged["ticker"] == "AAPL"].set_index("date").sort_index()
common = f_aapl.index.intersection(fwd_returns.index)

if len(common) > 1:
    es_aligned = f_aapl.loc[common, "earnings_surprise"]
    mom_aligned = f_aapl.loc[common, "momentum_20d"]
    rev_aligned = f_aapl.loc[common, "reversal_2d"]
    fwd_aligned = fwd_returns.loc[common]

    es_corr = es_aligned.corr(fwd_aligned)
    mom_corr = mom_aligned.corr(fwd_aligned)
    rev_corr = rev_aligned.corr(fwd_aligned)

    print("  corr(earnings_surprise_t, return_t+1) = {:+.6f}".format(es_corr))
    print("  corr(momentum_20d_t, return_t+1) = {:+.6f}".format(mom_corr))
    print("  corr(reversal_2d_t, return_t+1) = {:+.6f}".format(rev_corr))

# ── 5. Full model test ──
print(f"\n── 5. Full MultiFactorModel test ──")
model = MultiFactorModel(model_type="logistic")
model.fit(f_merged, df)
coefs = model.get_factor_attribution()
for k, v in coefs.items():
    print(f"  {k}: coeff={v:+.6f}")

# Predict and check direction distribution
preds = model.predict(f_merged)
n_pos = (preds == "POSITIVE").sum()
n_neg = (preds == "NEGATIVE").sum()
print(f"  Predictions: POS={n_pos}, NEG={n_neg}, ratio={n_pos/(n_pos+n_neg):.1%}")

# Build XY manually to check target distribution
X, y = model._build_xy(f_merged, df)
print(f"  Training samples: {len(X)}")
print(f"  Target distribution: POS={(y=='POSITIVE').sum()}, NEG={(y=='NEGATIVE').sum()}")
print(f"  POS ratio: {(y=='POSITIVE').mean():.1%}")

# Feature importance: for each feature, show correlation with target
for col in X.columns:
    binary_y = (y == "POSITIVE").astype(float)
    corr_with_target = X[col].corr(binary_y)
    print(f"  corr({col}, target_direction) = {corr_with_target:+.6f}")

print(f"\n{'='*60}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*60}")
