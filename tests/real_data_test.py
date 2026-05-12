"""Real-data pipeline test using Polygon + FRED."""
import sys
sys.path.insert(0, ".")

from src.data.polygon import PolygonFetcher
from src.data.fred import FREDFetcher
from src.events.earnings import EarningsSurpriseDetector
from src.models.event_study import EventStudyModel
from src.utils.seed import seed_everything

seed_everything(42)

# ── 1. Real NASDAQ price data from Polygon ──
print("Fetching from Polygon.io...")
pf = PolygonFetcher()
df = pf.fetch(
    ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN"],
    "2023-01-01", "2025-12-31",
)
print(f"[1] Price: {len(df)} rows, {df.ticker.nunique()} tickers")
print(f"    Date range: {df.index[0].date()} to {df.index[-1].date()}")

# ── 2. FRED macro data ──
print("Fetching from FRED...")
fred = FREDFetcher()
macro = fred.fetch(["cpi", "unemployment"], "2023-01-01", "2025-12-31")
print(f"[2] Macro: {len(macro)} obs, columns={list(macro.columns)}")

# ── 3. Detect earnings events ──
print("Detecting earnings proxy events...")
detector = EarningsSurpriseDetector(
    return_threshold_pct=3.0, volume_multiple=1.5
)
events = detector.detect(df)
print(f"[3] Events: {len(events)}")
for _, e in events.head(10).iterrows():
    print(f"    {e.date.date()} | {e.ticker:5s} | "
          f"surprise={e.surprise_pct:+.3f}")

# ── 4. Event study ──
print("Running event study...")
model = EventStudyModel(estimation_window=60, event_window=5)
model.fit(df, events)
r = model.results

print(f"\n[4] EVENT STUDY (REAL DATA)")
print(f"    Events analyzed : {r['n_events']}")
print(f"    Mean CAR        : {r['mean_car']:.4%}")
ct = r.get("car_t_stat", "N/A")
if isinstance(ct, float):
    print(f"    CAR t-statistic : {ct:.4f}")
else:
    print(f"    CAR t-statistic : {ct}")

print(f"\n    Top events by |CAR|:")
sorted_events = sorted(r["events"], key=lambda e: abs(e["car"]), reverse=True)
for e in sorted_events[:8]:
    print(f"      {e['date'].date()} | {e['ticker']:5s} | "
          f"CAR={e['car']:+.4%} | {e['direction']}")

# ── 5. Direction distribution ──
pos = sum(1 for e in r["events"] if e["direction"] == "POSITIVE")
neg = sum(1 for e in r["events"] if e["direction"] == "NEGATIVE")
print(f"\n[5] Direction distribution: POSITIVE={pos}, NEGATIVE={neg} "
      f"({pos/len(r['events']):.0%} / {neg/len(r['events']):.0%})")

print(f"\nREAL-DATA PIPELINE VALIDATED")
