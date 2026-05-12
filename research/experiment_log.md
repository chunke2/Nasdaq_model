# Research Experiment Log

NASDAQ Event-Factor Model — chronological experiment records.

---

## [2026-05-12 12:34] Iteration 0: Phase 1 Event Study Model

**Objective**: Build end-to-end pipeline — data ingestion, event detection, factor construction, event study model.

**Changes**:
- `src/data/`: 6 data fetchers (yfinance, Alpha Vantage, Polygon, FRED, synthetic, cache)
- `src/events/earnings.py`: EarningsSurpriseDetector (proxy-based)
- `src/factors/event_factors.py`: EarningsSurpriseFactor + MomentumFactor with check_leakage()
- `src/models/event_study.py`: EventStudyModel (market-model CAR, cross-sectional t-test)
- `src/utils/`: seed, timezone, logging, experiment_logger, validation
- `config/`: settings.yaml, secrets.yaml (API keys)

**Results**:
- **Synthetic test**: 3 tickers, 1173 rows, 42 events, 73% direction accuracy
- **Real data test**: 5 tickers (AAPL/NVDA/MSFT/GOOGL/AMZN), 2055 rows, 84 events
  - Mean CAR: +0.55%, CAR t-stat: 1.33
  - Direction: POSITIVE 52% / NEGATIVE 48%

**Leakage audit**: PASSED — both factors verified, no future peek

**Next**: Real earnings calendar (not proxy), macro factors, backtest engine

---
