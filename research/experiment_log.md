# Research Experiment Log

NASDAQ Event-Factor Model — chronological experiment records.

---

## [2026-05-12 13:00] Iteration 1: Unified Data Pipeline

**Objective**: Create unified data pipeline with automatic fallback chain and batch fetching.

**Changes**:
- `src/data/pipeline.py`: DataPipeline class with fallback Priority: Polygon → Alpha Vantage → synthetic
- `src/data/pipeline.py`: `fetch_default_universe()` reads tickers from config/settings.yaml
- `src/data/pipeline.py`: `refresh_cache()` force-refreshes all tickers
- `src/data/__init__.py`: export DataPipeline
- `CLAUDE.md`: switched to fully autonomous agent policy

**Results**:
- Pipeline test: 3 tickers (AAPL/NVDA/MSFT), 1233 rows via Polygon
- Fallback chain verified: polygon > alpha_vantage > synthetic
- Import check: PASSED
- End-to-end test: PASSED (42 events, 73% direction accuracy)
- Anti-leakage: PASSED (both factors, no future peek)

**Next**: Increase ticker coverage, add daily cache refresh scheduling

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

## [2026-05-12 13:15] Iteration 3: Backtest Engine

**Objective**: Build event-driven backtest engine with full performance metrics.

**Changes**:
- `src/backtest/engine.py`: BacktestEngine — event-driven walk-forward simulation, position sizing, transaction costs, MTM, experiment logging
- `src/backtest/metrics.py`: Sharpe, MaxDD, hit rate, profit factor, Calmar, annual return/vol, beta, information ratio
- `src/backtest/__init__.py`: exports BacktestEngine + metrics
- `tests/test_backtest.py`: integration test with synthetic data

**Results**:
- 28 trades executed across 3 tickers (synthetic data, 2024-2025)
- Hit rate: 60.7%, Profit factor: 1.84
- POSITIVE direction: 60% hit rate, NEGATIVE: 61.5%
- Import check: PASSED
- End-to-end regression: PASSED (42 events, 73% accuracy unchanged)
- Backtest test: PASSED
- Anti-leakage: PASSED (no future peek)

**Next**: Real earnings calendar, macro factors, walk-forward train/test split

---
