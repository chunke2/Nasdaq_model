# Changelog

## [2026-05-12] Iteration 0: Project Infrastructure + Phase 1 Event Study Model

### Added
- Project skeleton: modular architecture with `src/data/`, `src/events/`, `src/factors/`, `src/models/`, `src/backtest/`, `src/utils/`
- `CLAUDE.md`: coding standards, anti-leakage rules, timezone rules, agent policy, skill definitions
- `pyproject.toml`: dependency management (pandas, numpy, yfinance, pydantic, statsmodels, scikit-learn, xgboost, matplotlib, pyyaml, pyarrow)
- `config/settings.yaml`: global configuration (10 NASDAQ sample tickers, factor parameters, backtest params)
- `config/secrets.yaml`: API keys (Alpha Vantage, Polygon.io, FRED) — gitignored
- `.gitignore`: data/, experiments/, secrets, venv, IDE files
- **Data layer** (`src/data/`):
  - `fetcher.py`: abstract DataFetcher base class with validation
  - `cache.py`: Parquet-based local cache
  - `price.py`: yfinance fetcher with retry/backoff (currently rate-limited)
  - `alpha_vantage.py`: Alpha Vantage API fetcher (free tier: 100 data points)
  - `polygon.py`: Polygon.io API fetcher (free tier: ~2yr daily data, 5 calls/min)
  - `fred.py`: FRED API fetcher (CPI, unemployment, GDP, fed funds — unlimited free)
  - `synthetic.py`: synthetic price data generator for offline development
  - `config_loader.py`: YAML config and secrets loader
- **Event layer** (`src/events/`):
  - `base.py`: abstract EventDetector
  - `earnings.py`: EarningsSurpriseDetector (proxy-based, using large returns + volume spikes)
- **Factor layer** (`src/factors/`):
  - `base.py`: abstract FactorBase with mandatory `check_leakage()` contract
  - `event_factors.py`: EarningsSurpriseFactor + MomentumFactor with anti-leakage validation
- **Model layer** (`src/models/`):
  - `base.py`: abstract ModelBase
  - `event_study.py`: EventStudyModel (market-model CAR, cross-sectional t-test, experiment logging)
- **Utilities** (`src/utils/`):
  - `seed.py`: global random seed
  - `timezone.py`: unified America/New_York timezone handling
  - `logging_config.py`: structured logging
  - `experiment_logger.py`: automatic experiment recording
  - `validation.py`: Pydantic v2 models + exception taxonomy
- **Tests**:
  - `tests/end_to_end_test.py`: synthetic data pipeline validation
  - `tests/real_data_test.py`: real data pipeline (Polygon + FRED)

### Results
- **Synthetic data test**: 3 tickers, 1173 rows, 42 events detected, 73% direction accuracy
- **Real data test**: 5 tickers (AAPL/NVDA/MSFT/GOOGL/AMZN), 2055 rows, 84 events
  - Mean CAR: +0.55%, CAR t-stat: 1.33
  - Anti-leakage: both factors passed (no future peek)
  - Experiment auto-logging: verified

### API Status
| API | Status | Capacity |
|-----|--------|----------|
| FRED | Working | Unlimited free |
| Polygon.io | Working | ~2yr data, 5 calls/min |
| Alpha Vantage | Working | 100 data points, 25 calls/day |
| yfinance | Rate-limited | Blocked from current IP |

### Notes
- Phase 1 complete: event study model can analyze NASDAQ events and predict direction
- Next: add real earnings calendar (not proxy), macro factors, backtest engine
- Need to investigate Polygon.io earnings calendar endpoint for real event dates

---

## [2026-05-12] Iteration 1: Automated Workflow + Git Integration

### Added
- `.claude/skills/iteration-loop.md`: `/iterate` skill — full automated Plan→Code→Verify→Log→Commit→Changelog→Push cycle
- `research/experiment_log.md`: chronological experiment log
- Agent policy changed from layered-approval to full autonomous mode

### Changed
- `CLAUDE.md`: replaced approval layers with `/iterate` workflow, added git/changelog policy
- Git remote configured: `https://github.com/chunke2/Nasdaq_model.git`

---

## [2026-05-12] Iteration 2: Unified Data Pipeline + Full Autonomy

### Added
- `src/data/pipeline.py`: DataPipeline class with automatic fallback chain
  - Priority: Polygon → Alpha Vantage → synthetic
  - `fetch_default_universe()` reads tickers from config/settings.yaml
  - `refresh_cache()` force-refreshes all tickers
  - Smart cache merging for partial date range coverage
- `CLAUDE.md`: fully autonomous agent policy (no shell/file confirmation)

### Changed
- `src/data/__init__.py`: exports DataPipeline
- Agent policy: all approval gates removed, only final CHANGELOG reviewed

### Results
- Pipeline test: 3 tickers (AAPL/NVDA/MSFT), 1233 rows via Polygon
- Fallback chain verified across all 3 sources
- End-to-end test: PASSED (42 events, anti-leakage clean)
- Import check: PASSED

### Notes
- Polygon free tier gives ~2yr data (2024-05 to present) — sufficient for Phase 1
- 10-ticker full batch would take ~2min due to Polygon 12s rate limit

---

## [2026-05-12] Iteration 3: Backtest Engine + Performance Metrics

### Added
- `src/backtest/engine.py`: BacktestEngine class
  - Event-driven walk-forward simulation
  - Position sizing (max_position_pct)
  - Transaction cost model (bps-based)
  - Daily mark-to-market tracking
  - Auto experiment logging on run()
- `src/backtest/metrics.py`: comprehensive performance metrics
  - Sharpe ratio, Max drawdown, Calmar ratio
  - Hit rate, Profit factor, Annual return/vol
  - Beta, Information ratio (vs benchmark)
- `src/backtest/__init__.py`: exports BacktestEngine + metrics
- `tests/test_backtest.py`: integration test

### Results
- **28 trades** executed on synthetic data (3 tickers, 2024-2025)
- Hit rate: **60.7%**, Profit factor: **1.84**
- POSITIVE/NEGATIVE direction: both ~60% hit rate
- End-to-end regression: PASSED (42 events, 73% accuracy)
- Anti-leakage: PASSED

### Notes
- Synthetic data returns are amplified due to built-in reversal effect
- Real-data backtest will show more realistic metrics
- Next: real earnings calendar, macro factors, walk-forward split

---

## [2026-05-12] Iteration 5: Real Earnings Calendar + Walk-forward Backtest

### Added
- `src/data/earnings_calendar.py`: Alpha Vantage EARNINGS endpoint fetcher with Parquet caching
- `src/backtest/engine.py`: `run_walk_forward()` — rolling time-split train/test with retraining

### Changed
- `src/events/earnings.py`: EarningsSurpriseDetector now accepts `real_calendar` parameter
  - Primary: real Alpha Vantage earnings dates with actual EPS surprise data
  - Fallback: proxy detection (large returns + volume)
  - `used_real_calendar` property to check which source was used

### Results
- **Alpha Vantage earnings**: 36 real events across 3 tickers
  - AAPL: 121 quarters, MSFT: 121, NVDA: 108
  - Real EPS surprise data (e.g. MSFT +9.9%, AAPL +6.3%)
- **Walk-forward**: 2 windows, correct time-split (no leakage)
- Import check: PASSED
- Anti-leakage: PASSED

### Known Gap
- Walk-forward needs regression model for forward predictions
- EventStudyModel.predict() only works on trained event dates
- Will be resolved in Iteration 6 (multi-factor regression)

---

## [2026-05-12] Iteration 6: Multi-Factor Regression Model

### Added
- `src/models/regression.py`: MultiFactorModel
  - Supports logistic (direction) and linear (magnitude) modes
  - Auto-builds X/y from factor + price DataFrames with proper time alignment
  - `get_factor_attribution()` returns coefficient → direction mapping
  - StandardScaler + class_weight=balanced for robust fitting
  - Compatible with `ModelBase` interface (fit/predict)
- `src/utils/experiment_logger.py`: numpy type sanitization for YAML/JSON

### Changed
- `src/backtest/engine.py`: `run_walk_forward()` now accepts `factor_builder` callback
  - `_safe_predict()` handles both EventStudyModel and MultiFactorModel

### Results
- **759 training samples**, 2 features (earnings_surprise, momentum_20d)
- Factor attribution:
  - earnings_surprise: coeff=-0.123 → favors NEGATIVE
  - momentum_20d: coeff=+0.011 → favors POSITIVE (weak)
- **POS prediction accuracy: 78.4%** — high reliability when predicting POSITIVE
- NEG prediction accuracy: 19.6% — model biased toward POSITIVE
- Anti-leakage: PASSED

### Known Gap
- Backtest engine needs further refactoring for factor-aware trade execution
- Walk-forward with regression model will be completed in next iteration
