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
