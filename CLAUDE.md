# CLAUDE.md — NASDAQ Event-Factor Model

## Project Overview

Quantitative model analyzing NASDAQ-related events and their impact on stock returns.
Data flow: `data → events → factors → models → backtest`

## Language & Environment

- Python >= 3.10, managed via `pyproject.toml` (setuptools)
- Install: `pip install -e ".[dev]"`
- Type check: `mypy src/`
- Lint: `ruff check src/`
- Test: `pytest tests/ -v`

## Coding Standards (Mandatory)

### Type Safety
- All public functions/methods MUST have type hints
- Pydantic v2 for external data validation (API responses, DataFrame schemas)

### Defensive Programming
- Data entry points (fetchers) must validate: non-empty, columns present, time-monotonic
- Factor computation must check data completeness BEFORE calculation
- Model inputs pass through Pydantic schema validation
- Exceptions use the taxonomy: `DataError`, `ModelError`, `ConfigError` — never bare `Exception`

### Reproducibility
- `src/utils/seed.py` is the single source of truth for random seeds — every module imports it
- Notebooks must annotate: run date, data snapshot date

### Module Contracts
- Modules communicate via abstract base classes, not concrete implementations
- Data flows one direction only
- All configuration reads from `config/settings.yaml` — NEVER hardcode parameters

## Anti-Leakage Rules (CRITICAL — violations are bugs)

**Golden rule**: `factor_t → return_{t+1}` — today's factor predicts tomorrow's return.

### NEVER use these in factor construction:
- `shift(-1)` — peeks at future rows
- `pct_change().shift(-1)` — embeds future prices
- Any `rolling().aggregate()` that includes the current observation in the *prediction target*
- `center=True` in rolling windows
- `train_test_split(shuffle=True)` on time series — use `TimeSeriesSplit`

### Blocking checks (fail the build if violated):
- Earnings release date MUST be > factor calculation date
- Rolling window right edge MUST be <= current date
- Survivorship bias: use **historical** index constituents, not today's list
- Time-series train/test split (no shuffle)

### Warning checks:
- Always use adjusted close (Adj Close) from yfinance
- Event timestamps after 16:00 ET → map to NEXT trading day

### Every factor class MUST implement:
```python
def check_leakage(self, df: pd.DataFrame) -> LeakageReport:
    ...
```

## Timezone Rules

**All timestamps are `America/New_York` (EST/EDT).**
Use `src/utils/timezone.py` as the ONLY entry point — never call `.tz_localize()` or `.tz_convert()` elsewhere.

Key mapping:
- Events before/at 16:00 ET → affect SAME trading day
- Events after 16:00 ET → affect NEXT trading day
- yfinance data: tz-naive but is ET → `.tz_localize("America/New_York", ambiguous=True)`

## Research Log

Every experiment MUST auto-log to `experiments/{date}_{name}/`:
- `config.yaml` — hyperparams, data window, factor list
- `summary.json` — metrics, factor attribution, leakage check status
- `report.md` — human-readable conclusion

Triggered by: `BacktestEngine.run()`, `ModelBase.fit(log_experiment=True)`, notebook `%%run_with_log`

## MCP Servers

| Priority | Server | Purpose |
|----------|--------|---------|
| P0 | Yahoo Finance MCP | Price data, earnings dates, market cap |

## Agent Policy (Fully Autonomous)

**User reviews ONLY final CHANGELOG and milestone results. Never ask for confirmation.**

ALL operations execute autonomously — no per-change approval, no shell confirmation:
- Code creation, editing, deletion, refactoring
- Shell commands (git, pip, python, pytest, etc.)
- Data fetching, caching, API calls
- Model architecture, factors, hyperparameters, backtesting
- File reads, writes, moves
- Dependency management

**The only gate**: truly destructive actions outside the project (force-push to shared branches, deleting remote repos, etc.) — and even then, prefer to warn inline and proceed.

After every iteration: commit → update CHANGELOG → push. No questions asked.

## Git & Version Control

- **Remote**: `https://github.com/chunke2/Nasdaq_model.git`
- **Branch strategy**: commit directly to `main`
- **After every iteration**: `git add -A && git commit -m "<描述>" && git push origin main`
- **Commit style**: concise Chinese or English, describe what changed and why
- **CHANGELOG.md**: update at end of each iteration with the iteration's changes

## Changelog

Update `CHANGELOG.md` after every iteration. Format:

```
## [YYYY-MM-DD] Iteration N: Short Title

### Added
- new features, files, modules

### Changed
- modifications to existing code

### Results
- key metrics, test outputs, model performance

### Notes
- caveats, next steps, open questions
```

## Iteration Workflow — `/iterate`

Complete development cycle via a single command. Skill definition: `.claude/skills/iteration-loop.md`

```
/iterate <task description>
```

Automatically runs: **Plan → Code → Verify → Log → Commit → Changelog → Push**

- No user confirmation at any step
- Verification includes: import check + end-to-end test + anti-leakage audit
- Failures are fixed before pushing
- Research log auto-appended to `research/experiment_log.md`

## Scheduled Tasks
- Data refresh: every trading day after 17:00 ET
- Factor health check: every Monday
