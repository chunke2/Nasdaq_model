---
name: iteration-loop
description: Full development iteration — plan, code, verify, log, commit, changelog, push. Invoke with /iterate.
metadata:
  type: skill
  trigger: /iterate
---

# Iteration Loop Skill

Single-command full development cycle for the NASDAQ Event-Factor Model.

## Trigger

```
/iterate
/iterate <description of what to build>
/iterate --skip-plan  (skip planning phase, go straight to code)
```

## Workflow

### 1. PLAN — Analyze and design

- Enter plan mode (use Plan agent)
- Identify which files need changes
- Check anti-leakage implications if touching factors
- Output: brief plan (3-5 lines) of what will be done

### 2. CODE — Execute changes

- Implement the planned changes
- Follow all coding standards from CLAUDE.md
- If creating new factors, implement `check_leakage()` immediately
- If adding new data sources, implement Pydantic validation immediately

### 3. VERIFY — Run tests and leakage checks

Run in this order:

```bash
# A. Import check — all modules load without error
python -c "import sys; sys.path.insert(0, '.'); import src; print('Import OK')"

# B. Run relevant test scripts
python tests/end_to_end_test.py

# C. Anti-leakage audit — run on ALL factors
python -c "
import sys; sys.path.insert(0, '.')
from src.factors.event_factors import EarningsSurpriseFactor, MomentumFactor
# (add any new factors here)
print('Leakage audit: ALL FACTORS PASSED')
"
```

If any step fails: fix it before proceeding.

### 4. LOG — Write research log

Append to `research/experiment_log.md`:

```markdown
## [YYYY-MM-DD HH:MM] <iteration title>

**Objective**: <1-line goal>

**Changes**:
- <file>: <what changed>

**Results**:
- <key metric 1>
- <key metric 2>

**Leakage audit**: PASSED / FAILED (<details>)

**Next**: <what to do next>
```

### 5. COMMIT — Conventional commit

Generate commit message following conventional commits format:

```
<type>(<scope>): <short description>

<bullet points of changes>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
Scopes: `data`, `events`, `factors`, `models`, `backtest`, `utils`, `config`, `tests`

### 6. CHANGELOG — Append to CHANGELOG.md

Add a new `## [YYYY-MM-DD] Iteration N: Title` section following the format:
- Added / Changed / Results / Notes

### 7. PUSH — Push to remote

```bash
git add -A
git commit -m "<generated message>"
git push origin main
```

## Rules

- **No user confirmation needed** at any step
- If verification fails, fix and re-verify before pushing
- ALWAYS run anti-leakage checks before committing
- NEVER skip the changelog step
- Push even if tests are imperfect — note issues in changelog
