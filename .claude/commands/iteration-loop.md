#!/usr/bin/env claude

# /iterate <task description>
# Full autonomous development iteration: Plan → Code → Verify → Log → Commit → Changelog → Push

## Workflow

Execute all 7 steps without asking for confirmation:

### 1. PLAN
- Analyze the task, identify files to change
- Check anti-leakage implications if touching factors
- Output a 3-5 line plan, then proceed immediately

### 2. CODE
- Implement changes following CLAUDE.md standards
- New factors MUST include check_leakage() immediately
- New data sources MUST include Pydantic validation immediately

### 3. VERIFY — run ALL three checks:
```bash
python -c "import sys; sys.path.insert(0, '.'); import src; print('Import OK')"
python tests/end_to_end_test.py
```

### 4. LOG — append to research/experiment_log.md:
```markdown
## [YYYY-MM-DD HH:MM] <title>
**Objective**: <1-line>
**Changes**: <file list>
**Results**: <metrics>
**Leakage audit**: PASSED / FAILED
**Next**: <next step>
```

### 5. COMMIT — conventional commits format:
```
<type>(<scope>): <description>
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```
Types: feat, fix, refactor, test, docs, chore
Scopes: data, events, factors, models, backtest, utils, config, tests

### 6. CHANGELOG — append new ## [date] Iteration N: Title section to CHANGELOG.md

### 7. PUSH — git add -A && git commit && git push origin main

## Rules
- NO user confirmation at any step
- Fix failures before pushing
- ALWAYS anti-leakage check before commit
- NEVER skip changelog
