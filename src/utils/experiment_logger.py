"""Automatic experiment recording.

Every model training run and backtest produces a structured experiment
directory under experiments/. This module provides the programmatic API.
See config/settings.yaml for the ExperimentLogger configuration.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sanitize(obj: Any) -> Any:
    """Convert numpy types to native Python for YAML/JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    return obj

EXPERIMENTS_ROOT: Path = Path("experiments")


class ExperimentLogger:
    """Structured experiment recorder.

    Usage:
        logger = ExperimentLogger()
        exp_id = logger.start("earnings_baseline", config_dict)
        logger.log_metrics({"sharpe_ratio": 1.2, ...})
        logger.log_factor_attribution({"earnings_surprise": {...}})
        logger.finalize(report_md="...", artifacts=[...])
    """

    def __init__(self, root: Path = EXPERIMENTS_ROOT) -> None:
        self._root = Path(root)
        self._exp_dir: Path | None = None
        self._summary: dict[str, Any] = {}

    def start(self, name: str, config: dict[str, Any]) -> str:
        """Initialize a new experiment directory.

        Args:
            name: Short snake_case description, e.g. "earnings_baseline".
            config: Full experiment configuration to save as config.yaml.

        Returns:
            experiment_id string (e.g. "2026-05-12_earnings_baseline").
        """
        today = datetime.now().strftime("%Y-%m-%d")
        exp_id = f"{today}_{name}"
        self._exp_dir = self._root / exp_id
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        (self._exp_dir / "artifacts").mkdir(exist_ok=True)

        # Save config (sanitize numpy types for YAML)
        config_clean = _sanitize(config)
        with open(self._exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_clean, f, default_flow_style=False, allow_unicode=True)

        # Initialize summary
        self._summary = {
            "experiment_id": exp_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "git_commit": _get_git_commit(),
            "data_window": config.get("data_window", {}),
            "factors_used": config.get("factors_used", []),
            "model_type": config.get("model_type", "unknown"),
            "hyperparameters": config.get("hyperparameters", {}),
            "train_period": config.get("train_period", ""),
            "test_period": config.get("test_period", ""),
            "metrics": {},
            "factor_attribution": {},
            "leakage_checks_passed": config.get("leakage_checks_passed", None),
            "data_snapshot_date": today,
            "notes": config.get("notes", ""),
        }

        return exp_id

    def log_metrics(self, metrics: dict[str, float | None]) -> None:
        """Record performance metrics."""
        self._summary["metrics"] = metrics
        self._flush_summary()

    def log_factor_attribution(
        self, attribution: dict[str, dict[str, float | None]]
    ) -> None:
        """Record factor-level attribution (coefficient, t_stat, p_value)."""
        self._summary["factor_attribution"] = attribution
        self._flush_summary()

    def finalize(
        self,
        report_md: str,
        artifacts: list[Path] | None = None,
    ) -> Path:
        """Write final report.md and return the experiment directory path.

        Args:
            report_md: Markdown content for report.md.
            artifacts: Optional list of artifact file paths to copy into
                       the artifacts/ subdirectory.
        """
        if self._exp_dir is None:
            raise RuntimeError("ExperimentLogger.start() must be called first")

        # Write report
        with open(self._exp_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        # Copy artifacts
        if artifacts:
            import shutil

            for src in artifacts:
                src_path = Path(src)
                if src_path.exists():
                    dst = self._exp_dir / "artifacts" / src_path.name
                    shutil.copy2(src_path, dst)

        self._flush_summary()
        return self._exp_dir

    def _flush_summary(self) -> None:
        """Write current summary to summary.json."""
        if self._exp_dir is None:
            return
        with open(self._exp_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(self._summary, f, indent=2, default=str, ensure_ascii=False)


def _get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
