from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from minigrid_solver.domain import EpisodeLog


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _to_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_episode_logs(logs: Sequence[EpisodeLog], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, log in enumerate(logs, start=1):
        with (output_dir / f"episode_{idx:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump(_to_json_safe(log.to_dict()), handle, indent=2)


def save_summary(summary: Dict[str, object], output_dir: Path, filename: str = "summary.json") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(summary), handle, indent=2)
    return path


def write_plot_data(results: Dict[str, Dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(results), handle, indent=2)
    return path
