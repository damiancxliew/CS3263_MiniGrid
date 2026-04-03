from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved comparison metrics.")
    parser.add_argument("--comparison-json", type=str, default="logs/doorkey_baseline/comparison.json")
    parser.add_argument("--output", type=str, default="logs/doorkey_baseline/comparison.png")
    args = parser.parse_args()

    comparison_path = Path(args.comparison_json)
    with comparison_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    agents = list(results.keys())
    metrics = ["success_rate", "average_reward", "average_episode_length", "average_planner_expansions"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for axis, metric in zip(axes, metrics):
        values = [results[agent].get(metric, 0.0) for agent in agents]
        axis.bar(agents, values, color=["#1f77b4", "#ff7f0e"][: len(agents)])
        axis.set_title(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
