from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSET_ROOT = ROOT / "data" / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Grid-Universe agent across authored levels."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Seeds to evaluate for each level and observation mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        help="Directory for JSON/CSV summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = None

    if InconsistentVersionWarning is not None:
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    from final import Agent
    from gameplay_levels import TURN_LIMIT
    import gameplay_levels as gameplay_levels_module
    from grid_universe.gym_env import GridUniverseEnv

    builders = [
        getattr(gameplay_levels_module, name)
        for name in dir(gameplay_levels_module)
        if name.startswith("build_level_") and callable(getattr(gameplay_levels_module, name))
    ]
    builders.sort(key=lambda fn: fn.__name__)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_types = ["level", "image"]

    rows: list[dict[str, object]] = []

    for observation_type in observation_types:
        for builder in builders:
            for seed in args.seeds:
                sample_state = builder(seed=seed)

                def initial_state_fn(
                    *_args,
                    builder=builder,
                    seed=seed,
                    **_kwargs,
                ):
                    state = builder(seed=seed)
                    return replace(state, seed=seed, turn_limit=TURN_LIMIT)

                env = GridUniverseEnv(
                    initial_state_fn=initial_state_fn,
                    width=sample_state.width,
                    height=sample_state.height,
                    observation_type=observation_type,
                    render_asset_root=str(ASSET_ROOT),
                )

                started = time.perf_counter()
                state, _ = env.reset()
                agent = Agent()
                total_reward = 0.0
                win = False
                lose = False
                steps = 0
                error = ""

                try:
                    while not (win or lose):
                        action = agent.step(state)
                        state, reward, win, lose, _ = env.step(action)
                        total_reward += float(reward)
                        steps += 1
                        if steps > 500:
                            error = "step_limit_guard_triggered"
                            break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

                elapsed = time.perf_counter() - started
                env.close()

                rows.append(
                    {
                        "level_name": builder.__name__,
                        "observation_type": observation_type,
                        "seed": seed,
                        "win": bool(win),
                        "lose": bool(lose),
                        "success": bool(win and not error),
                        "steps": steps,
                        "total_reward": total_reward,
                        "elapsed_sec": elapsed,
                        "error": error,
                    }
                )
                print(
                    builder.__name__,
                    observation_type,
                    seed,
                    f"success={bool(win and not error)}",
                    f"reward={total_reward}",
                    f"steps={steps}",
                    f"error={error or 'none'}",
                )

    summary_rows: list[dict[str, object]] = []
    for observation_type in observation_types:
        subset = [row for row in rows if row["observation_type"] == observation_type]
        if not subset:
            continue
        success_rate = sum(1 for row in subset if row["success"]) / len(subset)
        avg_reward = sum(float(row["total_reward"]) for row in subset) / len(subset)
        avg_steps = sum(int(row["steps"]) for row in subset) / len(subset)
        avg_elapsed = sum(float(row["elapsed_sec"]) for row in subset) / len(subset)
        summary_rows.append(
            {
                "scope": "overall",
                "level_name": "ALL_LEVELS",
                "observation_type": observation_type,
                "episodes": len(subset),
                "success_rate": success_rate,
                "average_reward": avg_reward,
                "average_steps": avg_steps,
                "average_elapsed_sec": avg_elapsed,
            }
        )

        for level_name in sorted({str(row["level_name"]) for row in subset}):
            level_subset = [row for row in subset if row["level_name"] == level_name]
            summary_rows.append(
                {
                    "scope": "per_level",
                    "level_name": level_name,
                    "observation_type": observation_type,
                    "episodes": len(level_subset),
                    "success_rate": sum(1 for row in level_subset if row["success"]) / len(level_subset),
                    "average_reward": sum(float(row["total_reward"]) for row in level_subset) / len(level_subset),
                    "average_steps": sum(int(row["steps"]) for row in level_subset) / len(level_subset),
                    "average_elapsed_sec": sum(float(row["elapsed_sec"]) for row in level_subset) / len(level_subset),
                }
            )

    (output_dir / "grid_universe_episode_results.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "grid_universe_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "grid_universe_episode_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "grid_universe_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
