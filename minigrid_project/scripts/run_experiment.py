from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minigrid_solver.agents import ExplainableHybridAgent, QLearningAgent
from minigrid_solver.envs import make_env
from minigrid_solver.utils import evaluate_agent, save_episode_logs, save_summary, summarize_logs, write_plot_data


def build_agent(agent_name: str, args: argparse.Namespace):
    if agent_name == "hybrid":
        return ExplainableHybridAgent()
    if agent_name == "qlearning":
        return QLearningAgent(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            revisit_penalty=args.revisit_penalty,
            key_bonus=args.key_bonus,
            door_bonus=args.door_bonus,
            distance_bonus=args.distance_bonus,
        )
    raise ValueError(f"Unsupported agent '{agent_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniGrid DoorKey experiments.")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--agent", choices=["hybrid", "qlearning", "compare"], default="hybrid")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-eval-only", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results/logs/doorkey_baseline")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--revisit-penalty", type=float, default=0.02)
    parser.add_argument("--key-bonus", type=float, default=0.3)
    parser.add_argument("--door-bonus", type=float, default=0.5)
    parser.add_argument("--distance-bonus", type=float, default=0.05)
    args = parser.parse_args()

    output_root = ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}
    agents_to_run: List[str] = ["hybrid", "qlearning"] if args.agent == "compare" else [args.agent]

    for agent_name in agents_to_run:
        agent = build_agent(agent_name, args)
        agent_output = output_root / agent_name
        video_root = agent_output / "videos"

        def env_factory(record_video: bool = False, video_folder: Path | None = None, video_prefix: str | None = None):
            return make_env(
                args.env,
                render=args.render and not record_video,
                record_video=record_video,
                video_folder=str(video_folder or video_root),
                video_prefix=video_prefix or f"{agent_name}-{args.env}",
            )

        if agent_name == "qlearning":
            training_seeds = [args.seed + idx for idx in range(args.train_episodes)]
            training_logs = evaluate_agent(
                agent=agent,
                env_factory=lambda: env_factory(record_video=False),
                seeds=training_seeds,
                training=True,
                max_steps=args.max_steps,
                render=False,
            )
            training_summary = summarize_logs(training_logs)
            training_summary["episodes"] = args.train_episodes
            save_summary(training_summary, agent_output, filename="training_summary.json")
            if getattr(agent, "training_history", None):
                with (agent_output / "training_curve.json").open("w", encoding="utf-8") as handle:
                    json.dump(agent.training_history, handle, indent=2)

        eval_seeds = [args.seed + 10_000 + idx for idx in range(args.episodes)]
        record_count = min(args.video_episodes, len(eval_seeds)) if args.record_video else 0
        standard_eval_seeds = eval_seeds[record_count:] if args.record_eval_only else []
        if not args.record_eval_only:
            standard_eval_seeds = eval_seeds
            record_count = min(args.video_episodes, len(eval_seeds)) if args.record_video else 0

        logs = []
        if args.record_video and record_count > 0:
            for episode_index, seed in enumerate(eval_seeds[:record_count], start=1):
                episode_video_dir = video_root / f"episode_{episode_index:03d}"
                recorded_logs = evaluate_agent(
                    agent=agent,
                    env_factory=lambda episode_video_dir=episode_video_dir, episode_index=episode_index: env_factory(
                        record_video=True,
                        video_folder=episode_video_dir,
                        video_prefix=f"{agent_name}-{args.env}-eval-{episode_index:03d}",
                    ),
                    seeds=[seed],
                    training=False,
                    max_steps=args.max_steps,
                    render=False,
                )
                logs.extend(recorded_logs)

        remaining_seeds = standard_eval_seeds
        if remaining_seeds:
            logs.extend(
                evaluate_agent(
                    agent=agent,
                    env_factory=lambda: env_factory(record_video=False),
                    seeds=remaining_seeds,
                    training=False,
                    max_steps=args.max_steps,
                    render=args.render,
                )
            )

        if not logs:
            logs = evaluate_agent(
                agent=agent,
                env_factory=lambda: env_factory(record_video=False),
                seeds=eval_seeds,
                training=False,
                max_steps=args.max_steps,
                render=args.render,
            )

        summary = summarize_logs(logs)
        if agent_name == "qlearning":
            summary["training_episodes"] = args.train_episodes
        if args.record_video:
            summary["video_folder"] = str(video_root)
            summary["recorded_episodes"] = record_count

        save_episode_logs(logs, agent_output)
        save_summary(summary, agent_output)
        results[agent_name] = summary

        print(f"\n[{agent_name}]")
        print(json.dumps(summary, indent=2))

    if len(results) > 1:
        write_plot_data(results, output_root)


if __name__ == "__main__":
    main()
