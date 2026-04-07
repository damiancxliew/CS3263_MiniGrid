from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSET_ROOT = ROOT / "data" / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Grid Universe gameplay level to an MP4 video."
    )
    parser.add_argument(
        "--level",
        default="build_level_required_two",
        help="Level builder function name from gameplay_levels.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to store on the initial state/environment.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=480,
        help="TextureRenderer output resolution.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Defaults to results/videos/<level>_seed<seed>.mp4",
    )
    parser.add_argument(
        "--observation-type",
        default="image",
        choices=["image", "level"],
        help="Observation type passed to GridUniverseEnv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = None

    if InconsistentVersionWarning is not None:
        # The embedded sklearn pipeline in final.py may have been serialized by a
        # nearby sklearn version. For video rendering, we only need the agent to
        # run; keep the warning from drowning out the actual export result.
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    try:
        from final import Agent
        from gameplay_levels import __dict__ as gameplay_namespace
        from grid_universe.gym_env import GridUniverseEnv
        from grid_universe.renderer.texture import TextureRenderer
    except ModuleNotFoundError as exc:
        print(
            "Missing dependency while trying to render video:",
            exc,
            file=sys.stderr,
        )
        print(
            "This script needs at least `grid_universe` and `torch` installed "
            "in the current Python environment.",
            file=sys.stderr,
        )
        return 1

    builder = gameplay_namespace.get(args.level)
    if builder is None or not callable(builder):
        print(f"Unknown level builder: {args.level}", file=sys.stderr)
        return 2

    sample_state = builder(seed=args.seed)

    def initial_state_fn(*_args, **_kwargs):
        state = builder(seed=args.seed)
        return replace(state, seed=args.seed)

    env = GridUniverseEnv(
        initial_state_fn=initial_state_fn,
        width=sample_state.width,
        height=sample_state.height,
        render_asset_root=str(ASSET_ROOT),
        observation_type=args.observation_type,
    )

    renderer = TextureRenderer(resolution=args.resolution)
    agent = Agent()

    state, _ = env.reset()
    if env.state is None:
        print("Environment failed to initialize state.", file=sys.stderr)
        return 3

    frames = [np.array(renderer.render(env.state).convert("RGB"))]

    done = False
    steps = 0
    while not done:
        action = agent.step(state)
        state, _reward, win, lose, _info = env.step(action)
        if env.state is None:
            print("Environment returned an empty state after stepping.", file=sys.stderr)
            return 4
        frames.append(np.array(renderer.render(env.state).convert("RGB")))
        steps += 1
        done = win or lose

    output_path = (
        Path(args.output)
        if args.output is not None
        else ROOT / "results" / "videos" / f"{args.level}_seed{args.seed}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip = ImageSequenceClip(frames, fps=args.fps)
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=False,
        logger=None,
    )

    print(f"Saved {len(frames)} frames from {steps} steps to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
