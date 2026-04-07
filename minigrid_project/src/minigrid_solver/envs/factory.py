from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def make_env(
    env_name: str,
    render: bool = False,
    record_video: bool = False,
    video_folder: str | None = None,
    video_prefix: str = "minigrid",
):
    render_mode = "human" if render else ("rgb_array" if record_video else None)
    env = gym.make(env_name, render_mode=render_mode)

    if record_video:
        if video_folder is None:
            raise ValueError("video_folder must be provided when record_video is enabled.")
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True,
            name_prefix=video_prefix,
        )

    return env
