from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np

from minigrid_solver.domain import EpisodeLog
from minigrid_solver.perception import SymbolicStateExtractor


def run_episode(env, agent, seed: int, training: bool = False, max_steps: int = 300, render: bool = False) -> EpisodeLog:
    _, reset_info = env.reset(seed=seed)
    agent.reset()
    extractor = SymbolicStateExtractor()
    initial_state = extractor.extract(env)

    reward_total = 0.0
    step_count = 0
    terminated = False
    truncated = False
    step_traces = []
    start = time.perf_counter()

    while not (terminated or truncated) and step_count < max_steps:
        if render:
            env.render()

        if training and hasattr(agent, "act"):
            action, metadata = agent.act(env, step_count, training=True)
        else:
            action, metadata = agent.act(env, step_count)

        pre_state = extractor.extract(env)
        _, reward, terminated, truncated, _ = env.step(action)
        reward_total += reward
        next_state = extractor.extract(env)
        if hasattr(agent, "shaped_reward"):
            shaped_reward, shaping_terms = agent.shaped_reward(pre_state, next_state, reward, terminated or truncated)
            metadata = {**metadata, "shaped_reward": shaped_reward, "reward_terms": shaping_terms}
        step_trace = agent.observe_transition(
            env=env,
            t=step_count,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metadata=metadata,
        )
        step_traces.append(step_trace)

        if training and hasattr(agent, "update"):
            agent.update(pre_state, action, reward, next_state, terminated or truncated)

        step_count += 1

    if training and hasattr(agent, "decay_epsilon"):
        agent.decay_epsilon()

    final_state = extractor.extract(env)
    elapsed_sec = time.perf_counter() - start
    env_name = env.spec.id if env.spec is not None else "unknown-env"
    seed_used = int(reset_info.get("seed", seed))

    episode_log = EpisodeLog(
        agent_name=agent.name,
        env_name=env_name,
        seed=seed_used,
        solved=bool(terminated),
        reward=float(reward_total),
        steps=step_count,
        elapsed_sec=elapsed_sec,
        planner_expansions=int(getattr(agent, "total_expanded_nodes", 0)),
        planner_generated=int(getattr(agent, "total_generated_nodes", 0)),
        replan_count=int(getattr(agent, "replan_count", 0)),
        summary_reasoning=list(getattr(agent, "reasoning_summary", [])),
        initial_abstract_state=initial_state.to_dict(),
        final_abstract_state=final_state.to_dict(),
        plan_history=list(getattr(agent, "plan_history", [])),
        step_traces=step_traces,
        extra_metrics={
            "training": training,
            "max_steps": max_steps,
            "epsilon": float(getattr(agent, "epsilon", 0.0)) if hasattr(agent, "epsilon") else None,
            "q_table_size": int(len(getattr(agent, "q_table", {}))) if hasattr(agent, "q_table") else None,
            "shaped_reward_total": float(getattr(agent, "last_shaped_reward_total", 0.0))
            if hasattr(agent, "last_shaped_reward_total")
            else None,
        },
    )
    if hasattr(agent, "end_episode"):
        agent.end_episode(episode_log, training=training)
    return episode_log


def evaluate_agent(
    agent,
    env_factory,
    seeds: Sequence[int],
    training: bool = False,
    max_steps: int = 300,
    render: bool = False,
) -> List[EpisodeLog]:
    logs: List[EpisodeLog] = []
    for seed in seeds:
        env = env_factory()
        try:
            logs.append(run_episode(env, agent, seed=seed, training=training, max_steps=max_steps, render=render))
        finally:
            env.close()
    return logs


def summarize_logs(logs: Sequence[EpisodeLog]) -> dict:
    if not logs:
        return {}

    successes = sum(1 for log in logs if log.solved)
    rewards = [log.reward for log in logs]
    lengths = [log.steps for log in logs]
    expansions = [log.planner_expansions for log in logs]
    replans = [log.replan_count for log in logs]
    elapsed = [log.elapsed_sec for log in logs]
    shaped_rewards = [float(log.extra_metrics.get("shaped_reward_total", 0.0) or 0.0) for log in logs]

    return {
        "episodes": len(logs),
        "agent_name": logs[0].agent_name,
        "env_name": logs[0].env_name,
        "success_rate": successes / len(logs),
        "average_reward": float(np.mean(rewards)),
        "average_episode_length": float(np.mean(lengths)),
        "average_planner_expansions": float(np.mean(expansions)),
        "average_replans": float(np.mean(replans)),
        "average_elapsed_sec": float(np.mean(elapsed)),
        "average_shaped_reward": float(np.mean(shaped_rewards)),
    }
