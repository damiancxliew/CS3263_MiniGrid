from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from minigrid_solver.domain import AbstractState, EpisodeLog, PrimitiveAction, StepTrace
from minigrid_solver.perception import SymbolicStateExtractor
from minigrid_solver.planning.symbolic_model import PlannerModel


class QLearningAgent:
    """Lightweight RL baseline over the same factored abstraction."""

    name = "tabular_q_learning"

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        revisit_penalty: float = 0.02,
        key_bonus: float = 0.3,
        door_bonus: float = 0.5,
        distance_bonus: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.revisit_penalty = revisit_penalty
        self.key_bonus = key_bonus
        self.door_bonus = door_bonus
        self.distance_bonus = distance_bonus
        self.extractor = SymbolicStateExtractor()
        self.n_actions = len(PrimitiveAction)
        self.q_table: Dict[Tuple[object, ...], np.ndarray] = {}
        self.reasoning_summary: List[str] = []
        self.plan_history: List[Dict[str, object]] = []
        self.total_expanded_nodes = 0
        self.total_generated_nodes = 0
        self.replan_count = 0
        self.state_visits: Dict[Tuple[object, ...], int] = {}
        self.training_history: List[Dict[str, object]] = []
        self.last_shaped_reward_total = 0.0

    def reset(self) -> None:
        self.reasoning_summary = []
        self.plan_history = []
        self.total_expanded_nodes = 0
        self.total_generated_nodes = 0
        self.replan_count = 0
        self.state_visits = {}
        self.last_shaped_reward_total = 0.0

    def encode(self, state: AbstractState) -> Tuple[object, ...]:
        return PlannerModel.state_signature(state)

    def q_values(self, encoded_state: Tuple[object, ...]) -> np.ndarray:
        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[encoded_state]

    def _select_best_valid_action(self, q_values: np.ndarray, valid_indices: List[int]) -> int:
        """Break ties among equally valued actions without defaulting to LEFT forever.

        For unseen states, tabular Q-values are often all zero. Plain argmax would always
        pick the first valid action, which creates spin-in-place behavior because LEFT has
        the smallest action index. We instead bias ties toward progress-making operators.
        """

        if not valid_indices:
            return int(PrimitiveAction.LEFT)

        valid_scores = q_values[valid_indices]
        best_score = float(np.max(valid_scores))
        best_indices = [index for index in valid_indices if float(q_values[index]) == best_score]

        tie_break_order = [
            int(PrimitiveAction.PICKUP),
            int(PrimitiveAction.TOGGLE),
            int(PrimitiveAction.FORWARD),
            int(PrimitiveAction.RIGHT),
            int(PrimitiveAction.LEFT),
        ]
        for preferred in tie_break_order:
            if preferred in best_indices:
                return preferred
        return best_indices[0]

    def _distance_to_subgoal(self, state: AbstractState) -> int:
        locked_doors = [door for door in state.doors if door.is_locked and not door.is_open]
        if locked_doors and state.carrying_key is None and state.keys:
            target = min(
                state.keys,
                key=lambda key: abs(state.agent_pos[0] - key.pos[0]) + abs(state.agent_pos[1] - key.pos[1]),
            )
            return abs(state.agent_pos[0] - target.pos[0]) + abs(state.agent_pos[1] - target.pos[1])
        if locked_doors and state.carrying_key is not None:
            target = next((door for door in locked_doors if door.color == state.carrying_key), locked_doors[0])
            return abs(state.agent_pos[0] - target.pos[0]) + abs(state.agent_pos[1] - target.pos[1])
        return abs(state.agent_pos[0] - state.goal_pos[0]) + abs(state.agent_pos[1] - state.goal_pos[1])

    def shaped_reward(self, state: AbstractState, next_state: AbstractState, env_reward: float, terminated: bool) -> Tuple[float, Dict[str, float]]:
        shaping = {
            "env_reward": float(env_reward),
            "distance_progress": 0.0,
            "key_bonus": 0.0,
            "door_bonus": 0.0,
            "revisit_penalty": 0.0,
            "terminal_bonus": 0.0,
        }

        prev_distance = self._distance_to_subgoal(state)
        next_distance = self._distance_to_subgoal(next_state)
        if next_distance < prev_distance:
            shaping["distance_progress"] = self.distance_bonus * float(prev_distance - next_distance)

        if state.carrying_key is None and next_state.carrying_key is not None:
            shaping["key_bonus"] = self.key_bonus

        opened_before = {(door.pos, door.is_open, door.is_locked) for door in state.doors}
        opened_after = {(door.pos, door.is_open, door.is_locked) for door in next_state.doors}
        if opened_before != opened_after and any(door.is_open for door in next_state.doors):
            shaping["door_bonus"] = self.door_bonus

        next_signature = self.encode(next_state)
        visit_count = self.state_visits.get(next_signature, 0)
        shaping["revisit_penalty"] = -self.revisit_penalty * float(visit_count)

        if terminated:
            shaping["terminal_bonus"] = 1.0

        total = float(sum(shaping.values()))
        return total, shaping

    def act(self, env, t: int, training: bool = False) -> Tuple[int, Dict[str, object]]:
        state = self.extractor.extract(env)
        encoded = self.encode(state)
        self.state_visits[encoded] = self.state_visits.get(encoded, 0) + 1
        q_values = self.q_values(encoded)
        valid_actions = PlannerModel.valid_actions(state)
        valid_indices = [int(action) for action in valid_actions]
        masked_values = np.full(self.n_actions, -1e9, dtype=np.float32)
        masked_values[valid_indices] = q_values[valid_indices]
        explore = training and random.random() < self.epsilon
        if explore:
            action = random.choice(valid_indices)
            rationale = (
                f"Exploration step under epsilon-greedy control with epsilon={self.epsilon:.3f}, "
                "sampled only from symbolically valid actions."
            )
        else:
            action = self._select_best_valid_action(masked_values, valid_indices)
            rationale = (
                "Choose the highest-value action among the currently valid symbolic operators, "
                "using a progress-oriented tie-breaker so unseen states do not degenerate into spin loops."
            )

        return action, {
            "policy_type": "q_learning",
            "action_name": PrimitiveAction(action).name,
            "rationale": rationale,
            "symbolic_subgoal": PlannerModel.subgoal_description(state),
            "symbolic_state": state.to_dict(),
            "q_values": {PrimitiveAction(idx).name: float(value) for idx, value in enumerate(q_values)},
            "valid_actions": [PrimitiveAction(idx).name for idx in valid_indices],
            "epsilon": self.epsilon,
            "training": training,
        }

    def update(
        self,
        state: AbstractState,
        action: int,
        reward: float,
        next_state: AbstractState,
        terminated: bool,
    ) -> None:
        encoded_state = self.encode(state)
        encoded_next = self.encode(next_state)
        q_state = self.q_values(encoded_state)
        q_next = self.q_values(encoded_next)
        shaped_reward, _ = self.shaped_reward(state, next_state, reward, terminated)
        self.last_shaped_reward_total += shaped_reward
        valid_next_actions = PlannerModel.valid_actions(next_state)
        next_indices = [int(item) for item in valid_next_actions]
        next_max = float(np.max(q_next[next_indices])) if next_indices else 0.0
        target = shaped_reward if terminated else shaped_reward + self.gamma * next_max
        q_state[action] += self.alpha * (target - q_state[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self, log: EpisodeLog, training: bool) -> None:
        if not training:
            return
        self.training_history.append(
            {
                "episode_index": len(self.training_history) + 1,
                "reward": log.reward,
                "shaped_reward": self.last_shaped_reward_total,
                "solved": log.solved,
                "steps": log.steps,
                "epsilon": self.epsilon,
            }
        )

    def observe_transition(
        self,
        env,
        t: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        metadata: Dict[str, object],
    ) -> StepTrace:
        next_state = self.extractor.extract(env)
        return StepTrace(
            t=t,
            action=PrimitiveAction(action).name,
            rationale=str(metadata["rationale"]),
            symbolic_state=next_state.to_dict(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metadata=metadata,
        )
