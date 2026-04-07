from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class PrimitiveAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6


DIR_TO_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


DIR_NAME = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}


@dataclass(frozen=True)
class DoorFact:
    pos: Tuple[int, int]
    color: str
    is_open: bool
    is_locked: bool


@dataclass(frozen=True)
class KeyFact:
    pos: Tuple[int, int]
    color: str


@dataclass(frozen=True)
class AbstractState:
    """Factored symbolic state used by both planning and RL.

    This mirrors course ideas from knowledge representation and MDPs:
    rather than treat the environment as one opaque observation, we expose a
    structured state made of agent, key, door, goal, and wall facts.
    """

    width: int
    height: int
    walls: frozenset[Tuple[int, int]]
    goal_pos: Tuple[int, int]
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying_key: Optional[str]
    doors: Tuple[DoorFact, ...]
    keys: Tuple[KeyFact, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "walls": sorted(self.walls),
            "goal_pos": self.goal_pos,
            "agent_pos": self.agent_pos,
            "agent_dir": self.agent_dir,
            "agent_dir_name": DIR_NAME[self.agent_dir],
            "carrying_key": self.carrying_key,
            "doors": [asdict(door) for door in self.doors],
            "keys": [asdict(key) for key in self.keys],
        }


@dataclass
class PlanStep:
    action: PrimitiveAction
    rationale: str
    predicted_state: Optional[Dict[str, object]] = None


@dataclass
class SearchTrace:
    expanded_nodes: int = 0
    generated_nodes: int = 0
    plan_cost: Optional[int] = None
    heuristic_at_start: Optional[int] = None
    chosen_strategy: str = "A*"
    notes: List[str] = field(default_factory=list)


@dataclass
class StepTrace:
    t: int
    action: str
    rationale: str
    symbolic_state: Dict[str, object]
    reward: float
    terminated: bool
    truncated: bool
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    agent_name: str
    env_name: str
    seed: int
    solved: bool
    reward: float
    steps: int
    elapsed_sec: float
    planner_expansions: int
    planner_generated: int
    replan_count: int
    summary_reasoning: List[str]
    initial_abstract_state: Dict[str, object]
    final_abstract_state: Optional[Dict[str, object]]
    plan_history: List[Dict[str, object]]
    step_traces: List[StepTrace]
    extra_metrics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["step_traces"] = [asdict(step) for step in self.step_traces]
        return payload
