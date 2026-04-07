from __future__ import annotations

from typing import List, Optional, Set, Tuple

from minigrid.core.world_object import Door, Goal, Key, Wall

from minigrid_solver.domain import AbstractState, DoorFact, KeyFact


class SymbolicStateExtractor:
    """Phase 1 abstraction layer.

    This currently reads the underlying MiniGrid state directly to produce a
    factored symbolic representation. The module boundary makes it easy to swap
    in a learned perception model in Phase 2.
    """

    @staticmethod
    def extract(env) -> AbstractState:
        base_env = env.unwrapped
        width, height = base_env.width, base_env.height
        walls: Set[Tuple[int, int]] = set()
        doors: List[DoorFact] = []
        keys: List[KeyFact] = []
        goal_pos: Optional[Tuple[int, int]] = None

        for x in range(width):
            for y in range(height):
                obj = base_env.grid.get(x, y)
                if obj is None:
                    continue
                if isinstance(obj, Wall):
                    walls.add((x, y))
                elif isinstance(obj, Door):
                    doors.append(
                        DoorFact(
                            pos=(x, y),
                            color=getattr(obj, "color", "unknown"),
                            is_open=bool(obj.is_open),
                            is_locked=bool(obj.is_locked),
                        )
                    )
                elif isinstance(obj, Key):
                    keys.append(KeyFact(pos=(x, y), color=getattr(obj, "color", "unknown")))
                elif isinstance(obj, Goal):
                    goal_pos = (x, y)

        if goal_pos is None:
            raise ValueError("The environment does not expose a goal position.")

        carrying_key = None
        if base_env.carrying is not None and isinstance(base_env.carrying, Key):
            carrying_key = getattr(base_env.carrying, "color", "unknown")

        return AbstractState(
            width=width,
            height=height,
            walls=frozenset(walls),
            goal_pos=goal_pos,
            agent_pos=tuple(base_env.agent_pos),
            agent_dir=int(base_env.agent_dir),
            carrying_key=carrying_key,
            doors=tuple(sorted(doors, key=lambda fact: fact.pos)),
            keys=tuple(sorted(keys, key=lambda fact: fact.pos)),
        )
