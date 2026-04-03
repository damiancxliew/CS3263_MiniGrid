from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from minigrid_solver.domain import (
    AbstractState,
    DIR_TO_VEC,
    DoorFact,
    KeyFact,
    PrimitiveAction,
)


class PlannerModel:
    """Deterministic planning model with STRIPS-like preconditions and effects."""

    @staticmethod
    def in_bounds(state: AbstractState, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < state.width and 0 <= y < state.height

    @staticmethod
    def cell_in_front(state: AbstractState) -> Tuple[int, int]:
        dx, dy = DIR_TO_VEC[state.agent_dir]
        x, y = state.agent_pos
        return (x + dx, y + dy)

    @staticmethod
    def door_at(state: AbstractState, pos: Tuple[int, int]) -> Optional[DoorFact]:
        return next((door for door in state.doors if door.pos == pos), None)

    @staticmethod
    def key_at(state: AbstractState, pos: Tuple[int, int]) -> Optional[KeyFact]:
        return next((key for key in state.keys if key.pos == pos), None)

    @staticmethod
    def is_blocked(state: AbstractState, pos: Tuple[int, int]) -> bool:
        if not PlannerModel.in_bounds(state, pos):
            return True
        if pos in state.walls:
            return True
        door = PlannerModel.door_at(state, pos)
        return bool(door is not None and not door.is_open)

    @staticmethod
    def state_signature(state: AbstractState) -> Tuple[object, ...]:
        """Compact factored hash for revisit-aware search and RL."""
        door_facts = tuple((door.pos, door.color, door.is_open, door.is_locked) for door in state.doors)
        key_facts = tuple((key.pos, key.color) for key in state.keys)
        return (
            state.agent_pos,
            state.agent_dir,
            state.carrying_key,
            door_facts,
            key_facts,
            state.goal_pos,
        )

    @staticmethod
    def subgoal_description(state: AbstractState) -> str:
        locked_doors = [door for door in state.doors if door.is_locked and not door.is_open]
        if locked_doors and state.carrying_key is None and state.keys:
            key = min(
                state.keys,
                key=lambda item: abs(state.agent_pos[0] - item.pos[0]) + abs(state.agent_pos[1] - item.pos[1]),
            )
            return f"reach key at {key.pos}"
        if locked_doors and state.carrying_key is not None:
            target_door = next((door for door in locked_doors if door.color == state.carrying_key), locked_doors[0])
            return f"unlock door at {target_door.pos}"
        if any(not door.is_open for door in state.doors):
            target_door = next(door for door in state.doors if not door.is_open)
            return f"open door at {target_door.pos}"
        return f"reach goal at {state.goal_pos}"

    @staticmethod
    def valid_actions(state: AbstractState) -> List[PrimitiveAction]:
        """Return the actions whose preconditions hold in the current symbolic state."""
        valid: List[PrimitiveAction] = [PrimitiveAction.LEFT, PrimitiveAction.RIGHT]
        front = PlannerModel.cell_in_front(state)
        if not PlannerModel.is_blocked(state, front):
            valid.append(PrimitiveAction.FORWARD)

        if PlannerModel.key_at(state, front) is not None and state.carrying_key is None:
            valid.append(PrimitiveAction.PICKUP)

        door_ahead = PlannerModel.door_at(state, front)
        if door_ahead is not None:
            can_unlock = state.carrying_key == door_ahead.color
            if (not door_ahead.is_locked) or can_unlock:
                valid.append(PrimitiveAction.TOGGLE)

        return valid

    @staticmethod
    def transition(
        state: AbstractState,
        action: PrimitiveAction,
    ) -> Optional[Tuple[AbstractState, str]]:
        """Apply one symbolic operator, mirroring the GridUniverse search style."""

        subgoal = PlannerModel.subgoal_description(state)
        front = PlannerModel.cell_in_front(state)
        door_ahead = PlannerModel.door_at(state, front)
        key_ahead = PlannerModel.key_at(state, front)

        if action == PrimitiveAction.LEFT:
            return (
                AbstractState(**{**state.__dict__, "agent_dir": (state.agent_dir - 1) % 4}),
                f"Rotate left to align the agent with the current symbolic sub-goal: {subgoal}.",
            )

        if action == PrimitiveAction.RIGHT:
            return (
                AbstractState(**{**state.__dict__, "agent_dir": (state.agent_dir + 1) % 4}),
                f"Rotate right to align the agent with the current symbolic sub-goal: {subgoal}.",
            )

        if action == PrimitiveAction.FORWARD and not PlannerModel.is_blocked(state, front):
            return (
                AbstractState(**{**state.__dict__, "agent_pos": front}),
                f"Move forward into {front} to make progress toward the current symbolic sub-goal: {subgoal}.",
            )

        if action == PrimitiveAction.PICKUP and key_ahead is not None and state.carrying_key is None:
            remaining_keys = tuple(key for key in state.keys if key.pos != key_ahead.pos)
            return (
                AbstractState(
                    **{
                        **state.__dict__,
                        "carrying_key": key_ahead.color,
                        "keys": remaining_keys,
                    }
                ),
                f"Pick up the {key_ahead.color} key because the current sub-goal is to enable door unlocking.",
            )

        if action == PrimitiveAction.TOGGLE and door_ahead is not None:
            can_unlock = state.carrying_key == door_ahead.color
            if (not door_ahead.is_locked) or can_unlock:
                new_doors = []
                for door in state.doors:
                    if door.pos != door_ahead.pos:
                        new_doors.append(door)
                    else:
                        new_doors.append(
                            DoorFact(
                                pos=door.pos,
                                color=door.color,
                                is_open=True,
                                is_locked=False,
                            )
                        )
                reason = (
                    f"Toggle the {door_ahead.color} door to open the path toward {state.goal_pos}."
                    if not door_ahead.is_locked
                    else f"Use the carried {door_ahead.color} key to unlock and open the door as the current sub-goal."
                )
                return (
                    AbstractState(
                        **{**state.__dict__, "doors": tuple(sorted(new_doors, key=lambda door: door.pos))}
                    ),
                    reason,
                )

        return None

    @staticmethod
    def successors(state: AbstractState) -> List[Tuple[PrimitiveAction, AbstractState, str]]:
        successors: List[Tuple[PrimitiveAction, AbstractState, str]] = []
        for action in PlannerModel.valid_actions(state):
            result = PlannerModel.transition(state, action)
            if result is None:
                continue
            next_state, rationale = result
            successors.append((action, next_state, rationale))
        return successors
